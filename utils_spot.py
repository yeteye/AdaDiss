"""
utils_spot.py — 转录本级别数据流水线（不依赖细胞分割）

v3 核心修复（按老师建议逐一落实）
====================================

[Fix 1] PCA 特征对齐（最关键，消除域偏移）
  问题：Xenium spot 文库大小约 20-50，scRNA 约 10000
        log1p(count/20 * 1e4) = log1p(500) ≈ 6.2
        log1p(count/10000 * 1e4) = log1p(1)  ≈ 0.69
        → 同一基因在两种模态间差 10x，StandardScaler 无法修复（OOD）
  修复：只在 scRNA 上 fit PCA，再把 Xenium 投影到同一 PCA 空间
        节点特征维度：n_genes(453) → n_components(50)

[Fix 2] 类别权重上限（防止 400x 崩溃）
  问题：compute_class_weight("balanced") 在 400:1 不平衡下产生 400x 权重
        模型疯狂预测稀有类，肿瘤细胞一致性只有 0.4%
  修复：weights / mean 归一化后，clip 到 max_weight_multiplier=5

[Fix 3] 聚合半径修复（Bug 修复）
  问题：aggregate_spot_to_cell 硬编码 radius_um=10，但 bin_size=15
        → 许多细胞在 10μm 内找不到任何 spot，强制使用不可靠的最近邻回退
  修复：radius_um 默认 = bin_size * 1.5（传入 bin_size 参数）

[Fix 4] Xenium 特征裁剪（减少极端值影响）
  修复：PCA 前将 Xenium log-norm 裁剪到 scRNA 第 99 百分位
"""

import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


# ══════════════════════════════════════════════════════════════════
# 1. 加载 Xenium 原始转录本
# ══════════════════════════════════════════════════════════════════

def load_xenium_transcripts(
    transcript_path: str,
    gene_list: list,
    qv_threshold: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """加载并过滤 Xenium 原始转录本文件。"""
    if verbose:
        print(f"加载转录本文件: {transcript_path}")

    if transcript_path.endswith(".parquet"):
        df = pd.read_parquet(transcript_path)
    else:
        df = pd.read_csv(transcript_path)

    if verbose:
        print(f"  原始转录本数: {len(df):,}")
        print(f"  列名: {list(df.columns)}")

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("qv", "q_value") or "quality" in lc:
            col_map[c] = "qv"
        elif lc in ("feature_name", "gene_name", "gene"):
            col_map[c] = "gene"
        elif lc in ("x_location", "x_coord", "x"):
            col_map[c] = "x_raw"
        elif lc in ("y_location", "y_coord", "y"):
            col_map[c] = "y_raw"
    df = df.rename(columns=col_map)

    if "qv" in df.columns:
        df = df[df["qv"] >= qv_threshold].copy()
        if verbose:
            print(f"  Q>={qv_threshold} 过滤后: {len(df):,}")

    gene_set = set(gene_list)
    df = df[df["gene"].isin(gene_set)].copy()
    if verbose:
        print(f"  基因对齐后: {len(df):,} 条转录本，{df['gene'].nunique()} 个基因")

    df["x"] = df["x_raw"].round(0).astype(int)
    df["y"] = df["y_raw"].round(0).astype(int)
    return df[["gene", "x", "y"]].copy()


# ══════════════════════════════════════════════════════════════════
# 2. 聚合转录本 -> spot 表达矩阵
# ══════════════════════════════════════════════════════════════════

def bin_transcripts_to_spots(
    df: pd.DataFrame,
    gene_list: list,
    bin_size: int = 10,
    min_transcripts: int = 1,
    verbose: bool = True,
) -> tuple:
    """将转录本聚合为 spot 级别的表达矩阵（返回原始 counts，尚未 log 归一化）。"""
    df = df.copy()
    df["bx"] = (df["x"] // bin_size) * bin_size + bin_size // 2
    df["by"] = (df["y"] // bin_size) * bin_size + bin_size // 2

    gene2idx = {g: i for i, g in enumerate(gene_list)}
    df["gene_idx"] = df["gene"].map(gene2idx)
    df = df.dropna(subset=["gene_idx"])
    df["gene_idx"] = df["gene_idx"].astype(int)

    spot_key = df[["bx", "by"]].drop_duplicates().reset_index(drop=True)
    spot_key["spot_id"] = spot_key.index
    df = df.merge(spot_key, on=["bx", "by"])

    n_spots = len(spot_key)
    n_genes = len(gene_list)

    rows = df["spot_id"].values
    cols = df["gene_idx"].values
    spot_matrix = sp.coo_matrix(
        (np.ones(len(df), dtype=np.float32), (rows, cols)),
        shape=(n_spots, n_genes),
    ).tocsr()

    total = np.array(spot_matrix.sum(axis=1)).ravel()
    keep  = total >= min_transcripts
    spot_matrix  = spot_matrix[keep]
    coords_keep  = spot_key[keep][["bx", "by"]].values.astype(np.float32)

    spot_expr   = spot_matrix.toarray().astype(np.float32)
    spot_coords = coords_keep

    if verbose:
        print(f"  bin_size={bin_size}um -> {spot_expr.shape[0]:,} 个有效 spot")
        print(f"  特征维度: {spot_expr.shape[1]}")
        print(f"  每 spot 平均转录本数: {spot_expr.sum(axis=1).mean():.1f}")

    return spot_expr, spot_coords


# ══════════════════════════════════════════════════════════════════
# 3. 归一化
# ══════════════════════════════════════════════════════════════════

def log_normalize(X: np.ndarray, scale_factor: float = 1e4) -> np.ndarray:
    """
    Library-size normalization + log1p（等价于 Seurat LogNormalize）。
    注意：Xenium spot 文库小（20-50），scale_factor=1e4 会大幅放大单个计数。
    这是域偏移的根源之一——后续通过 PCA 对齐解决。
    """
    total = X.sum(axis=1, keepdims=True).clip(min=1)
    return np.log1p(X / total * scale_factor).astype(np.float32)


def unified_normalize_spot(
    scrna_expr: np.ndarray,
    spot_expr:  np.ndarray,
) -> tuple:
    """
    统一归一化（兼容旧接口，主要供 TopACT SVM 使用）。
    GNN 训练请使用 pca_align_features() 获取更好的特征对齐。
    """
    scrna_log = log_normalize(scrna_expr)
    spot_log  = log_normalize(spot_expr)

    scaler     = StandardScaler()
    scrna_norm = scaler.fit_transform(scrna_log).astype(np.float32)
    spot_norm  = scaler.transform(spot_log).astype(np.float32)

    return scrna_norm, spot_norm, scaler


# ══════════════════════════════════════════════════════════════════
# 4. [Fix 1] PCA 特征对齐（核心域偏移修复）
# ══════════════════════════════════════════════════════════════════

def pca_align_features(
    scrna_lognorm: np.ndarray,
    xenium_lognorm: np.ndarray,
    n_components: int = 50,
    clip_percentile: float = 99.0,
    verbose: bool = True,
) -> tuple:
    """
    基于 PCA 的跨模态特征对齐（老师建议的核心修复方案）。

    解决问题
    --------
    scRNA 单细胞文库大小 ~10000，Xenium spot 文库 ~20-50。
    同一基因在 Xenium 中经 log1p(count/20 * 1e4) 值远大于 scRNA。
    StandardScaler 在 scRNA 上拟合后，无法正确对齐 Xenium 特征（OOD）。

    修复方案（老师建议）
    --------------------
    1. 将 Xenium log-norm 裁剪到 scRNA 第99百分位（消除放大噪声）
    2. 只在 scRNA 上 fit PCA（n_components=50）
    3. 将 Xenium 投影到同一 PCA 空间
    4. 结果：453-dim → 50-dim，两种模态在同一参考系中

    Parameters
    ----------
    scrna_lognorm  : (n_scrna, n_genes) 已 log_normalize 的 scRNA 表达
    xenium_lognorm : (n_spots, n_genes) 已 log_normalize 的 Xenium 表达
    n_components   : PCA 保留主成分数
    clip_percentile: Xenium 裁剪百分位（相对 scRNA 分布）

    Returns
    -------
    scrna_pca  : (n_scrna, n_components) PCA 投影特征
    xenium_pca : (n_spots, n_components) PCA 投影特征
    pca_model  : 已拟合的 sklearn PCA 对象（供后续使用/保存）
    """
    # ── Step 1: 裁剪 Xenium 极端值（[Fix 4]）─────────────────────
    if clip_percentile is not None:
        clip_vals = np.percentile(scrna_lognorm, clip_percentile, axis=0)
        before_sum = (xenium_lognorm > clip_vals).sum()
        xenium_clipped = np.clip(xenium_lognorm, None, clip_vals)
        if verbose:
            pct = 100.0 * before_sum / xenium_lognorm.size
            print(f"    [PCA] Xenium 裁剪: {before_sum:,} 个值 ({pct:.2f}%) 超过 scRNA {clip_percentile}th pct")
    else:
        xenium_clipped = xenium_lognorm.copy()

    # ── Step 2: 只在 scRNA 上拟合 PCA ─────────────────────────────
    n_comp = min(n_components, min(scrna_lognorm.shape) - 1)
    if verbose:
        print(f"    [PCA] 在 {scrna_lognorm.shape[0]:,} 个 scRNA 细胞上拟合 PCA (n_components={n_comp})")

    t0  = time.time()
    pca = PCA(n_components=n_comp, random_state=42)
    scrna_pca = pca.fit_transform(scrna_lognorm)  # ← 只在 scRNA 上 fit

    explained = pca.explained_variance_ratio_.sum()
    if verbose:
        print(f"    [PCA] 解释方差比: {explained:.3f}  ({time.time()-t0:.1f}s)")

    # ── Step 3: 将 Xenium 投影到相同 PCA 空间 ─────────────────────
    xenium_pca = pca.transform(xenium_clipped)
    if verbose:
        print(f"    [PCA] 输入维度: {scrna_lognorm.shape[1]} → 输出: {n_comp}")
        print(f"    [PCA] scRNA PCA 范围: [{scrna_pca.min():.2f}, {scrna_pca.max():.2f}]")
        print(f"    [PCA] Xenium PCA 范围: [{xenium_pca.min():.2f}, {xenium_pca.max():.2f}]")

    return (
        scrna_pca.astype(np.float32),
        xenium_pca.astype(np.float32),
        pca,
    )


# ══════════════════════════════════════════════════════════════════
# 5. [Fix 2] 上限类别权重计算
# ══════════════════════════════════════════════════════════════════

def compute_capped_class_weights(
    labels: np.ndarray,
    max_weight_multiplier: float = 5.0,
    verbose: bool = True,
) -> torch.Tensor:
    """
    计算类别权重，并对极端不平衡情况进行上限限制。

    问题
    ----
    compute_class_weight("balanced") 在 400:1 不平衡下产生 400x 权重，
    导致模型极度偏向稀有类型，主要肿瘤细胞的一致率跌至 0.4%。

    修复（老师建议）
    ----------------
    1. 计算 balanced 权重
    2. 归一化到均值 = 1
    3. clip 到 max_weight_multiplier（默认 5）

    Returns
    -------
    class_weights : (n_classes,) float tensor，上限为 max_weight_multiplier
    """
    classes = np.unique(labels)
    n_classes = int(labels.max()) + 1

    raw_weights = compute_class_weight("balanced", classes=classes, y=labels)

    # 归一化：均值 = 1
    normalized = raw_weights / raw_weights.mean()

    # 上限 clip（老师建议 5x）
    capped = np.clip(normalized, 0.0, max_weight_multiplier)

    if verbose:
        print(f"    [权重] 类别数: {len(classes)}")
        print(f"    [权重] 原始范围: [{raw_weights.min():.2f}, {raw_weights.max():.2f}]")
        print(f"    [权重] 归一化后范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
        print(f"    [权重] Clip 后范围: [{capped.min():.2f}, {capped.max():.2f}]  (上限={max_weight_multiplier})")

    # 需要为所有类别填充（防止 classes 不连续）
    weight_vec = np.ones(n_classes, dtype=np.float32)
    for c, w in zip(classes, capped):
        weight_vec[int(c)] = float(w)

    return torch.from_numpy(weight_vec).float()


# ══════════════════════════════════════════════════════════════════
# 6. 快速 kNN 内部工具
# ══════════════════════════════════════════════════════════════════

def _fast_feature_knn(
    X: np.ndarray,
    k: int,
    n_pca: int = 50,
    verbose: bool = True,
) -> tuple:
    """
    快速特征 kNN：TruncatedSVD 降维 + kd_tree euclidean。
    注意：此处的 PCA 仅用于加速 kNN 计算，与 Fix 1 的特征对齐 PCA 不同。
    """
    n = X.shape[0]
    if n_pca is not None and X.shape[1] > n_pca:
        n_comp = min(n_pca, min(X.shape) - 1)
        if verbose:
            print(f"    TruncatedSVD: {X.shape[1]} dims -> {n_comp} dims ...")
        t0  = time.time()
        svd = TruncatedSVD(n_components=n_comp, random_state=42, n_iter=5)
        Xr  = svd.fit_transform(X).astype(np.float32)
        var = svd.explained_variance_ratio_.sum()
        if verbose:
            print(f"    SVD done ({time.time()-t0:.1f}s), explained var: {var:.3f}")
    else:
        Xr = X

    if verbose:
        print(f"    kd_tree kNN: k={k}, n={n:,}, dim={Xr.shape[1]} ...")
    t0   = time.time()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=-1)
    nbrs.fit(Xr)
    _, idx = nbrs.kneighbors(Xr)
    if verbose:
        print(f"    kNN done ({time.time()-t0:.1f}s), edges: {n * k:,}")

    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()
    return src, dst


def _spatial_knn(
    coords: np.ndarray,
    k: int,
    verbose: bool = True,
) -> tuple:
    """空间 kNN（2D 坐标，kd_tree 极快）。"""
    n = len(coords)
    if verbose:
        print(f"    spatial kd_tree: k={k}, n={n:,} ...")
    t0   = time.time()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=-1)
    nbrs.fit(coords)
    _, idx = nbrs.kneighbors(coords)
    if verbose:
        print(f"    spatial kNN done ({time.time()-t0:.1f}s)")
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()
    return src, dst


# ══════════════════════════════════════════════════════════════════
# 7. 构建联合图（核心，已优化 + PCA 特征对齐）
# ══════════════════════════════════════════════════════════════════

def build_spot_graph(
    scrna_norm:       np.ndarray,
    spot_norm:        np.ndarray,
    spot_coords:      np.ndarray,
    scrna_labels:     np.ndarray,
    k_feat:           int   = 15,
    k_spatial:        int   = 10,
    val_ratio:        float = 0.2,
    n_pca_components: int   = 50,
    max_weight_multiplier: float = 5.0,
    verbose:          bool  = True,
) -> tuple:
    """
    构建 scRNA 细胞 + Xenium spot 的联合异质图。

    v3 改动
    -------
    - scrna_norm / spot_norm 现在接受 PCA 对齐后的特征
      （请在调用此函数前先调用 pca_align_features()）
    - 类别权重使用 compute_capped_class_weights()（上限 max_weight_multiplier）
    - 内部 kNN 加速仍使用 TruncatedSVD（仅用于 kNN 计算，不改变图节点特征）

    Parameters
    ----------
    scrna_norm / spot_norm : 已对齐的特征（PCA 输出，50-dim）
    max_weight_multiplier  : 类别权重上限（老师建议 5）
    """
    t_total = time.time()

    n_scrna = scrna_norm.shape[0]
    n_spots = spot_norm.shape[0]
    n_total = n_scrna + n_spots

    if verbose:
        print(f"  scRNA 节点: {n_scrna:,}  |  spot 节点: {n_spots:,}")
        print(f"  总节点数  : {n_total:,}")
        feat_dim = scrna_norm.shape[1]
        mem_mb = n_total * feat_dim * 4 / 1024**2
        print(f"  特征维度  : {feat_dim}  内存: {mem_mb:.0f} MB")

    X_all = np.vstack([scrna_norm, spot_norm])

    # ── 特征 kNN（PCA 加速版，注意此处 n_pca 仅用于加速 kNN）─────
    if verbose:
        print(f"\n  [1/2] 特征 kNN (k={k_feat})")
    # 如果特征已是 PCA（50-dim），跳过再次 SVD
    _n_pca_for_knn = None if scrna_norm.shape[1] <= 50 else n_pca_components
    src_f, dst_f = _fast_feature_knn(X_all, k=k_feat,
                                      n_pca=_n_pca_for_knn, verbose=verbose)

    # ── 空间 kNN（仅 spot 节点）──────────────────────────────────
    if verbose:
        print(f"\n  [2/2] 空间 kNN (k={k_spatial})")
    src_s_loc, dst_s_loc = _spatial_knn(spot_coords, k=k_spatial, verbose=verbose)
    src_s = src_s_loc + n_scrna
    dst_s = dst_s_loc + n_scrna

    # ── 合并 + 双向 + 去重 + 去自环 ──────────────────────────────
    if verbose:
        print("\n  合并边集...")
    src_all = np.concatenate([src_f, dst_f, src_s, dst_s])
    dst_all = np.concatenate([dst_f, src_f, dst_s, src_s])
    edges   = np.stack([src_all, dst_all], axis=1)
    edges   = np.unique(edges, axis=0)
    edges   = edges[edges[:, 0] != edges[:, 1]]
    edge_index = torch.from_numpy(edges.T).long()

    if verbose:
        print(f"  总边数 (去重后): {edge_index.shape[1]:,}")

    # ── 标签向量（-1 = 无标签）────────────────────────────────────
    labels_all = np.full(n_total, -1, dtype=np.int64)
    labels_all[:n_scrna] = scrna_labels

    # ── 分层划分验证集 ────────────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    scrna_range = np.arange(n_scrna)
    train_local, val_local = next(sss.split(scrna_range, scrna_labels))
    train_idx = scrna_range[train_local]
    val_idx   = scrna_range[val_local]

    train_mask  = torch.zeros(n_total, dtype=torch.bool)
    val_mask    = torch.zeros(n_total, dtype=torch.bool)
    xenium_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[train_idx]  = True
    val_mask[val_idx]      = True
    xenium_mask[n_scrna:]  = True

    # ── [Fix 2] 上限类别权重 ──────────────────────────────────────
    if verbose:
        print(f"\n  计算类别权重（上限={max_weight_multiplier}）...")
    class_weights = compute_capped_class_weights(
        scrna_labels,
        max_weight_multiplier=max_weight_multiplier,
        verbose=verbose,
    )

    # ── 组装 PyG Data ─────────────────────────────────────────────
    data = Data(
        x           = torch.from_numpy(X_all).float(),
        edge_index  = edge_index,
        y           = torch.from_numpy(labels_all).long(),
        train_mask  = train_mask,
        val_mask    = val_mask,
        spot_mask   = xenium_mask,
    )
    data.xenium_mask = xenium_mask
    data.n_scrna = n_scrna
    data.n_spots = n_spots

    split_info = {
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "n_scrna":   n_scrna,
        "n_spots":   n_spots,
    }

    if verbose:
        elapsed = time.time() - t_total
        print(f"\n  图构建完成，总耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  训练: {train_mask.sum():,}  验证: {val_mask.sum():,}"
              f"  推断 spot: {xenium_mask.sum():,}")

    return data, class_weights, split_info


# ══════════════════════════════════════════════════════════════════
# 8. spot 预测 -> 细胞级别聚合（[Fix 3] 半径修复）
# ══════════════════════════════════════════════════════════════════

def aggregate_spot_to_cell(
    spot_proba:  np.ndarray,
    spot_coords: np.ndarray,
    cell_coords: np.ndarray,
    n_classes:   int,
    bin_size:    float = 15.0,
    radius_um:   float = None,   # None → 自动 = bin_size * 1.5
) -> tuple:
    """
    将 spot 预测概率聚合到细胞质心级别（仅用于最终评估）。

    [Fix 3] 半径修复
    ----------------
    原代码硬编码 radius_um=10.0，但 bin_size=15μm。
    当 spot 中心间距为 15μm 时，10μm 半径覆盖不全，
    导致部分细胞附近没有 spot，强制使用不可靠的最近邻回退。

    修复：radius_um = bin_size * 1.5（例如 15μm → 22.5μm）
         完整覆盖一个 bin 加边缘缓冲。

    Parameters
    ----------
    bin_size  : spot 聚合的 bin 大小（μm），用于自动计算半径
    radius_um : 手动指定半径（None = 自动计算）
    """
    from sklearn.neighbors import BallTree

    # [Fix 3] 自动计算半径
    if radius_um is None:
        radius_um = bin_size * 1.5

    tree    = BallTree(spot_coords, metric="euclidean")
    indices = tree.query_radius(cell_coords, r=radius_um)

    cell_proba = np.zeros((len(cell_coords), n_classes), dtype=np.float32)
    fallback_count = 0
    for i, nbr in enumerate(indices):
        if len(nbr) > 0:
            cell_proba[i] = spot_proba[nbr].mean(axis=0)
        else:
            # 最近邻回退（现在覆盖率更高，触发概率更低）
            _, nearest = tree.query(cell_coords[[i]], k=1)
            cell_proba[i] = spot_proba[nearest[0, 0]]
            fallback_count += 1

    if fallback_count > 0:
        pct = 100.0 * fallback_count / len(cell_coords)
        print(f"  [聚合] 使用最近邻回退的细胞: {fallback_count:,} ({pct:.1f}%)  "
              f"(radius={radius_um:.1f}μm, bin_size={bin_size}μm)")

    return cell_proba.argmax(axis=1), cell_proba.max(axis=1)

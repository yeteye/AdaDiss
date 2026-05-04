"""
utils_spot_v5.py — 融合版（PCA 对齐 + MNN 跨域边 + 跨域诊断）

融合策略
========
本版本融合两条修复路线：

  路线 A（老师方案 v3/v4）：PCA 投影对齐
    - scRNA 上 fit PCA(50)，Xenium 投影到同一基底
    - Xenium 投影前 99% clip 极端值
    - 节点特征 453 → 50 维
    → 修复"文库大小 333x 差导致的域偏移"

  路线 B（v2 MNN 方案）：显式跨域边
    - 在已对齐的特征空间里，构造 mutual nearest neighbors 跨域边
    - 打印"跨域边占比"作为图健康度可观察指标
    → 修复"vstack+kNN 实际跨域边稀少"问题

为什么必须两条都做
==================
单做 A：PCA 投影后两域共享坐标系，但分布不一定充分重叠；
        如果 Xenium 点云偏在 scRNA 一侧，kNN 仍"自己人找自己人"。

单做 B：MNN 在原始 453 维空间几乎找不到 mutual pair（域差异太大）；
        必须先 PCA 把分布拉近，再做 MNN 才有意义。

所以正确顺序是：log_norm → 99% clip → scRNA fit PCA → 投影 →
                  scRNA-内 kNN + spot-内 kNN + 跨域 MNN + 空间 kNN

向后兼容
========
- prepare_features_for_gnn() 接口与 v4 一致
- build_spot_graph() 新增 k_cross 参数，旧调用兼容（k_cross=10 默认）
- 返回 split_info 增加 cross_edge_pct 字段（用于诊断）
"""

import time
import warnings
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
# 1. 加载 / binning（与 v3/v4 一致）
# ══════════════════════════════════════════════════════════════════

def load_xenium_transcripts(transcript_path, gene_list, qv_threshold=20, verbose=True):
    if verbose:
        print(f"加载转录本文件: {transcript_path}")
    df = pd.read_parquet(transcript_path) if transcript_path.endswith(".parquet") \
         else pd.read_csv(transcript_path)
    if verbose:
        print(f"  原始转录本数: {len(df):,}")

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

    df = df[df["gene"].isin(set(gene_list))].copy()
    if verbose:
        print(f"  基因对齐后: {len(df):,} 条转录本，{df['gene'].nunique()} 个基因")

    df["x"] = df["x_raw"].round(0).astype(int)
    df["y"] = df["y_raw"].round(0).astype(int)
    return df[["gene", "x", "y"]].copy()


def bin_transcripts_to_spots(df, gene_list, bin_size=10, min_transcripts=1, verbose=True):
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

    n_spots, n_genes = len(spot_key), len(gene_list)
    spot_matrix = sp.coo_matrix(
        (np.ones(len(df), dtype=np.float32),
         (df["spot_id"].values, df["gene_idx"].values)),
        shape=(n_spots, n_genes),
    ).tocsr()

    keep = np.array(spot_matrix.sum(axis=1)).ravel() >= min_transcripts
    spot_matrix = spot_matrix[keep]
    spot_coords = spot_key[keep][["bx", "by"]].values.astype(np.float32)
    spot_expr = spot_matrix.toarray().astype(np.float32)

    if verbose:
        lib = spot_expr.sum(axis=1)
        print(f"  bin_size={bin_size}μm -> {spot_expr.shape[0]:,} 个有效 spot")
        print(f"  特征维度: {spot_expr.shape[1]}")
        print(f"  每 spot 平均转录本数: {lib.mean():.1f}  (中位 {np.median(lib):.1f})")
        print(f"  scRNA 典型 ~10000 → 比值 {10000/max(lib.mean(),1):.0f}x （域偏移源头）")

    return spot_expr, spot_coords


# ══════════════════════════════════════════════════════════════════
# 2. 归一化
# ══════════════════════════════════════════════════════════════════

def log_normalize(X, scale_factor=1e4):
    """Library-size normalization + log1p。"""
    total = X.sum(axis=1, keepdims=True).clip(min=1)
    return np.log1p(X / total * scale_factor).astype(np.float32)


def unified_normalize_spot(scrna_expr, spot_expr):
    """
    [兼容接口] 仅供 TopACT SVM 使用。
    GNN 训练应该使用 prepare_features_for_gnn()。
    """
    warnings.warn(
        "unified_normalize_spot() 无法修复域偏移；GNN 训练请使用 "
        "prepare_features_for_gnn()。本函数仅供 TopACT SVM 使用。",
        UserWarning, stacklevel=2,
    )
    scrna_log = log_normalize(scrna_expr)
    spot_log  = log_normalize(spot_expr)
    scaler = StandardScaler()
    scrna_norm = scaler.fit_transform(scrna_log).astype(np.float32)
    spot_norm  = scaler.transform(spot_log).astype(np.float32)
    return scrna_norm, spot_norm, scaler


# ══════════════════════════════════════════════════════════════════
# 3. 路线 A：PCA 跨域对齐（域偏移核心修复）
# ══════════════════════════════════════════════════════════════════

def pca_align_features(
    scrna_lognorm,
    xenium_lognorm,
    n_components: int = 50,
    clip_percentile: float = 99.0,
    verbose: bool = True,
):
    """
    在 scRNA 上 fit PCA，把 Xenium 投影到同一空间。

    步骤：
      1. Xenium log-norm 裁剪到 scRNA 第 99% 分位（消除文库小放大的极端值）
      2. PCA 仅在 scRNA 上 fit
      3. Xenium 用同一个 PCA transform

    返回 (scrna_pca, xenium_pca, pca_model)。
    """
    # Step 1: clip Xenium 极端值
    if clip_percentile is not None:
        clip_vals = np.percentile(scrna_lognorm, clip_percentile, axis=0)
        n_clipped = int((xenium_lognorm > clip_vals).sum())
        xenium_clipped = np.clip(xenium_lognorm, None, clip_vals)
        if verbose:
            pct = 100.0 * n_clipped / xenium_lognorm.size
            print(f"    [PCA] Xenium 裁剪: {n_clipped:,} 值 ({pct:.2f}%) "
                  f"超 scRNA {clip_percentile}th pct")
    else:
        xenium_clipped = xenium_lognorm.copy()

    # Step 2: scRNA 上 fit PCA
    n_comp = min(n_components, min(scrna_lognorm.shape) - 1)
    if verbose:
        print(f"    [PCA] 在 {scrna_lognorm.shape[0]:,} scRNA 细胞上 fit "
              f"PCA(n_components={n_comp})…")
    t0 = time.time()
    pca = PCA(n_components=n_comp, random_state=42)
    scrna_pca = pca.fit_transform(scrna_lognorm)
    if verbose:
        print(f"    [PCA] 解释方差: {pca.explained_variance_ratio_.sum():.3f}  "
              f"({time.time()-t0:.1f}s)")

    # Step 3: 投影 Xenium
    xenium_pca = pca.transform(xenium_clipped)
    if verbose:
        print(f"    [PCA] {scrna_lognorm.shape[1]} → {n_comp} dims")
        print(f"    [PCA] scRNA  PC range: [{scrna_pca.min():+.2f}, {scrna_pca.max():+.2f}]")
        print(f"    [PCA] Xenium PC range: [{xenium_pca.min():+.2f}, {xenium_pca.max():+.2f}]")

        # 诊断：两域在 PC1-PC2 上的均值距离（看分布是否真重叠）
        d12_sc = scrna_pca[:, :2].mean(axis=0)
        d12_xe = xenium_pca[:, :2].mean(axis=0)
        gap = float(np.linalg.norm(d12_sc - d12_xe))
        sc_std = float(np.linalg.norm(scrna_pca[:, :2].std(axis=0)))
        print(f"    [PCA] PC1-PC2 域均值差: {gap:.2f}  vs scRNA std: {sc_std:.2f}  "
              f"(差/std 比 {gap/max(sc_std,1e-6):.2f}, <1 健康)")

    return (scrna_pca.astype(np.float32),
            xenium_pca.astype(np.float32),
            pca)


def prepare_features_for_gnn(
    scrna_counts,
    spot_counts,
    n_pca: int = 50,
    clip_percentile: float = 99.0,
    verbose: bool = True,
):
    """
    GNN 特征流水线一步完成：原始 counts → log_normalize → PCA 对齐。

    这是 Notebook Cell 3 构图的唯一推荐入口（修复 v4 BUG-1）。
    """
    if verbose:
        sc_lib = scrna_counts.sum(axis=1)
        sp_lib = spot_counts.sum(axis=1)
        ratio  = sc_lib.mean() / max(sp_lib.mean(), 1)
        print(f"  scRNA 文库: 中位 {np.median(sc_lib):.0f}  均值 {sc_lib.mean():.0f}")
        print(f"  spot  文库: 中位 {np.median(sp_lib):.0f}  均值 {sp_lib.mean():.0f}")
        print(f"  文库比 {ratio:.0f}x → log-norm 后差 ~{np.log1p(ratio):.1f}x（PCA 前）")

    if verbose: print("  Step 1/2: log_normalize …")
    scrna_log = log_normalize(scrna_counts.astype(np.float32))
    spot_log  = log_normalize(spot_counts.astype(np.float32))

    if verbose: print(f"  Step 2/2: pca_align_features (n_pca={n_pca}) …")
    scrna_pca, spot_pca, pca_model = pca_align_features(
        scrna_lognorm  = scrna_log,
        xenium_lognorm = spot_log,
        n_components   = n_pca,
        clip_percentile= clip_percentile,
        verbose        = verbose,
    )
    if verbose:
        print(f"  ✅ 特征准备完成: {scrna_counts.shape[1]} genes → {n_pca} PCA dims")
    return scrna_pca, spot_pca, pca_model


# ══════════════════════════════════════════════════════════════════
# 4. 类权重 cap
# ══════════════════════════════════════════════════════════════════

def compute_capped_class_weights(
    labels,
    max_weight_multiplier: float = 5.0,
    min_weight:            float = 1.0,
    verbose:               bool  = True,
):
    """
    修改：增加 min_weight 下限。原 normed/mean 后多数类权重 < 1，
          导致模型对多数类错分惩罚不足，引发亚型反向偏置。
          min_weight=1.0 保证多数类至少与平均类等权。
    """
    classes = np.unique(labels)
    n_classes = int(labels.max()) + 1
    raw = compute_class_weight("balanced", classes=classes, y=labels)
    normed = raw / raw.mean()
    capped = np.clip(normed, min_weight, max_weight_multiplier)

    if verbose:
        print(f"    [权重] {len(classes)} 类 | raw [{raw.min():.2f}, {raw.max():.2f}] | "
              f"capped [{capped.min():.2f}, {capped.max():.2f}] "
              f"(范围 [{min_weight}, {max_weight_multiplier}]×)")

    weight_vec = np.ones(n_classes, dtype=np.float32)
    for c, w in zip(classes, capped):
        weight_vec[int(c)] = float(w)
    return torch.from_numpy(weight_vec).float()


# ══════════════════════════════════════════════════════════════════
# 5. kNN 内部工具
# ══════════════════════════════════════════════════════════════════

def _intra_knn(X, k, label="", verbose=True):
    """单域内 kNN（不含跨域边）。返回局部 (src, dst)。"""
    n = len(X)
    if verbose:
        print(f"    [{label}] intra kd_tree k={k}, n={n:,}, dim={X.shape[1]} …")
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="kd_tree", n_jobs=-1).fit(X)
    _, idx = nbrs.kneighbors(X)
    if verbose:
        print(f"    [{label}] done {time.time()-t0:.1f}s")
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()
    return src.astype(np.int64), dst.astype(np.int64)


def _spatial_knn(coords, k, verbose=True):
    return _intra_knn(coords, k, label="spatial", verbose=verbose)


def _mutual_cross_knn(X_a, X_b, k=10, verbose=True):
    """
    路线 B：跨域 mutual nearest neighbors（Scanorama / Seurat anchor 标准做法）。

    返回 (src_in_a, dst_in_b) 的 mutual NN 边对。
    """
    t0 = time.time()
    n_a, n_b = len(X_a), len(X_b)
    k_eff = min(k, n_a, n_b)

    nbrs_b = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree", n_jobs=-1).fit(X_b)
    _, idx_a2b = nbrs_b.kneighbors(X_a)

    nbrs_a = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree", n_jobs=-1).fit(X_a)
    _, idx_b2a = nbrs_a.kneighbors(X_b)

    # mutual 判断
    b_set_of_a = [set(row.tolist()) for row in idx_a2b]
    a_set_of_b = [set(row.tolist()) for row in idx_b2a]

    src, dst = [], []
    for i in range(n_a):
        for j in b_set_of_a[i]:
            if i in a_set_of_b[j]:
                src.append(i); dst.append(j)

    if verbose:
        sym = len(src) / max(n_a * k_eff, 1)
        print(f"    [cross MNN] k={k_eff}: {len(src):,} edges in "
              f"{time.time()-t0:.1f}s (mutual rate {sym:.1%})")
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)

# ══════════════════════════════════════════════════════════════════
# 5b. 修复版跨域边构造（mutual + 双向 asymmetric）
# ══════════════════════════════════════════════════════════════════

def _cross_domain_edges(
    X_a, X_b,
    k_mutual: int = 10,
    k_asym:   int = 10,
    verbose:  bool = True,
):
    """
    [新增] 跨域边构造，修复纯 mutual NN 在域规模不对称时的稀疏问题。

    输出三类边的并集（去重在 build_spot_graph 末尾统一做）：
      1. mutual NN（高质量但稀疏）
      2. cell  → spot top-k_asym（保证每个 cell 至少 k_asym 条跨域边）
      3. spot → cell top-k_asym（保证每个 spot 至少 k_asym 条跨域边）

    覆盖率分析（k_asym=10, n_a=16569, n_b=298053）：
      asym 边数 = n_a*k_asym + n_b*k_asym = 165K + 2.98M = ~3.15M
      占总边比 ≈ 3.15M / (16M base + 3.15M cross) ≈ 16%   ✅ 健康区间
    """
    t0 = time.time()
    n_a, n_b = len(X_a), len(X_b)
    k_max = min(max(k_mutual, k_asym), n_a, n_b)

    # 一次性查询 max(k_mutual, k_asym) 个邻居，复用结果
    nbrs_b = NearestNeighbors(n_neighbors=k_max, algorithm="kd_tree",
                              n_jobs=-1).fit(X_b)
    _, idx_a2b = nbrs_b.kneighbors(X_a)        # cell  → spot

    nbrs_a = NearestNeighbors(n_neighbors=k_max, algorithm="kd_tree",
                              n_jobs=-1).fit(X_a)
    _, idx_b2a = nbrs_a.kneighbors(X_b)        # spot → cell

    src_list, dst_list = [], []

    # 1. mutual NN（用 k_mutual）─────────────────────────────
    a2b_m = idx_a2b[:, :k_mutual]
    b2a_m = idx_b2a[:, :k_mutual]
    b_set_of_a = [set(r.tolist()) for r in a2b_m]
    a_set_of_b = [set(r.tolist()) for r in b2a_m]
    n_mut = 0
    for i in range(n_a):
        for j in b_set_of_a[i]:
            if i in a_set_of_b[j]:
                src_list.append(i); dst_list.append(j); n_mut += 1

    # 2. cell → spot 单向 top-k_asym ────────────────────────
    a2b_a = idx_a2b[:, :k_asym]
    src_list.extend(np.repeat(np.arange(n_a), k_asym).tolist())
    dst_list.extend(a2b_a.ravel().tolist())

    # 3. spot → cell 单向 top-k_asym ────────────────────────
    b2a_a = idx_b2a[:, :k_asym]
    src_list.extend(b2a_a.ravel().tolist())
    dst_list.extend(np.repeat(np.arange(n_b), k_asym).tolist())

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)

    if verbose:
        n_asym = len(src) - n_mut
        mut_rate = n_mut / max(n_a * k_mutual, 1)
        print(f"    [cross] mutual={n_mut:,} asym={n_asym:,} "
              f"total(去重前)={len(src):,}  ({time.time()-t0:.1f}s)")
        print(f"    [cross] mutual rate {mut_rate:.1%}  "
              f"(asym 边保证每节点至少 {k_asym} 条跨域连接)")

    return src, dst

# ══════════════════════════════════════════════════════════════════
# 6. 构建联合图（融合：PCA 已对齐 + MNN 跨域边）
# ══════════════════════════════════════════════════════════════════

def build_spot_graph(
    scrna_norm,            # 已 PCA 对齐的 scRNA 特征 (n_scrna, 50)
    spot_norm,             # 已 PCA 对齐的 spot 特征  (n_spots, 50)
    spot_coords,
    scrna_labels,
    k_feat: int = 15,
    k_spatial: int = 10,
    k_cross: int = 10,                   # ← 跨域 MNN 的 k
    val_ratio: float = 0.2,
    max_weight_multiplier: float = 5.0,
    verbose: bool = True,
):
    """
    构建联合图。输入是 PCA 对齐后的特征。

    边的四个来源：
      (a) scRNA-内 kNN     k_feat
      (b) spot-内  kNN     k_feat
      (c) 跨域 MNN         k_cross   ★ 关键
      (d) spot 空间 kNN    k_spatial

    新增打印"跨域边占比"：健康 5–50%；<5% 警告（图断裂）。
    """
    t_total = time.time()
    n_scrna = scrna_norm.shape[0]
    n_spots = spot_norm.shape[0]
    n_total = n_scrna + n_spots

    if verbose:
        feat_dim = scrna_norm.shape[1]
        print(f"  scRNA: {n_scrna:,}  |  spot: {n_spots:,}  |  total: {n_total:,}")
        print(f"  特征维度: {feat_dim}  内存: "
              f"{n_total*feat_dim*4/1024**2:.0f} MB")
        if feat_dim > 100:
            print(f"  ⚠  特征维度 {feat_dim} 较大，建议先调用 prepare_features_for_gnn() "
                  f"得到 50d PCA 特征")

    Xs = scrna_norm
    Xx = spot_norm

    # (a) scRNA-内
    if verbose: print(f"\n  [1/4] scRNA-内 kNN  (k={k_feat})")
    src_s, dst_s = _intra_knn(Xs, k=k_feat, label="scRNA", verbose=verbose)

    # (b) spot-内
    if verbose: print(f"\n  [2/4] spot-内 kNN  (k={k_feat})")
    src_x, dst_x = _intra_knn(Xx, k=k_feat, label="spot", verbose=verbose)
    src_x = src_x + n_scrna
    dst_x = dst_x + n_scrna

    # (c) 跨域 MNN — 路线 B 关键
    if verbose: print(f"\n  [3/4] 跨域 mutual NN  (k={k_cross})  ★")
    # (c) 跨域边（mutual + 双向 asymmetric）
    if verbose: print(f"\n  [3/4] 跨域边  (k_mutual={k_cross}, k_asym={k_cross})  ★")
    src_ca, dst_cb = _cross_domain_edges(
        Xs, Xx,
        k_mutual=k_cross,
        k_asym=k_cross,
        verbose=verbose,
    )
    src_c = src_ca
    dst_c = dst_cb + n_scrna

    # (d) spot 空间
    if verbose: print(f"\n  [4/4] spot 空间 kNN  (k={k_spatial})")
    src_sp, dst_sp = _spatial_knn(spot_coords, k=k_spatial, verbose=verbose)
    src_sp = src_sp + n_scrna
    dst_sp = dst_sp + n_scrna

    # 合并 + 双向 + 去重
    if verbose: print(f"\n  合并 4 类边集 …")
    src_all = np.concatenate([src_s, dst_s, src_x, dst_x,
                              src_c, dst_c, src_sp, dst_sp])
    dst_all = np.concatenate([dst_s, src_s, dst_x, src_x,
                              dst_c, src_c, dst_sp, src_sp])
    edges = np.unique(np.stack([src_all, dst_all], 1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edge_index = torch.from_numpy(edges.T).long()

    # 跨域占比诊断
    is_a_sc = edges[:, 0] < n_scrna
    is_b_sc = edges[:, 1] < n_scrna
    n_cross = int((is_a_sc ^ is_b_sc).sum())
    n_edges = edge_index.shape[1]
    cross_pct = 100.0 * n_cross / max(n_edges, 1)

    if verbose:
        flag = "✅" if 5 <= cross_pct <= 50 else ("⚠️" if cross_pct < 5 else "❓")
        print(f"\n  总边: {n_edges:,}  | 跨域: {n_cross:,} ({cross_pct:.1f}%) {flag}")
        print(f"     5–50% 健康；<5% 域间断裂；>50% 噪声偏多")

    # 标签 / split
    labels_all = np.full(n_total, -1, dtype=np.int64)
    labels_all[:n_scrna] = scrna_labels

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_local, val_local = next(sss.split(np.arange(n_scrna), scrna_labels))

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask   = torch.zeros(n_total, dtype=torch.bool)
    spot_mask  = torch.zeros(n_total, dtype=torch.bool)
    train_mask[train_local] = True
    val_mask[val_local]     = True
    spot_mask[n_scrna:]     = True

    # 类权重
    if verbose: print(f"\n  计算类别权重 (cap={max_weight_multiplier}×) …")
    class_weights = compute_capped_class_weights(
    scrna_labels,
    max_weight_multiplier=max_weight_multiplier,
    min_weight=1.0,                                    # ← 修改处：显式传 1.0
    verbose=verbose,
    )   

    # 组装
    X_all = np.vstack([scrna_norm, spot_norm])
    data = Data(
        x          = torch.from_numpy(X_all).float(),
        edge_index = edge_index,
        y          = torch.from_numpy(labels_all).long(),
        train_mask = train_mask,
        val_mask   = val_mask,
        spot_mask  = spot_mask,
    )
    data.xenium_mask = spot_mask
    data.n_scrna = n_scrna
    data.n_spots = n_spots

    split_info = {
        "train_idx": train_local,
        "val_idx":   val_local,
        "n_scrna":   n_scrna,
        "n_spots":   n_spots,
        "cross_edge_pct": cross_pct,    # ← 健康度指标
        "n_edges": n_edges,
        "n_cross_edges": n_cross,
    }

    if verbose:
        elapsed = time.time() - t_total
        print(f"\n  ✅ 图构建完成 {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"     train {train_mask.sum():,}  val {val_mask.sum():,}  "
              f"spot {spot_mask.sum():,}")

    return data, class_weights, split_info


# ══════════════════════════════════════════════════════════════════
# 7. spot → 细胞质心聚合（radius 自适应）
# ══════════════════════════════════════════════════════════════════

def aggregate_spot_to_cell(
    spot_proba,
    spot_coords,
    cell_coords,
    n_classes,
    bin_size: float = 10.0,
    radius_um: float = None,    # None → bin_size * 2.5
):
    """
    bin_size  : spot 的 bin 大小（μm）
    radius_um : 手动半径（None = bin_size * 2.5，覆盖 5x5 邻域）

    建议 radius_um = None，让函数自动适配 bin_size。
    """
    from sklearn.neighbors import BallTree

    if radius_um is None:
        radius_um = bin_size * 1.5

    tree = BallTree(spot_coords, metric="euclidean")
    indices = tree.query_radius(cell_coords, r=radius_um)

    cell_proba = np.zeros((len(cell_coords), n_classes), dtype=np.float32)
    n_fb = 0
    for i, nbr in enumerate(indices):
        if len(nbr) > 0:
            cell_proba[i] = spot_proba[nbr].mean(axis=0)
        else:
            _, nearest = tree.query(cell_coords[[i]], k=1)
            cell_proba[i] = spot_proba[nearest[0, 0]]
            n_fb += 1

    if n_fb > 0:
        pct = 100.0 * n_fb / len(cell_coords)
        print(f"  [聚合] 最近邻回退 {n_fb:,} ({pct:.1f}%)  "
              f"(radius={radius_um:.1f}μm, bin={bin_size}μm)")
        if pct > 10:
            print(f"     建议 radius 增加到 {radius_um*1.5:.0f}–{radius_um*2:.0f}")

    return cell_proba.argmax(axis=1), cell_proba.max(axis=1)

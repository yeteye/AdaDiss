"""
utils_spot.py — 转录本级别数据流水线（不依赖细胞分割）

修复清单
--------
spot_mask → xenium_mask（与 models.py 统一）
验证集划分 → StratifiedShuffleSplit（修复 P1-⑧）

设计思路
--------
- scRNA 细胞作为有标签节点（ground truth 来自 Flex 注释）
- Xenium spot 作为无标签节点（目标预测对象）
- 两类节点共处同一张图，通过特征 kNN + 空间 kNN 连边
- GNN 通过消息传递端到端学习聚合权重

节点特征对齐
-----------
- 基因集：Xenium panel ∩ scRNA 基因
- 归一化：log1p → StandardScaler（只在 scRNA 上 fit，避免数据泄露）
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


# ══════════════════════════════════════════════════════════
# 1. 加载 Xenium 原始转录本
# ══════════════════════════════════════════════════════════

def load_xenium_transcripts(
    transcript_path: str,
    gene_list: list,
    qv_threshold: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    加载并过滤 Xenium 原始转录本文件。

    Parameters
    ----------
    transcript_path : transcripts.csv.gz 的路径
    gene_list       : scRNA 中的基因列表，用于特征对齐
    qv_threshold    : Q-score 阈值（论文使用 20）

    Returns
    -------
    DataFrame，列：gene, x, y
    """
    if verbose:
        print(f"加载转录本文件: {transcript_path}")

    if transcript_path.endswith(".parquet"):
        df = pd.read_parquet(transcript_path)
    else:
        df = pd.read_csv(transcript_path)

    if verbose:
        print(f"  原始转录本数: {len(df):,}")
        print(f"  列名: {list(df.columns)}")

    # 列名标准化
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
            print(f"  Q≥{qv_threshold} 过滤后: {len(df):,}")

    gene_set = set(gene_list)
    df = df[df["gene"].isin(gene_set)].copy()
    if verbose:
        print(f"  基因对齐后: {len(df):,} 条转录本，{df['gene'].nunique()} 个基因")

    df["x"] = df["x_raw"].round(0).astype(int)
    df["y"] = df["y_raw"].round(0).astype(int)
    return df[["gene", "x", "y"]].copy()


# ══════════════════════════════════════════════════════════
# 2. 聚合转录本 → spot 表达矩阵
# ══════════════════════════════════════════════════════════

def bin_transcripts_to_spots(
    df: pd.DataFrame,
    gene_list: list,
    bin_size: int = 5,
    min_transcripts: int = 1,
    verbose: bool = True,
) -> tuple:
    """
    将转录本聚合为 spot 级别表达矩阵。

    Parameters
    ----------
    bin_size : bin 边长（μm）。推荐 5μm（节点数适中）

    Returns
    -------
    spot_expr   : (n_spots, n_genes) float32 raw counts
    spot_coords : (n_spots, 2) float32 质心坐标（μm）
    """
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
    spot_matrix = spot_matrix[keep]
    coords_keep = spot_key[keep][["bx", "by"]].values.astype(np.float32)

    spot_expr   = spot_matrix.toarray().astype(np.float32)
    spot_coords = coords_keep

    if verbose:
        print(f"  bin_size={bin_size}μm → {spot_expr.shape[0]:,} 个有效 spot")
        print(f"  特征维度: {spot_expr.shape[1]}")
        print(f"  每 spot 平均转录本数: {spot_expr.sum(axis=1).mean():.1f}")

    return spot_expr, spot_coords


# ══════════════════════════════════════════════════════════
# 3. 归一化
# ══════════════════════════════════════════════════════════

def log_normalize(X: np.ndarray) -> np.ndarray:
    """log1p + library-size normalization"""
    total = X.sum(axis=1, keepdims=True).clip(min=1)
    return np.log1p(X / total * 1e4).astype(np.float32)


def unified_normalize_spot(
    scrna_expr: np.ndarray,
    spot_expr:  np.ndarray,
) -> tuple:
    """
    统一归一化：
    1. 各自 log1p 归一化
    2. StandardScaler 只在 scRNA 上 fit，然后 transform 两侧
    """
    scrna_log = log_normalize(scrna_expr)
    spot_log  = log_normalize(spot_expr)

    scaler     = StandardScaler()
    scrna_norm = scaler.fit_transform(scrna_log).astype(np.float32)
    spot_norm  = scaler.transform(spot_log).astype(np.float32)

    return scrna_norm, spot_norm, scaler


# ══════════════════════════════════════════════════════════
# 4. 构建联合图（核心）
# ══════════════════════════════════════════════════════════

def build_spot_graph(
    scrna_norm:   np.ndarray,
    spot_norm:    np.ndarray,
    spot_coords:  np.ndarray,
    scrna_labels: np.ndarray,
    k_feat:       int   = 15,
    k_spatial:    int   = 10,
    val_ratio:    float = 0.2,
    verbose:      bool  = True,
) -> tuple:
    """
    构建 scRNA 细胞 + Xenium spot 的联合异质图。

    修复
    ----
    - 节点掩码统一命名为 xenium_mask（与 models.py 一致）
    - 验证集改用 StratifiedShuffleSplit（修复 P1-⑧）

    边的设计
    --------
    - 特征 kNN（k_feat）：归一化基因表达空间，覆盖跨模态连边
    - 空间 kNN（k_spatial）：spot 物理坐标，利用空间邻近性

    创新点 vs TopACT
    ----------------
    TopACT  : 固定半径聚合 + SVM，无学习
    本方法  : GNN 端到端学习聚合权重，半监督跨模态对齐
    """
    n_scrna = scrna_norm.shape[0]
    n_spots = spot_norm.shape[0]
    n_total = n_scrna + n_spots

    if verbose:
        print(f"  scRNA 节点: {n_scrna:,}  |  spot 节点: {n_spots:,}")
        print(f"  总节点数  : {n_total:,}")

    X_all = np.vstack([scrna_norm, spot_norm])

    # ── 特征空间 kNN ─────────────────────────────────────
    if verbose:
        print(f"  构建特征 kNN (k={k_feat})...")
    nn_feat = NearestNeighbors(
        n_neighbors=k_feat + 1, algorithm="ball_tree", n_jobs=-1
    )
    nn_feat.fit(X_all)
    _, feat_idx = nn_feat.kneighbors(X_all)
    src_f = np.repeat(np.arange(n_total), k_feat)
    dst_f = feat_idx[:, 1:].ravel()

    # ── 空间 kNN（仅 spot 之间）──────────────────────────
    if verbose:
        print(f"  构建空间 kNN (k={k_spatial})...")
    nn_sp = NearestNeighbors(
        n_neighbors=k_spatial + 1, algorithm="ball_tree", n_jobs=-1
    )
    nn_sp.fit(spot_coords)
    _, sp_idx = nn_sp.kneighbors(spot_coords)
    src_s = np.repeat(np.arange(n_spots), k_spatial) + n_scrna
    dst_s = sp_idx[:, 1:].ravel() + n_scrna

    # ── 合并 + 双向 + 去重 + 去自环 ──────────────────────
    src_all = np.concatenate([src_f, dst_f, src_s, dst_s])
    dst_all = np.concatenate([dst_f, src_f, dst_s, src_s])
    edges   = np.stack([src_all, dst_all], axis=1)
    edges   = np.unique(edges, axis=0)
    edges   = edges[edges[:, 0] != edges[:, 1]]
    edge_index = torch.from_numpy(edges.T).long()

    if verbose:
        print(f"  总边数: {edge_index.shape[1]:,}")

    # ── 标签向量（-1 = 无标签）────────────────────────────
    labels_all = np.full(n_total, -1, dtype=np.int64)
    labels_all[:n_scrna] = scrna_labels

    # ── 分层划分验证集（修复 P1-⑧）──────────────────────
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=42
    )
    scrna_range = np.arange(n_scrna)
    train_local, val_local = next(sss.split(scrna_range, scrna_labels))
    train_idx = scrna_range[train_local]   # 全局索引（scRNA 节点在 0..n_scrna-1）
    val_idx   = scrna_range[val_local]

    train_mask   = torch.zeros(n_total, dtype=torch.bool)
    val_mask     = torch.zeros(n_total, dtype=torch.bool)
    xenium_mask  = torch.zeros(n_total, dtype=torch.bool)   # 修复：统一命名
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    xenium_mask[n_scrna:] = True

    # ── 类别权重 ──────────────────────────────────────────
    classes = np.unique(scrna_labels)
    cw = compute_class_weight("balanced", classes=classes, y=scrna_labels)
    class_weights = torch.from_numpy(cw).float()

    # ── 组装 PyG Data ─────────────────────────────────────
    data = Data(
        x           = torch.from_numpy(X_all).float(),
        edge_index  = edge_index,
        y           = torch.from_numpy(labels_all).long(),
        train_mask  = train_mask,
        val_mask    = val_mask,
        spot_mask   = xenium_mask,   # 保留 spot_mask 别名（向下兼容）
    )
    data.xenium_mask = xenium_mask   # 修复：添加 xenium_mask 属性
    data.n_scrna = n_scrna
    data.n_spots = n_spots

    split_info = {
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "n_scrna":   n_scrna,
        "n_spots":   n_spots,
    }

    if verbose:
        print(f"  训练: {train_mask.sum()}  验证: {val_mask.sum()}"
              f"  推断 spot: {xenium_mask.sum():,}")

    return data, class_weights, split_info


# ══════════════════════════════════════════════════════════
# 5. spot 预测 → 细胞级别聚合（评估用）
# ══════════════════════════════════════════════════════════

def aggregate_spot_to_cell(
    spot_proba:  np.ndarray,
    spot_coords: np.ndarray,
    cell_coords: np.ndarray,
    n_classes:   int,
    radius_um:   float = 10.0,
) -> tuple:
    """
    将 spot 预测概率聚合到细胞质心级别（仅用于最终评估）。
    对每个细胞，取 radius_um 范围内所有 spot 的概率均值。
    """
    from sklearn.neighbors import BallTree

    tree    = BallTree(spot_coords, metric="euclidean")
    indices = tree.query_radius(cell_coords, r=radius_um)

    cell_proba = np.zeros((len(cell_coords), n_classes), dtype=np.float32)
    for i, nbr in enumerate(indices):
        if len(nbr) > 0:
            cell_proba[i] = spot_proba[nbr].mean(axis=0)
        else:
            _, nearest = tree.query(cell_coords[[i]], k=1)
            cell_proba[i] = spot_proba[nearest[0, 0]]

    return cell_proba.argmax(axis=1), cell_proba.max(axis=1)

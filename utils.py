"""
utils.py — 数据预处理 / 图构建 / Domain Adaptation 损失函数

修复清单
--------
P0-①  双重独立归一化   → unified_normalize()，只在 scRNA 上 fit
P0-②  state_dict 浅拷贝 → 调用方使用 copy.deepcopy()（此处提供辅助 save_best）
P0-③  有向图未对称     → build_mutual_knn_graph() 用 to_undirected + mutual filter
P1-⑥  原始 counts 未预处理 → log_normalize()
P1-⑦  类别不均衡      → build_combined_dataset() 返回 class_weights
P1-⑧  验证集未分层    → StratifiedShuffleSplit
P3-⑬  mutual kNN      → 双向确认才保留边
"""

import copy
import warnings
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════
# 1. 生物学数据预处理
# ══════════════════════════════════════════════════════════

def log_normalize(counts: np.ndarray, scale_factor: float = 1e4) -> np.ndarray:
    """
    Library-size normalization + log1p（等价于 Seurat LogNormalize）。

    必须在 StandardScaler 之前做，否则高表达基因主导归一化方向。

    Parameters
    ----------
    counts : (n_cells, n_genes) raw UMI counts
    scale_factor : 目标文库大小（默认 10,000）
    """
    lib_sizes = counts.sum(axis=1, keepdims=True).clip(min=1)
    return np.log1p(counts / lib_sizes * scale_factor)


def unified_normalize(
    scrna_expr: np.ndarray,
    spatial_expr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    统一特征空间归一化。

    关键：只在 scRNA 上 fit_transform，Xenium 用 transform。
    两个数据集共享同一坐标系，kNN 图才有意义。

    Returns
    -------
    scrna_norm, spatial_norm, fitted_scaler
    """
    scaler = StandardScaler()
    scrna_norm = scaler.fit_transform(scrna_expr)
    spatial_norm = scaler.transform(spatial_expr)
    return scrna_norm, spatial_norm, scaler


# ══════════════════════════════════════════════════════════
# 2. 图构建
# ══════════════════════════════════════════════════════════

def build_mutual_knn_graph(
    features: np.ndarray,
    k: int = 30,
) -> torch.Tensor:
    """
    构建 mutual kNN 图，并强制对称（无向图）。

    修复点：
    - 原始代码只加单向边，GCN/SAGE 需要无向图
    - Mutual filter：只保留双向确认的近邻（降噪）

    Parameters
    ----------
    features : (n_cells, n_features)
    k        : 每个节点保留的邻居数

    Returns
    -------
    edge_index : (2, n_edges) 无向、无重复、无自环
    """
    n = features.shape[0]
    nbrs = NearestNeighbors(
        n_neighbors=k + 1, metric="cosine", n_jobs=-1
    ).fit(features)
    _, indices = nbrs.kneighbors(features)

    # 建立有向邻居集
    neighbor_sets = [set(indices[i, 1:].tolist()) for i in range(n)]

    # Mutual edges：i∈N(j) AND j∈N(i)
    src, dst = [], []
    for i in range(n):
        for j in neighbor_sets[i]:
            if i in neighbor_sets[j]:   # mutual 确认
                src.append(i)
                dst.append(j)

    if len(src) == 0:
        warnings.warn(
            "No mutual edges found (k may be too small). "
            "Falling back to standard symmetric kNN."
        )
        for i in range(n):
            for j in neighbor_sets[i]:
                src.append(i)
                dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=n)     # 补全对称边
    # 去重（to_undirected 已去重，但保险起见）
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def build_combined_dataset(
    flex_expr_norm: np.ndarray,
    xenium_expr_norm: np.ndarray,
    flex_labels: np.ndarray,
    k: int = 30,
    val_ratio: float = 0.2,
) -> tuple[Data, torch.Tensor, dict]:
    """
    构建半监督联合数据集。

    修复点：
    - P1-⑦ 计算 class_weights（不均衡处理）
    - P1-⑧ 分层采样验证集

    Returns
    -------
    data         : PyG Data 对象，含 train/val/xenium 掩码
    class_weights: (n_classes,) 用于加权损失函数
    split_info   : {'train_idx': ..., 'val_idx': ...}
    """
    n_flex   = len(flex_expr_norm)
    n_xenium = len(xenium_expr_norm)
    n_classes = int(flex_labels.max()) + 1

    # ── 分层采样 ─────────────────────────────────────────
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=SEED
    )
    train_idx, val_idx = next(sss.split(flex_expr_norm, flex_labels))

    # ── 类别权重 ──────────────────────────────────────────
    cls_w = compute_class_weight(
        "balanced",
        classes=np.arange(n_classes),
        y=flex_labels,
    )
    class_weights = torch.tensor(cls_w, dtype=torch.float)

    # ── 合并特征 + 构图 ───────────────────────────────────
    combined = np.vstack([flex_expr_norm, xenium_expr_norm])
    print(f"  Building mutual kNN graph on {n_flex + n_xenium:,} cells (k={k})…")
    edge_index = build_mutual_knn_graph(combined, k=k)

    # ── 标签（Xenium 用 -1 表示无标签）──────────────────────
    combined_labels = np.concatenate(
        [flex_labels, -np.ones(n_xenium, dtype=int)]
    )

    # ── PyG Data ─────────────────────────────────────────
    data = Data(
        x=torch.tensor(combined, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(combined_labels, dtype=torch.long),
    )

    N = n_flex + n_xenium
    train_mask  = torch.zeros(N, dtype=torch.bool)
    val_mask    = torch.zeros(N, dtype=torch.bool)
    xen_mask    = torch.zeros(N, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    xen_mask[n_flex:]     = True

    data.train_mask  = train_mask
    data.val_mask    = val_mask
    data.xenium_mask = xen_mask
    data.n_flex      = n_flex
    data.n_xenium    = n_xenium

    print(
        f"  Edges: {edge_index.shape[1]:,} | "
        f"Train: {train_mask.sum():,} | "
        f"Val: {val_mask.sum():,} | "
        f"Xenium: {n_xenium:,}"
    )
    return data, class_weights, {"train_idx": train_idx, "val_idx": val_idx}


# ══════════════════════════════════════════════════════════
# 3. Domain Adaptation 损失函数
# ══════════════════════════════════════════════════════════

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, bw: float) -> torch.Tensor:
    """RBF kernel matrix K(x, y) with bandwidth bw."""
    n, m = x.size(0), y.size(0)
    xx = x.pow(2).sum(1, keepdim=True).expand(n, m)
    yy = y.pow(2).sum(1, keepdim=True).t().expand(n, m)
    sq = (xx + yy - 2.0 * torch.mm(x, y.t())).clamp(min=0.0)
    return torch.exp(-sq / (2.0 * bw))


def mmd_loss(
    source_h: torch.Tensor,
    target_h: torch.Tensor,
    bandwidths: tuple = (1.0, 10.0, 100.0),
) -> torch.Tensor:
    """
    Multi-kernel Maximum Mean Discrepancy (MK-MMD)。

    强制对齐 scRNA 和 Xenium 的隐层特征分布，
    是 domain adaptation 的核心约束。

    Parameters
    ----------
    source_h : (n_scrna, h_dim) scRNA 节点的隐层表示
    target_h : (n_xenium, h_dim) Xenium 节点的隐层表示
    """
    loss = torch.zeros(1, device=source_h.device)
    for bw in bandwidths:
        Kss = _rbf_kernel(source_h, source_h, bw)
        Ktt = _rbf_kernel(target_h, target_h, bw)
        Kst = _rbf_kernel(source_h, target_h, bw)
        loss = loss + Kss.mean() + Ktt.mean() - 2.0 * Kst.mean()
    return loss / len(bandwidths)


def entropy_regularization(log_probs: torch.Tensor) -> torch.Tensor:
    """
    熵最小化正则化（Xenium 无标签节点）。

    鼓励模型对 Xenium 节点作出高置信度预测（低熵），
    等价于"让决策边界远离无标签数据点"。

    Parameters
    ----------
    log_probs : (n_xenium, n_classes) log-softmax 输出
    """
    probs   = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy.mean()


def get_pseudo_labels(
    log_probs: torch.Tensor,
    threshold: float = 0.90,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    提取 Xenium 节点的高置信度伪标签。

    Returns
    -------
    pseudo_labels   : (n_xenium,) 预测类别（所有节点）
    high_conf_mask  : (n_xenium,) bool，是否超过 threshold
    confidence      : (n_xenium,) 最大 softmax 概率
    """
    probs                  = log_probs.exp()
    confidence, pseudo_lbl = probs.max(dim=1)
    high_conf_mask         = confidence >= threshold
    return pseudo_lbl, high_conf_mask, confidence


# ══════════════════════════════════════════════════════════
# 4. 训练辅助
# ══════════════════════════════════════════════════════════

def save_best_state(model: torch.nn.Module) -> dict:
    """
    深拷贝当前模型权重（修复 P0-② 浅拷贝 Bug）。

    .copy() 只复制字典结构，tensor 值仍是引用；
    GPU 继续训练会原地修改这些 tensor。
    必须用 copy.deepcopy()。
    """
    return copy.deepcopy(model.state_dict())


def set_seed(seed: int = SEED):
    """全局随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

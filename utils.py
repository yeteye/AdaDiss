"""
utils.py — 数据预处理 / 图构建 / Domain Adaptation 损失函数

修复清单
--------
P0-②  state_dict 浅拷贝  → save_best_state() 使用 deepcopy
P0-③  有向图未对称       → build_mutual_knn_graph() + to_undirected
P1-⑥  原始 counts 未预处理 → log_normalize()
P1-⑦  类别不均衡         → build_combined_dataset() 返回 class_weights
P1-⑧  验证集未分层       → StratifiedShuffleSplit
P3-⑬  mutual kNN         → 双向确认才保留边

新增
----
run_experiment : Notebook 调用接口适配（与 Cell 4b 参数名一致）
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
    """
    lib_sizes = counts.sum(axis=1, keepdims=True).clip(min=1)
    return np.log1p(counts / lib_sizes * scale_factor)


def unified_normalize(
    scrna_expr: np.ndarray,
    spatial_expr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    统一特征空间归一化。
    只在 scRNA 上 fit_transform，Xenium 用 transform，防止数据泄露。
    """
    scaler       = StandardScaler()
    scrna_norm   = scaler.fit_transform(scrna_expr)
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
    构建 mutual kNN 图（双向确认 + 强制对称）。

    修复 P0-③：原始代码只加单向边，GCN/SAGE 需要无向图。
    Mutual filter：只保留双向确认的近邻（降低噪声边）。
    """
    n = features.shape[0]
    nbrs = NearestNeighbors(
        n_neighbors=k + 1, metric="cosine", n_jobs=-1
    ).fit(features)
    _, indices = nbrs.kneighbors(features)

    neighbor_sets = [set(indices[i, 1:].tolist()) for i in range(n)]

    src, dst = [], []
    for i in range(n):
        for j in neighbor_sets[i]:
            if i in neighbor_sets[j]:
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
    edge_index = to_undirected(edge_index, num_nodes=n)
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
    构建半监督联合数据集（scRNA 有标签 + Xenium 无标签）。

    修复 P1-⑦：class_weights（类别不均衡处理）
    修复 P1-⑧：StratifiedShuffleSplit 分层采样验证集
    注意：节点的 spot_mask 属性（与 utils_spot 统一命名）
    """
    n_flex   = len(flex_expr_norm)
    n_xenium = len(xenium_expr_norm)
    n_classes = int(flex_labels.max()) + 1

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=SEED
    )
    train_idx, val_idx = next(sss.split(flex_expr_norm, flex_labels))

    cls_w = compute_class_weight(
        "balanced",
        classes=np.arange(n_classes),
        y=flex_labels,
    )
    class_weights = torch.tensor(cls_w, dtype=torch.float)

    combined = np.vstack([flex_expr_norm, xenium_expr_norm])
    print(f"  Building mutual kNN graph on {n_flex + n_xenium:,} cells (k={k})…")
    edge_index = build_mutual_knn_graph(combined, k=k)

    combined_labels = np.concatenate(
        [flex_labels, -np.ones(n_xenium, dtype=int)]
    )

    data = Data(
        x=torch.tensor(combined, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(combined_labels, dtype=torch.long),
    )

    N = n_flex + n_xenium
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    spot_mask  = torch.zeros(N, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    spot_mask[n_flex:]    = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.spot_mask  = spot_mask   # 统一命名（不再使用 xenium_mask）
    data.n_scrna    = n_flex
    data.n_spots    = n_xenium

    print(
        f"  Edges: {edge_index.shape[1]:,} | "
        f"Train: {train_mask.sum():,} | "
        f"Val: {val_mask.sum():,} | "
        f"Spots: {n_xenium:,}"
    )
    return data, class_weights, {"train_idx": train_idx, "val_idx": val_idx}


# ══════════════════════════════════════════════════════════
# 3. Domain Adaptation 损失函数
# ══════════════════════════════════════════════════════════

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, bw: float) -> torch.Tensor:
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
    """Multi-kernel MMD：对齐 scRNA 和 Xenium 的隐层特征分布。"""
    loss = torch.zeros(1, device=source_h.device)
    for bw in bandwidths:
        Kss = _rbf_kernel(source_h, source_h, bw)
        Ktt = _rbf_kernel(target_h, target_h, bw)
        Kst = _rbf_kernel(source_h, target_h, bw)
        loss = loss + Kss.mean() + Ktt.mean() - 2.0 * Kst.mean()
    return loss / len(bandwidths)


def entropy_regularization(log_probs: torch.Tensor) -> torch.Tensor:
    """熵最小化正则化：鼓励 Xenium 节点作出高置信度预测。"""
    probs   = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy.mean()


def get_pseudo_labels(
    log_probs: torch.Tensor,
    threshold: float = 0.90,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """提取高置信度伪标签。"""
    probs                  = log_probs.exp()
    confidence, pseudo_lbl = probs.max(dim=1)
    high_conf_mask         = confidence >= threshold
    return pseudo_lbl, high_conf_mask, confidence


# ══════════════════════════════════════════════════════════
# 4. 训练辅助
# ══════════════════════════════════════════════════════════

def save_best_state(model: torch.nn.Module) -> dict:
    """深拷贝当前模型权重（修复 P0-② 浅拷贝 Bug）。"""
    return copy.deepcopy(model.state_dict())


def set_seed(seed: int = SEED):
    """全局随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════
# 5. Notebook 接口适配：run_experiment
#    匹配 Cell 4b 的调用签名
# ══════════════════════════════════════════════════════════

def run_experiment(
    model: "torch.nn.Module",
    data,
    class_weights: "torch.Tensor",
    n_classes: int,
    device,
    params: dict,
    model_name: str,
    save_dir: str | None = None,
) -> dict:
    """
    Notebook Cell 4b 调用的训练入口。

    将 Notebook 的调用签名适配到 models.run_experiment 的签名：
    - n_classes (int)  →  cell_types (list[str])
    - device 单独参数  →  合并进 params["device"]

    Parameters
    ----------
    model         : 已实例化的 GNN 模型（GCN_AMP / GraphSAGE_AMP / GAT_AMP）
    data          : PyG Data 对象（utils_spot.build_spot_graph 的输出）
    class_weights : (n_classes,) 类别权重张量
    n_classes     : 细胞类型数量
    device        : torch.device 或 str（"cuda:0" / "cpu"）
    params        : 全局超参字典（Cell 1 的 PARAMS）
    model_name    : 模型名称（用于保存权重文件名）
    save_dir      : 权重保存目录（None = 不保存）

    Returns
    -------
    result dict（与 models.run_experiment 完全一致）
    """
    from models import run_experiment as _train

    # 生成 cell_types 占位列表（训练时只需长度，预测时标签由 Notebook 外层重映射）
    cell_types = [str(i) for i in range(n_classes)]

    # 合并 device 到 params（models.py 从 params["device"] 读取）
    _params = dict(params)
    _params["device"] = str(device) if not isinstance(device, str) else device

    return _train(
        model=model,
        model_name=model_name,
        data=data,
        cell_types=cell_types,
        params=_params,
        class_weights=class_weights,
        save_dir=save_dir,
    )

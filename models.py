"""
models.py — GNN 模型定义 + 半监督 Domain Adaptation 训练循环

修复清单
--------
P0-②  state_dict 浅拷贝  → save_best_state()（来自 utils）
P1-⑤  缺 DA 机制         → MMD + 熵正则 + 伪标签三合一
P1-⑦  无类别权重         → class_weights 传入 nll_loss
P2-⑨  无学习率调度器     → ReduceLROnPlateau
P2-⑩  无梯度裁剪         → clip_grad_norm_
P2-⑪  无批归一化         → BatchNorm1d 插在每层卷积后
P3-⑫  无伪标签           → get_pseudo_labels + pseudo_label_loss
P3-⑭  隐层维度跨度过大   → 可选 projection head（3000→512→256→C）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import numpy as np
from tqdm.auto import tqdm

from utils import (
    mmd_loss,
    entropy_regularization,
    get_pseudo_labels,
    save_best_state,
)


# ══════════════════════════════════════════════════════════
# 1. 模型定义
# ══════════════════════════════════════════════════════════

class GCN(nn.Module):
    """
    三层图卷积网络，带 BatchNorm + Dropout。

    修复 P2-⑪：每层卷积后插入 BatchNorm1d，稳定大规模图训练。
    修复 P3-⑭：可选输入投影层（in_dim→proj_dim）缩小跨度。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        proj_dim: int | None = None,     # 投影层维度，None=不用
    ):
        super().__init__()
        self.dropout = dropout

        # 可选投影层：3000→512 缓解跨度问题
        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(),
                nn.BatchNorm1d(proj_dim),
            )
            conv_in = proj_dim
        else:
            self.proj = None
            conv_in = in_dim

        self.conv1 = GCNConv(conv_in, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def encode(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (倒数第二层嵌入, 最终 logits)"""
        x, ei = data.x, data.edge_index
        if self.proj is not None:
            x = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(x, ei)))         # 隐层嵌入（用于 MMD）
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv3(h, ei)
        return h, F.log_softmax(out, dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        _, log_probs = self.encode(data)
        return log_probs


class GraphSAGE(nn.Module):
    """GraphSAGE，结构同 GCN，更适合归纳式推断（inductive）。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        proj_dim: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout

        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim), nn.ReLU(), nn.BatchNorm1d(proj_dim)
            )
            conv_in = proj_dim
        else:
            self.proj = None
            conv_in = in_dim

        self.conv1 = SAGEConv(conv_in, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)

    def encode(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x, ei = data.x, data.edge_index
        if self.proj is not None:
            x = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(x, ei)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv3(h, ei)
        return h, F.log_softmax(out, dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        _, log_probs = self.encode(data)
        return log_probs


class GAT(nn.Module):
    """
    图注意力网络，4头注意力。

    修复 P2-⑩：GAT 多头容易梯度爆炸，训练时需 clip_grad_norm_。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.5,
        proj_dim: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout

        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim), nn.ReLU(), nn.BatchNorm1d(proj_dim)
            )
            conv_in = proj_dim
        else:
            self.proj = None
            conv_in = in_dim

        self.conv1 = GATConv(conv_in, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn1   = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, out_dim, heads=1, dropout=dropout, concat=False)

    def encode(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x, ei = data.x, data.edge_index
        if self.proj is not None:
            x = self.proj(x)
        x = F.elu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.conv2(x, ei)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv3(h, ei)
        return h, F.log_softmax(out, dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        _, log_probs = self.encode(data)
        return log_probs


# ══════════════════════════════════════════════════════════
# 2. 训练 / 评估函数
# ══════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    # DA 超参
    lambda_mmd: float = 0.1,
    lambda_ent: float = 0.01,
    lambda_pl:  float = 0.3,
    pl_threshold: float = 0.90,
    # 梯度裁剪
    max_grad_norm: float = 1.0,
    # 伪标签（上一轮缓存）
    cached_pseudo_labels: torch.Tensor | None = None,
    cached_pseudo_mask: torch.Tensor | None = None,
) -> dict:
    """
    一个完整训练 epoch，包含：
    1. 监督损失（scRNA 有标签节点，加权 NLL）
    2. MMD 损失（对齐 scRNA/Xenium 隐层分布）
    3. 熵正则化（最小化 Xenium 预测熵）
    4. 伪标签损失（高置信度 Xenium 节点）
    5. 梯度裁剪

    Returns
    -------
    dict with 'loss_total', 'loss_ce', 'loss_mmd', 'loss_ent', 'loss_pl'
    """
    model.train()
    optimizer.zero_grad()

    # ── 前向传播（获取隐层嵌入 + logits）──────────────────
    h, log_probs = model.encode(data)

    # ── 1. 监督 CE 损失 ──────────────────────────────────
    loss_ce = F.nll_loss(
        log_probs[data.train_mask],
        data.y[data.train_mask],
        weight=class_weights.to(data.x.device),
    )

    # ── 2. MMD 损失 ──────────────────────────────────────
    # 随机采样 scRNA 和 Xenium 节点的隐层表示，避免 O(n^2) 内存
    scrna_idx  = data.train_mask.nonzero(as_tuple=True)[0]
    xenium_idx = data.xenium_mask.nonzero(as_tuple=True)[0]

    n_sample = min(512, len(scrna_idx), len(xenium_idx))
    s_sample = scrna_idx[torch.randperm(len(scrna_idx))[:n_sample]]
    t_sample = xenium_idx[torch.randperm(len(xenium_idx))[:n_sample]]

    loss_mmd = mmd_loss(h[s_sample], h[t_sample])

    # ── 3. 熵正则化 ───────────────────────────────────────
    loss_ent = entropy_regularization(log_probs[data.xenium_mask])

    # ── 4. 伪标签损失 ─────────────────────────────────────
    loss_pl = torch.tensor(0.0, device=data.x.device)
    if cached_pseudo_labels is not None and cached_pseudo_mask is not None:
        xen_global_idx = data.xenium_mask.nonzero(as_tuple=True)[0]
        high_conf_idx  = xen_global_idx[cached_pseudo_mask]
        if high_conf_idx.numel() > 0:
            loss_pl = F.nll_loss(
                log_probs[high_conf_idx],
                cached_pseudo_labels[cached_pseudo_mask].to(data.x.device),
                weight=class_weights.to(data.x.device),
            )

    # ── 5. 合并损失 ───────────────────────────────────────
    total_loss = (
        loss_ce
        + lambda_mmd * loss_mmd
        + lambda_ent * loss_ent
        + lambda_pl  * loss_pl
    )

    total_loss.backward()

    # ── 梯度裁剪（P2-⑩，GAT 尤其重要）──────────────────────
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    optimizer.step()

    return {
        "loss_total": total_loss.item(),
        "loss_ce":    loss_ce.item(),
        "loss_mmd":   loss_mmd.item(),
        "loss_ent":   loss_ent.item(),
        "loss_pl":    loss_pl.item(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    class_weights: torch.Tensor,
) -> dict:
    """
    在 train_mask 和 val_mask 上评估模型。
    仅用 scRNA 数据（绝不用 Xenium 标签）。
    """
    model.eval()
    log_probs = model(data)

    def _acc_f1(mask):
        pred  = log_probs[mask].argmax(dim=1).cpu().numpy()
        truth = data.y[mask].cpu().numpy()
        acc   = (pred == truth).mean()
        from sklearn.metrics import f1_score
        f1_m  = f1_score(truth, pred, average="macro",    zero_division=0)
        f1_w  = f1_score(truth, pred, average="weighted", zero_division=0)
        return acc, f1_m, f1_w

    t_acc, t_f1m, t_f1w = _acc_f1(data.train_mask)
    v_acc, v_f1m, v_f1w = _acc_f1(data.val_mask)

    # 验证集 CE 损失（用于 LR 调度器）
    val_loss = F.nll_loss(
        log_probs[data.val_mask],
        data.y[data.val_mask],
        weight=class_weights.to(data.x.device),
    ).item()

    return {
        "train_acc": t_acc, "train_f1_macro": t_f1m, "train_f1_weighted": t_f1w,
        "val_acc":   v_acc, "val_f1_macro":   v_f1m, "val_f1_weighted":   v_f1w,
        "val_loss":  val_loss,
    }


@torch.no_grad()
def predict_xenium(
    model: nn.Module,
    data: Data,
    cell_types: list[str],
) -> dict:
    """
    推断 Xenium 节点的细胞类型预测和概率。

    Returns
    -------
    dict with 'labels' (str list), 'indices' (int array), 'probs' (ndarray),
              'embeddings' (hidden layer ndarray, for UMAP/eval)
    """
    model.eval()
    h, log_probs = model.encode(data)

    xen_probs  = log_probs[data.xenium_mask].exp().cpu().numpy()
    xen_idx    = xen_probs.argmax(axis=1)
    xen_labels = [cell_types[i] for i in xen_idx]
    xen_h      = h[data.xenium_mask].cpu().numpy()

    return {
        "labels":     xen_labels,
        "indices":    xen_idx,
        "probs":      xen_probs,
        "embeddings": xen_h,
        "confidence": xen_probs.max(axis=1),
    }


# ══════════════════════════════════════════════════════════
# 3. 完整实验入口
# ══════════════════════════════════════════════════════════

def run_experiment(
    model_class,
    model_name: str,
    data: Data,
    cell_types: list[str],
    params: dict,
    class_weights: torch.Tensor,
) -> dict:
    """
    完整训练一个 GNN 模型，含所有 DA 机制。

    训练策略：
    - Epoch 1~warmup    : 仅 CE + MMD（稳定特征空间）
    - Epoch warmup~end  : CE + MMD + 熵正则 + 伪标签（全量 DA）
    - LR 调度器监控 val_loss

    Parameters
    ----------
    params : 全局超参字典（见 train.ipynb）
    """
    device = params["device"]
    from gpu_utils import get_mem_info, vram_str
    _m = get_mem_info(device)
    print(f"\n{'='*55}")
    print(f"  {model_name}  |  GPU 空闲 {_m['free']:.1f}/{_m['total']:.0f} GB")
    print(f"{'='*55}")

    # ── 初始化 ───────────────────────────────────────────
    model = model_class(
        in_dim=data.x.shape[1],
        hidden_dim=params["hidden_dim"],
        out_dim=len(cell_types),
        dropout=params["dropout"],
        proj_dim=params.get("proj_dim"),
    ).to(device)

    data = data.to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    # P2-⑨ 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params.get("lr_factor", 0.5),
        patience=params.get("lr_patience", 15),
        min_lr=params.get("min_lr", 1e-5),
    )

    best_val_f1   = 0.0
    best_state    = None
    patience_cnt  = 0
    warmup_epochs = params.get("warmup_epochs", 30)
    history       = []

    # 伪标签缓存（每隔 pl_update_freq epoch 刷新一次）
    pseudo_labels_cache = None
    pseudo_mask_cache   = None
    pl_update_freq      = params.get("pl_update_freq", 10)

    pbar = tqdm(
        range(1, params["n_epochs"] + 1),
        desc=f"{model_name}",
        unit="ep",
        ncols=110,
        colour="green",
    )
    for epoch in pbar:

        # ── 刷新伪标签 ─────────────────────────────────
        use_pl = epoch > warmup_epochs
        if use_pl and (epoch - warmup_epochs) % pl_update_freq == 1:
            model.eval()
            with torch.no_grad():
                _, lp = model.encode(data)
            pl, pm, _ = get_pseudo_labels(
                lp[data.xenium_mask], threshold=params.get("pl_threshold", 0.90)
            )
            pseudo_labels_cache = pl.detach()
            pseudo_mask_cache   = pm.detach()
            n_pl = pm.sum().item()
            if n_pl > 0:
                tqdm.write(f"  [Ep {epoch:3d}] 伪标签更新：{n_pl:,} 个 "
                           f"({100*n_pl/data.xenium_mask.sum().item():.1f}%)")

        # ── 训练一步 ───────────────────────────────────
        losses = train_epoch(
            model, data, optimizer, class_weights,
            lambda_mmd=params.get("lambda_mmd", 0.1),
            lambda_ent=params.get("lambda_ent", 0.01) if use_pl else 0.0,
            lambda_pl=params.get("lambda_pl",  0.3)  if use_pl else 0.0,
            pl_threshold=params.get("pl_threshold", 0.90),
            max_grad_norm=params.get("max_grad_norm", 1.0),
            cached_pseudo_labels=pseudo_labels_cache if use_pl else None,
            cached_pseudo_mask=pseudo_mask_cache     if use_pl else None,
        )

        # ── 评估 ───────────────────────────────────────
        metrics = evaluate(model, data, class_weights)
        scheduler.step(metrics["val_loss"])

        log = {**losses, **metrics, "epoch": epoch,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(log)

        # ── Early stopping（基于 val F1 macro）─────────
        if metrics["val_f1_macro"] > best_val_f1:
            best_val_f1 = metrics["val_f1_macro"]
            best_state  = save_best_state(model)        # deepcopy，修复 P0-②
            patience_cnt = 0
        else:
            patience_cnt += 1

        # ── 更新进度条 postfix（每 epoch 都更新）─────────
        _vram = vram_str(device)
        pbar.set_postfix({
            "CE":    f"{losses['loss_ce']:.3f}",
            "MMD":   f"{losses['loss_mmd']:.4f}",
            "F1":    f"{metrics['val_f1_macro']:.3f}",
            "VRAM":  _vram,
            "lr":    f"{log['lr']:.1e}",
            "pat":   f"{patience_cnt}/{params['patience']}",
        }, refresh=True)

        if patience_cnt >= params["patience"]:
            tqdm.write(f"  Early stop @ ep {epoch}  best F1={best_val_f1:.4f}")
            break

    # ── 加载最优权重 + 预测 ───────────────────────────────
    model.load_state_dict(best_state)
    preds = predict_xenium(model, data, cell_types)

    pbar.close()
    print(f"\n  ✅ {model_name} 完成  Best Val F1 = {best_val_f1:.4f}")

    return {
        "model_name":    model_name,
        "model":         model,
        "best_val_f1":   best_val_f1,
        "history":       history,
        "predictions":   preds["labels"],
        "pred_indices":  preds["indices"],
        "probabilities": preds["probs"],
        "embeddings":    preds["embeddings"],
        "confidence":    preds["confidence"],
    }

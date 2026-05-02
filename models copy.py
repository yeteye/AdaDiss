"""
models.py — GNN 模型定义 + 半监督 Domain Adaptation 训练循环

修复清单
--------
P0-②  state_dict 浅拷贝  → save_best_state()（deepcopy）
P0-③  data.xenium_mask   → data.spot_mask（与 utils_spot.py 统一）
P1-⑤  缺 DA 机制         → MMD + 熵正则 + 伪标签
P1-⑦  无类别权重         → class_weights 传入 nll_loss
P2-⑨  无学习率调度器     → ReduceLROnPlateau
P2-⑩  无梯度裁剪         → clip_grad_norm_
P2-⑪  无批归一化         → BatchNorm1d
P3-⑫  无伪标签           → get_pseudo_labels
P3-⑭  维度跨度大         → proj_dim 投影层
新增   save_dir 权重保存  → {model_name}_best.pt
新增   scrna_embeddings   → predict_xenium 同时返回 scRNA 嵌入（供 eval Fig 2）
"""

import os
import copy
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

    参数
    ----
    in_dim     : 输入特征维度
    hidden_dim : 隐层维度
    out_dim    : 输出类别数
    dropout    : Dropout 概率
    proj_dim   : 可选投影层维度（None=不用），用于缓解高维输入跨度问题
    """

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
        """返回 (隐层嵌入 h, log-softmax 输出)"""
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


class GraphSAGE(nn.Module):
    """GraphSAGE，更适合归纳式推断（inductive）。结构同 GCN。"""

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
    """图注意力网络，4头注意力。"""

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

        self.conv1 = GATConv(conv_in, hidden_dim,
                             heads=heads, dropout=dropout, concat=True)
        self.bn1   = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim,
                             heads=1, dropout=dropout, concat=False)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, out_dim,
                             heads=1, dropout=dropout, concat=False)

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
    lambda_mmd: float = 0.1,
    lambda_ent: float = 0.01,
    lambda_pl:  float = 0.3,
    max_grad_norm: float = 1.0,
    cached_pseudo_labels: torch.Tensor | None = None,
    cached_pseudo_mask: torch.Tensor | None = None,
) -> dict:
    """
    一个完整训练 epoch。

    修复：所有 data.xenium_mask → data.spot_mask
    """
    model.train()
    optimizer.zero_grad()

    h, log_probs = model.encode(data)

    # 1. 监督 CE 损失（scRNA 有标签节点）
    loss_ce = F.nll_loss(
        log_probs[data.train_mask],
        data.y[data.train_mask],
        weight=class_weights.to(data.x.device),
    )

    # 2. MMD 损失（随机子采样，避免 O(n²) 显存）
    scrna_idx  = data.train_mask.nonzero(as_tuple=True)[0]
    spot_idx   = data.spot_mask.nonzero(as_tuple=True)[0]   # 修复：spot_mask

    n_sample = min(512, len(scrna_idx), len(spot_idx))
    s_sample = scrna_idx[torch.randperm(len(scrna_idx))[:n_sample]]
    t_sample = spot_idx[torch.randperm(len(spot_idx))[:n_sample]]

    loss_mmd = mmd_loss(h[s_sample], h[t_sample])

    # 3. 熵正则化（Xenium spot 节点）
    loss_ent = entropy_regularization(log_probs[data.spot_mask])  # 修复

    # 4. 伪标签损失
    loss_pl = torch.tensor(0.0, device=data.x.device)
    if cached_pseudo_labels is not None and cached_pseudo_mask is not None:
        spot_global_idx = data.spot_mask.nonzero(as_tuple=True)[0]  # 修复
        high_conf_idx   = spot_global_idx[cached_pseudo_mask]
        if high_conf_idx.numel() > 0:
            loss_pl = F.nll_loss(
                log_probs[high_conf_idx],
                cached_pseudo_labels[cached_pseudo_mask].to(data.x.device),
                weight=class_weights.to(data.x.device),
            )

    total_loss = (
        loss_ce
        + lambda_mmd * loss_mmd
        + lambda_ent * loss_ent
        + lambda_pl  * loss_pl
    )

    total_loss.backward()
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
    """在 train_mask 和 val_mask 上评估（仅 scRNA，不用 Xenium 标签）。"""
    model.eval()
    log_probs = model(data)

    def _acc_f1(mask):
        pred  = log_probs[mask].argmax(dim=1).cpu().numpy()
        truth = data.y[mask].cpu().numpy()
        acc   = (pred == truth).mean()
        from sklearn.metrics import f1_score
        f1_m  = f1_score(truth, pred, average="macro",    zero_division=0)
        f1_w  = f1_score(truth, pred, average="weighted", zero_division=0)
        return float(acc), float(f1_m), float(f1_w)

    t_acc, t_f1m, t_f1w = _acc_f1(data.train_mask)
    v_acc, v_f1m, v_f1w = _acc_f1(data.val_mask)

    val_loss = F.nll_loss(
        log_probs[data.val_mask],
        data.y[data.val_mask],
        weight=class_weights.to(data.x.device),
    ).item()

    return {
        "train_acc":         t_acc,
        "train_f1_macro":    t_f1m,
        "train_f1_weighted": t_f1w,
        "val_acc":           v_acc,
        "val_f1_macro":      v_f1m,
        "val_f1_weighted":   v_f1w,
        "val_loss":          val_loss,
    }


@torch.no_grad()
def predict_xenium(
    model: nn.Module,
    data: Data,
    cell_types: list[str],
) -> dict:
    """
    推断所有 spot 节点的细胞类型。

    修复：
    - xenium_mask → spot_mask
    - 同时返回 scrna_embeddings（供 eval.py Fig 2 UMAP 使用）
    """
    model.eval()
    h, log_probs = model.encode(data)

    # Spot 节点预测
    spot_probs  = log_probs[data.spot_mask].exp().cpu().numpy()   # 修复
    spot_idx    = spot_probs.argmax(axis=1)
    spot_labels = [cell_types[i] for i in spot_idx]
    spot_h      = h[data.spot_mask].cpu().numpy()                  # 修复

    # scRNA 节点嵌入（用于 Fig 2 域对齐 UMAP）
    n_scrna    = getattr(data, "n_scrna", data.train_mask.shape[0])
    scrna_h    = h[:n_scrna].cpu().numpy()

    return {
        "labels":          spot_labels,
        "indices":         spot_idx,
        "probs":           spot_probs,
        "embeddings":      spot_h,
        "scrna_embeddings": scrna_h,
        "confidence":      spot_probs.max(axis=1),
    }


# ══════════════════════════════════════════════════════════
# 3. 完整实验入口
# ══════════════════════════════════════════════════════════

def run_experiment(
    model: nn.Module,
    model_name: str,
    data: Data,
    cell_types: list[str],
    params: dict,
    class_weights: torch.Tensor,
    save_dir: str | None = None,
) -> dict:
    """
    完整训练一个 GNN 模型（接收已实例化的 model）。

    修复：
    - 接收 model 实例而非 model_class（与 notebook 调用一致）
    - 新增 save_dir：保存 {model_name}_best.pt
    - 返回 best_val_acc、best_epoch
    - 修复所有 xenium_mask → spot_mask
    """
    device = params.get("device", "cpu")
    from gpu_utils import get_mem_info, vram_str
    _m = get_mem_info(device)
    print(f"\n{'='*55}")
    print(f"  {model_name}  |  GPU 空闲 {_m['free']:.1f}/{_m['total']:.0f} GB")
    print(f"{'='*55}")

    model         = model.to(device)
    data          = data.to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params.get("lr_factor", 0.5),
        patience=params.get("lr_patience", 15),
        min_lr=params.get("min_lr", 1e-5),
    )

    best_val_f1   = 0.0
    best_val_acc  = 0.0
    best_epoch    = 0
    best_state    = None
    patience_cnt  = 0
    warmup_epochs = params.get("warmup_epochs", 30)
    history       = []

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

        # ── 刷新伪标签 ───────────────────────────────────
        use_pl = epoch > warmup_epochs
        if use_pl and (epoch - warmup_epochs) % pl_update_freq == 1:
            model.eval()
            with torch.no_grad():
                _, lp = model.encode(data)
            pl, pm, _ = get_pseudo_labels(
                lp[data.spot_mask], threshold=params.get("pl_threshold", 0.90)  # 修复
            )
            pseudo_labels_cache = pl.detach()
            pseudo_mask_cache   = pm.detach()
            n_pl = pm.sum().item()
            if n_pl > 0:
                total_spot = data.spot_mask.sum().item()  # 修复
                tqdm.write(f"  [Ep {epoch:3d}] 伪标签更新：{n_pl:,} 个 "
                           f"({100*n_pl/total_spot:.1f}%)")

        # ── 训练一步 ─────────────────────────────────────
        losses = train_epoch(
            model, data, optimizer, class_weights,
            lambda_mmd=params.get("lambda_mmd", 0.1),
            lambda_ent=params.get("lambda_ent", 0.01) if use_pl else 0.0,
            lambda_pl=params.get("lambda_pl",  0.3)   if use_pl else 0.0,
            max_grad_norm=params.get("max_grad_norm", 1.0),
            cached_pseudo_labels=pseudo_labels_cache if use_pl else None,
            cached_pseudo_mask=pseudo_mask_cache     if use_pl else None,
        )

        # ── 评估 ─────────────────────────────────────────
        metrics = evaluate(model, data, class_weights)
        scheduler.step(metrics["val_loss"])

        log = {
            **losses, **metrics,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(log)

        # ── Early stopping ───────────────────────────────
        if metrics["val_f1_macro"] > best_val_f1:
            best_val_f1  = metrics["val_f1_macro"]
            best_val_acc = metrics["val_acc"]
            best_epoch   = epoch
            best_state   = save_best_state(model)   # deepcopy，修复 P0-②
            patience_cnt = 0
        else:
            patience_cnt += 1

        pbar.set_postfix({
            "CE":   f"{losses['loss_ce']:.3f}",
            "MMD":  f"{losses['loss_mmd']:.4f}",
            "F1":   f"{metrics['val_f1_macro']:.3f}",
            "VRAM": vram_str(device),
            "lr":   f"{log['lr']:.1e}",
            "pat":  f"{patience_cnt}/{params['patience']}",
        }, refresh=True)

        if patience_cnt >= params["patience"]:
            tqdm.write(f"  Early stop @ ep {epoch}  best F1={best_val_f1:.4f}")
            break

    # ── 保存最优权重 ─────────────────────────────────────
    if save_dir is not None and best_state is not None:
        os.makedirs(save_dir, exist_ok=True)
        weight_path = os.path.join(save_dir, f"{model_name}_best.pt")
        torch.save(best_state, weight_path)
        tqdm.write(f"  💾 权重已保存: {weight_path}")

    # ── 加载最优权重 + 推断 ──────────────────────────────
    model.load_state_dict(best_state)
    preds = predict_xenium(model, data, cell_types)

    pbar.close()
    print(f"\n  ✅ {model_name} 完成  Best Val F1={best_val_f1:.4f}  "
          f"Acc={best_val_acc:.4f}  Epoch={best_epoch}")

    return {
        "model_name":       model_name,
        "model":            model,
        "best_val_f1":      best_val_f1,
        "best_val_acc":     best_val_acc,
        "best_epoch":       best_epoch,
        "history":          history,
        "predictions":      preds["labels"],
        "pred_indices":     preds["indices"],
        "probabilities":    preds["probs"],
        "embeddings":       preds["embeddings"],
        "scrna_embeddings": preds["scrna_embeddings"],
        "confidence":       preds["confidence"],
    }

"""
models_amp.py — 显存优化模型封装 + Notebook 接口适配层

职责
----
1. 暴露 GCN_AMP / GraphSAGE_AMP / GAT_AMP：
   - 参数名与 Notebook Cell 4a 一致（in_channels / hidden_channels / out_channels）
   - 内部继承 models.py 中的 GCN / GraphSAGE 和本文件的 GATMemEfficient
2. GAT 显存优化（GATMemEfficient）：
   - 支持 torch.utils.checkpoint（梯度检查点，OOM 时使用）
   - 支持 AMP 混合精度

修复清单
--------
- data.xenium_mask → data.spot_mask（与 utils_spot.py 统一）
- run_gat_amp history 增加 train_f1_macro 字段（eval.py plot_training_curves 必需）
- run_gat_amp 返回值增加 scrna_embeddings（eval.py Fig 2 必需）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from tqdm.auto import tqdm

# 从 models.py 导入基础类（GCN_AMP / GraphSAGE_AMP 直接复用）
from models import GCN, GraphSAGE
from utils import mmd_loss, entropy_regularization, get_pseudo_labels, save_best_state


# ══════════════════════════════════════════════════════════
# 1. Notebook 接口适配层
#    统一参数名：in_channels / hidden_channels / out_channels
# ══════════════════════════════════════════════════════════

class GCN_AMP(GCN):
    """
    GCN 的 Notebook 接口封装。
    参数名与 Cell 4a 一致，内部调用 models.GCN。
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        proj_dim: int | None = None,
        dropout: float = 0.5,
    ):
        super().__init__(
            in_dim=in_channels,
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            dropout=dropout,
            proj_dim=proj_dim,
        )


class GraphSAGE_AMP(GraphSAGE):
    """
    GraphSAGE 的 Notebook 接口封装。
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        proj_dim: int | None = None,
        dropout: float = 0.5,
    ):
        super().__init__(
            in_dim=in_channels,
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            dropout=dropout,
            proj_dim=proj_dim,
        )


# ══════════════════════════════════════════════════════════
# 2. 显存优化版 GAT（梯度检查点 + AMP）
# ══════════════════════════════════════════════════════════

class GATMemEfficient(nn.Module):
    """
    显存优化版 GAT：
    - 支持 torch.utils.checkpoint（OOM 时使用）
    - 与 models.GAT 接口一致
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.5,
        proj_dim: int | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dropout        = dropout
        self.use_checkpoint = use_checkpoint

        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim), nn.ReLU(), nn.BatchNorm1d(proj_dim)
            )
            conv_in = proj_dim
        else:
            self.proj = None
            conv_in   = in_dim

        self.conv1 = GATConv(conv_in, hidden_dim,
                             heads=heads, dropout=dropout, concat=True)
        self.bn1   = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim,
                             heads=1, dropout=dropout, concat=False)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, out_dim,
                             heads=1, dropout=dropout, concat=False)

    def _forward_body(self, x, edge_index):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.conv2(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv3(h, edge_index)
        return h, out

    def encode(self, data: Data):
        x, ei = data.x, data.edge_index
        if self.proj is not None:
            x = self.proj(x)

        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            h_dim = self.conv2.out_channels

            def fwd(x_, ei_):
                h_, out_ = self._forward_body(x_, ei_)
                return torch.cat([h_, out_], dim=1)

            combined = checkpoint(fwd, x, ei, use_reentrant=False)
            h   = combined[:, :h_dim]
            out = combined[:, h_dim:]
        else:
            h, out = self._forward_body(x, ei)

        return h, F.log_softmax(out, dim=1)

    def forward(self, data: Data):
        _, lp = self.encode(data)
        return lp


class GAT_AMP(GATMemEfficient):
    """
    GAT 的 Notebook 接口封装（使用 GATMemEfficient 实现）。
    参数名与 Cell 4a 一致：in_channels / hidden_channels / out_channels。
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        proj_dim: int | None = None,
        dropout: float = 0.5,
        heads: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__(
            in_dim=in_channels,
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            heads=heads,
            dropout=dropout,
            proj_dim=proj_dim,
            use_checkpoint=use_checkpoint,
        )


# ══════════════════════════════════════════════════════════
# 3. AMP 版 GAT 专用训练循环（OOM 时替代 run_experiment）
#    正常情况下 Cell 4b 不调用此函数，改用 utils.run_experiment
# ══════════════════════════════════════════════════════════

def run_gat_amp(
    data: Data,
    cell_types: list,
    params: dict,
    class_weights: torch.Tensor,
) -> dict:
    """
    带 AMP 混合精度的 GAT 训练（OOM 应急使用）。

    用法：当 Cell 4b 中 GAT 出现 CUDA OOM 时，
    单独对 GAT 调用此函数替代 run_experiment。

    修复：
    - data.xenium_mask → data.spot_mask（5 处）
    - history 增加 train_f1_macro（eval.py 必需）
    - 返回值增加 scrna_embeddings（eval.py Fig 2 必需）
    """
    import os
    device       = params["device"]
    use_amp      = params.get("use_amp",   True)
    gat_heads    = params.get("gat_heads", 4)
    use_ckpt     = params.get("use_ckpt",  False)
    save_dir     = params.get("save_dir",  None)

    from gpu_utils import get_mem_info, vram_str
    _m = get_mem_info(device)
    _flags = f"AMP={'on' if use_amp else 'off'}  heads={gat_heads}  ckpt={'on' if use_ckpt else 'off'}"
    print(f"\n{'='*55}")
    print(f"  GAT ({_flags})")
    print(f"  GPU 空闲 {_m['free']:.1f}/{_m['total']:.0f} GB")
    print(f"{'='*55}")

    model = GATMemEfficient(
        in_dim         = data.x.shape[1],
        hidden_dim     = params["hidden_dim"],
        out_dim        = len(cell_types),
        heads          = gat_heads,
        dropout        = params["dropout"],
        proj_dim       = params.get("proj_dim"),
        use_checkpoint = use_ckpt,
    ).to(device)

    data          = data.to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor   = params.get("lr_factor",   0.5),
        patience = params.get("lr_patience", 15),
        min_lr   = params.get("min_lr",      1e-5),
    )
    scaler = GradScaler(enabled=use_amp)

    best_val_f1  = 0.0
    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None
    patience_cnt = 0
    warmup       = params.get("warmup_epochs", 30)
    history      = []

    pseudo_labels_cache = None
    pseudo_mask_cache   = None
    pl_freq = params.get("pl_update_freq", 10)

    pbar = tqdm(
        range(1, params["n_epochs"] + 1),
        desc="GAT(AMP)", unit="ep", ncols=110, colour="cyan",
    )
    for epoch in pbar:
        use_da = epoch > warmup

        # ── 更新伪标签 ────────────────────────────────────
        if use_da and (epoch - warmup) % pl_freq == 1:
            model.eval()
            with torch.no_grad():
                with autocast(enabled=use_amp):
                    _, lp = model.encode(data)
            pl, pm, _ = get_pseudo_labels(
                lp[data.spot_mask], threshold=params.get("pl_threshold", 0.90)  # 修复
            )
            pseudo_labels_cache = pl.detach()
            pseudo_mask_cache   = pm.detach()
            if pm.sum() > 0:
                total_spot = data.spot_mask.sum().item()  # 修复
                tqdm.write(f"  [Ep {epoch:3d}] 伪标签更新：{pm.sum():,} 个 "
                           f"({100*pm.float().mean():.1f}%)")

        # ── 训练一步 ──────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            h, log_probs = model.encode(data)

            loss_ce = F.nll_loss(
                log_probs[data.train_mask],
                data.y[data.train_mask],
                weight=class_weights,
            )

            scrna_idx = data.train_mask.nonzero(as_tuple=True)[0]
            spot_idx  = data.spot_mask.nonzero(as_tuple=True)[0]   # 修复
            n_s = min(512, len(scrna_idx), len(spot_idx))
            s_s = scrna_idx[torch.randperm(len(scrna_idx))[:n_s]]
            t_s = spot_idx[torch.randperm(len(spot_idx))[:n_s]]
            loss_mmd = mmd_loss(h[s_s], h[t_s])

            loss_ent = torch.tensor(0.0, device=device)
            loss_pl  = torch.tensor(0.0, device=device)
            if use_da:
                loss_ent = entropy_regularization(log_probs[data.spot_mask])  # 修复
                if pseudo_labels_cache is not None and pseudo_mask_cache is not None:
                    spot_global = spot_idx[pseudo_mask_cache]   # 修复
                    if spot_global.numel() > 0:
                        loss_pl = F.nll_loss(
                            log_probs[spot_global],
                            pseudo_labels_cache[pseudo_mask_cache].to(device),
                            weight=class_weights,
                        )

            total_loss = (
                loss_ce
                + params.get("lambda_mmd", 0.1) * loss_mmd
                + (params.get("lambda_ent", 0.01) * loss_ent if use_da else 0)
                + (params.get("lambda_pl",  0.30) * loss_pl  if use_da else 0)
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.get("max_grad_norm", 1.0))
        scaler.step(optimizer)
        scaler.update()

        # ── 评估 ──────────────────────────────────────────
        model.eval()
        with torch.no_grad(), autocast(enabled=use_amp):
            lp_eval = model(data)

        def _f1_acc(mask):
            pred  = lp_eval[mask].argmax(dim=1).cpu().numpy()
            truth = data.y[mask].cpu().numpy()
            from sklearn.metrics import f1_score
            acc  = float((pred == truth).mean())
            f1_m = float(f1_score(truth, pred, average="macro", zero_division=0))
            return acc, f1_m

        train_acc, train_f1 = _f1_acc(data.train_mask)
        val_acc,   val_f1   = _f1_acc(data.val_mask)
        val_loss = F.nll_loss(
            lp_eval[data.val_mask],
            data.y[data.val_mask],
            weight=class_weights,
        ).item()
        scheduler.step(val_loss)

        log = {
            "epoch":          epoch,
            "loss_ce":        loss_ce.item(),
            "loss_mmd":       loss_mmd.item(),
            "loss_ent":       loss_ent.item() if use_da else 0.0,
            "loss_pl":        loss_pl.item()  if use_da else 0.0,
            "train_acc":      train_acc,
            "train_f1_macro": train_f1,        # 修复：eval.py 必需
            "val_acc":        val_acc,
            "val_f1_macro":   val_f1,
            "val_loss":       val_loss,
            "lr":             optimizer.param_groups[0]["lr"],
        }
        history.append(log)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = save_best_state(model)
            patience_cnt = 0
        else:
            patience_cnt += 1

        pbar.set_postfix({
            "CE":   f"{loss_ce.item():.3f}",
            "MMD":  f"{loss_mmd.item():.4f}",
            "F1":   f"{val_f1:.4f}",
            "VRAM": vram_str(device),
            "lr":   f"{optimizer.param_groups[0]['lr']:.1e}",
            "pat":  f"{patience_cnt}/{params['patience']}",
        }, refresh=True)

        if patience_cnt >= params["patience"]:
            tqdm.write(f"  Early stop @ ep {epoch}  best F1={best_val_f1:.4f}")
            break

    # ── 保存权重 ──────────────────────────────────────────
    if save_dir is not None and best_state is not None:
        os.makedirs(save_dir, exist_ok=True)
        weight_path = os.path.join(save_dir, "GAT_best.pt")
        torch.save(best_state, weight_path)
        tqdm.write(f"  💾 权重已保存: {weight_path}")

    # ── 加载最优权重 + 推断 ───────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), autocast(enabled=use_amp):
        h_final, lp_final = model.encode(data)

    spot_proba  = lp_final[data.spot_mask].float().exp().cpu().numpy()   # 修复
    spot_idx_   = spot_proba.argmax(axis=1)
    spot_labels = [cell_types[i] for i in spot_idx_]
    spot_h      = h_final[data.spot_mask].float().cpu().numpy()           # 修复

    # scRNA 嵌入（供 eval.py Fig 2）
    n_scrna = getattr(data, "n_scrna", data.train_mask.shape[0])
    scrna_h = h_final[:n_scrna].float().cpu().numpy()                     # 修复

    pbar.close()
    print(f"\n  ✅ GAT(AMP) 完成  Best Val F1={best_val_f1:.4f}  "
          f"Acc={best_val_acc:.4f}  Epoch={best_epoch}")

    return {
        "model_name":       "GAT",
        "model":            model,
        "best_val_f1":      best_val_f1,
        "best_val_acc":     best_val_acc,
        "best_epoch":       best_epoch,
        "history":          history,
        "predictions":      spot_labels,
        "pred_indices":     spot_idx_,
        "probabilities":    spot_proba,
        "embeddings":       spot_h,
        "scrna_embeddings": scrna_h,
        "confidence":       spot_proba.max(axis=1),
    }

"""
models_amp.py — 对 models.py 的 GAT 部分进行显存优化补丁

包含三个级别的优化：
  Level 1 (AMP)         : 混合精度，显存减少约 45%，几乎无精度损失
  Level 2 (AMP + 减头)  : 头数 4→2，显存再减约 30%
  Level 3 (AMP + 检查点): 梯度检查点，以时间换显存（慢约 20%）

在 Notebook 中使用方式：
  # 在 Cell 1 的 PARAMS 里加入:
  PARAMS["use_amp"]  = True   # 开启混合精度（首选）
  PARAMS["gat_heads"] = 2     # 减少注意力头（OOM 时再用）
  PARAMS["use_ckpt"] = False  # 梯度检查点（最极端情况）

  # Cell 4c 改为调用本文件的 run_gat_amp()
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from utils import mmd_loss, entropy_regularization, get_pseudo_labels, save_best_state
from tqdm.auto import tqdm


# ══════════════════════════════════════════════════════════
# 单卡显存优化版 GAT
# ══════════════════════════════════════════════════════════

class GATMemEfficient(nn.Module):
    """
    显存优化版 GAT：
    - 支持 torch.utils.checkpoint（梯度检查点）
    - heads 可配置（建议 OOM 时从 4 降到 2）
    - 与原 GAT 接口完全一致
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 heads=4, dropout=0.5, proj_dim=None, use_checkpoint=False):
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
        """可被梯度检查点包裹的前向主体"""
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
            # checkpoint 不支持多返回值，需拆分
            def fwd(x_, ei_):
                h_, out_ = self._forward_body(x_, ei_)
                # 拼接后返回，checkpoint 只接受 tensor 输出
                return torch.cat([h_, out_], dim=1)

            combined = checkpoint(fwd, x, ei, use_reentrant=False)
            h_dim    = self.conv2.out_channels
            h        = combined[:, :h_dim]
            out      = combined[:, h_dim:]
        else:
            h, out = self._forward_body(x, ei)

        return h, F.log_softmax(out, dim=1)

    def forward(self, data: Data):
        _, lp = self.encode(data)
        return lp


# ══════════════════════════════════════════════════════════
# AMP 版训练循环
# ══════════════════════════════════════════════════════════

def run_gat_amp(
    data: Data,
    cell_types: list,
    params: dict,
    class_weights: torch.Tensor,
) -> dict:
    """
    带 AMP 混合精度的 GAT 训练入口。
    接口与原 run_experiment() 完全一致，直接替换 Cell 4c。

    params 中新增字段（可选，有默认值）：
      use_amp   : bool = True   开启 AMP
      gat_heads : int  = 4      GAT 注意力头数
      use_ckpt  : bool = False  梯度检查点
    """
    device       = params["device"]
    use_amp      = params.get("use_amp",   True)
    gat_heads    = params.get("gat_heads", 4)
    use_ckpt     = params.get("use_ckpt",  False)

    from gpu_utils import get_mem_info, vram_str
    _m = get_mem_info(device)
    _flags = f"AMP={'on' if use_amp else 'off'}  heads={gat_heads}  ckpt={'on' if use_ckpt else 'off'}"
    print(f"\n{'='*55}")
    print(f"  GAT ({_flags})")
    print(f"  GPU 空闲 {_m['free']:.1f}/{_m['total']:.0f} GB")
    if use_amp:  print(f"  混合精度开启 → 显存减少 ~45%")
    if use_ckpt: print(f"  梯度检查点开启 → 再减 ~30%，速度慢 20%")
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
        factor  = params.get("lr_factor",   0.5),
        patience= params.get("lr_patience", 15),
        min_lr  = params.get("min_lr",      1e-5),
    )
    scaler = GradScaler(enabled=use_amp)   # AMP 梯度缩放器

    best_val_f1  = 0.0
    best_state   = None
    patience_cnt = 0
    warmup       = params.get("warmup_epochs", 30)
    history      = []

    pseudo_labels_cache = None
    pseudo_mask_cache   = None
    pl_freq = params.get("pl_update_freq", 10)

    pbar = tqdm(
        range(1, params["n_epochs"] + 1),
        desc="GAT(AMP)",
        unit="ep",
        ncols=110,
        colour="cyan",
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
                lp[data.xenium_mask], threshold=params.get("pl_threshold", 0.90)
            )
            pseudo_labels_cache = pl.detach()
            pseudo_mask_cache   = pm.detach()
            if pm.sum() > 0:
                tqdm.write(f"  [Ep {epoch:3d}] 伪标签更新：{pm.sum():,} 个 "
                           f"({100*pm.float().mean():.1f}%)")

        # ── 训练一步 ──────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            h, log_probs = model.encode(data)

            # CE 损失
            loss_ce = F.nll_loss(
                log_probs[data.train_mask],
                data.y[data.train_mask],
                weight=class_weights,
            )

            # MMD 损失
            scrna_idx  = data.train_mask.nonzero(as_tuple=True)[0]
            xenium_idx = data.xenium_mask.nonzero(as_tuple=True)[0]
            n_s = min(512, len(scrna_idx), len(xenium_idx))
            s_s = scrna_idx[torch.randperm(len(scrna_idx))[:n_s]]
            t_s = xenium_idx[torch.randperm(len(xenium_idx))[:n_s]]
            loss_mmd = mmd_loss(h[s_s], h[t_s])

            # 熵 + 伪标签
            loss_ent = torch.tensor(0.0, device=device)
            loss_pl  = torch.tensor(0.0, device=device)
            if use_da:
                loss_ent = entropy_regularization(log_probs[data.xenium_mask])
                if pseudo_labels_cache is not None and pseudo_mask_cache is not None:
                    xi  = xenium_idx[pseudo_mask_cache]
                    if xi.numel() > 0:
                        loss_pl = F.nll_loss(
                            log_probs[xi],
                            pseudo_labels_cache[pseudo_mask_cache].to(device),
                            weight=class_weights,
                        )

            total_loss = (
                loss_ce
                + params.get("lambda_mmd", 0.1) * loss_mmd
                + (params.get("lambda_ent", 0.01) * loss_ent if use_da else 0)
                + (params.get("lambda_pl",  0.30) * loss_pl  if use_da else 0)
            )

        # AMP 反向 + 梯度裁剪
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), params.get("max_grad_norm", 1.0)
        )
        scaler.step(optimizer)
        scaler.update()

        # ── 评估 ──────────────────────────────────────────
        model.eval()
        with torch.no_grad(), autocast(enabled=use_amp):
            lp_eval = model(data)

        def _f1(mask):
            pred  = lp_eval[mask].argmax(dim=1).cpu().numpy()
            truth = data.y[mask].cpu().numpy()
            from sklearn.metrics import f1_score
            return f1_score(truth, pred, average="macro", zero_division=0)

        val_f1   = _f1(data.val_mask)
        val_loss = F.nll_loss(
            lp_eval[data.val_mask],
            data.y[data.val_mask],
            weight=class_weights,
        ).item()
        scheduler.step(val_loss)

        log = {
            "epoch": epoch, "loss_ce": loss_ce.item(),
            "loss_mmd": loss_mmd.item(), "val_f1_macro": val_f1,
            "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(log)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = save_best_state(model)
            patience_cnt = 0
        else:
            patience_cnt += 1

        # ── 进度条 postfix（每 epoch 更新）────────────
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

    # ── 加载最优权重 + 推断 ───────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), autocast(enabled=use_amp):
        h_final, lp_final = model.encode(data)

    xen_proba  = lp_final[data.xenium_mask].float().exp().cpu().numpy()
    xen_idx    = xen_proba.argmax(axis=1)
    xen_labels = [cell_types[i] for i in xen_idx]
    xen_h      = h_final[data.xenium_mask].float().cpu().numpy()

    pbar.close()
    print(f"\n  ✅ GAT(AMP) 完成  Best Val F1 = {best_val_f1:.4f}")

    return {
        "model_name":   "GAT",
        "model":        model,
        "best_val_f1":  best_val_f1,
        "history":      history,
        "predictions":  xen_labels,
        "pred_indices": xen_idx,
        "probabilities":xen_proba,
        "embeddings":   xen_h,
        "confidence":   xen_proba.max(axis=1),
    }

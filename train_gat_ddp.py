"""
train_gat_ddp.py — 多 GPU DDP + NeighborSampler 训练 GAT

使用方式（在服务器终端，不是 Notebook 里）：
  # 使用全部 8 张卡
  torchrun --nproc_per_node=8 train_gat_ddp.py

  # 只用 4 张卡（推荐先测试）
  torchrun --nproc_per_node=4 train_gat_ddp.py --n_gpus 4

  # 指定卡号 (e.g. 只用 GPU 0,1,2,3)
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_gat_ddp.py

技术要点：
  ① NeighborLoader 把大图切成 mini-batch，每个 batch 只用一部分显存
  ② DDP 让每张卡独立处理一批 batch，通过 all-reduce 同步梯度
  ③ 混合精度 (AMP) 进一步节省显存约 45%
  ④ Domain Adaptation 在 mini-batch 级别计算 MMD / 熵正则 / 伪标签
  ⑤ 训练完后 rank-0 进程负责保存模型和结果
"""

import os
import sys
import copy
import json
import pickle
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# 配置（在这里修改超参）
# ══════════════════════════════════════════════════════════
CFG = {
    # 图缓存路径（与 train.ipynb 一致）
    "cache_file":    "./data/cache/graph/graph_k30_val0.2.pt",
    "scaler_file":   "./data/cache/graph/fitted_scaler.pkl",

    # 模型
    "hidden_dim":    256,
    "proj_dim":      512,
    "heads":         4,
    "dropout":       0.5,

    # 训练
    "n_epochs":      500,
    "lr":            1e-3,
    "weight_decay":  5e-4,
    "patience":      40,
    "lr_factor":     0.5,
    "lr_patience":   15,
    "min_lr":        1e-5,
    "max_grad_norm": 1.0,
    "warmup_epochs": 30,

    # DA
    "lambda_mmd":    0.1,
    "lambda_ent":    0.01,
    "lambda_pl":     0.3,
    "pl_threshold":  0.90,
    "pl_update_freq":10,

    # NeighborSampler
    "num_neighbors":  [25, 15, 10],  # 三层各采样的邻居数
    "batch_size":     2048,          # 每个 GPU 每步处理的种子节点数
    "num_workers":    4,             # DataLoader 并行 worker 数

    # 输出
    "output_dir":    "./results/models/",
    "result_json":   "./results/predictions/gat_ddp_result.json",
}


# ══════════════════════════════════════════════════════════
# GAT 模型（支持 mini-batch）
# ══════════════════════════════════════════════════════════

class GATInductive(nn.Module):
    """
    归纳式 GAT，支持 NeighborLoader mini-batch 推断。

    与全图 transductive GAT 的区别：
    - encode() 接受 (x, edge_index) 而非 Data 对象
    - 可以处理局部子图
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 heads: int = 4, dropout: float = 0.5,
                 proj_dim: int | None = None):
        super().__init__()
        self.dropout = dropout

        # 可选投影层（缓解特征维度跨度过大问题）
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

        self.conv1 = GATConv(conv_in, hidden_dim,
                             heads=heads, dropout=dropout, concat=True)
        self.bn1   = nn.BatchNorm1d(hidden_dim * heads)

        self.conv2 = GATConv(hidden_dim * heads, hidden_dim,
                             heads=1, dropout=dropout, concat=False)
        self.bn2   = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GATConv(hidden_dim, out_dim,
                             heads=1, dropout=dropout, concat=False)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Returns (h, log_probs) where h is the penultimate embedding.
        """
        if self.proj is not None:
            x = self.proj(x)
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.conv2(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv3(h, edge_index)
        return h, F.log_softmax(out, dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        _, log_probs = self.encode(x, edge_index)
        return log_probs


# ══════════════════════════════════════════════════════════
# Domain Adaptation 损失（mini-batch 版本）
# ══════════════════════════════════════════════════════════

def _rbf(x, y, bw):
    sq = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
    return torch.exp(-sq / (2.0 * bw))

def mmd_loss_batch(src_h, tgt_h, bws=(1., 10., 100.)):
    """mini-batch MMD（处理两个域节点的隐层嵌入）"""
    if src_h.shape[0] == 0 or tgt_h.shape[0] == 0:
        return torch.tensor(0.0, device=src_h.device)
    loss = torch.zeros(1, device=src_h.device)
    for bw in bws:
        Kss = _rbf(src_h, src_h, bw)
        Ktt = _rbf(tgt_h, tgt_h, bw)
        Kst = _rbf(src_h, tgt_h, bw)
        loss = loss + Kss.mean() + Ktt.mean() - 2 * Kst.mean()
    return loss / len(bws)

def entropy_loss_batch(log_probs):
    """对无标签节点的预测分布施加熵最小化约束"""
    if log_probs.shape[0] == 0:
        return torch.tensor(0.0, device=log_probs.device)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=1).mean()


# ══════════════════════════════════════════════════════════
# 完整推断（全图，用于评估和伪标签更新）
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def full_graph_inference(model_bare, data_cpu, device, cfg, n_classes):
    """
    使用 NeighborLoader 对全部节点做推断（避免全图放进一张卡）。
    model_bare: DDP 包装前的原始模型
    """
    model_bare.eval()
    loader = NeighborLoader(
        data_cpu,
        num_neighbors=cfg["num_neighbors"],
        batch_size=cfg["batch_size"] * 2,
        input_nodes=torch.arange(data_cpu.num_nodes),
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    log_probs_all = torch.zeros(data_cpu.num_nodes, n_classes)
    h_all         = torch.zeros(data_cpu.num_nodes, 256)

    for batch in loader:
        batch = batch.to(device)
        h_b, lp_b = model_bare.encode(batch.x, batch.edge_index)
        # NeighborLoader 只保证 batch.n_id[:batch.batch_size] 为当前种子节点
        seed_ids = batch.n_id[:batch.batch_size].cpu()
        log_probs_all[seed_ids] = lp_b[:batch.batch_size].cpu()
        h_all[seed_ids]          = h_b[:batch.batch_size].cpu()

    return log_probs_all, h_all


# ══════════════════════════════════════════════════════════
# DDP 训练主函数（每个 GPU 的 worker）
# ══════════════════════════════════════════════════════════

def train_worker(rank: int, world_size: int, cfg: dict,
                 data_cpu: Data, class_weights_cpu: torch.Tensor,
                 n_classes: int, cell_types: list):
    """
    rank: 当前进程对应的 GPU 编号 (0 ~ world_size-1)
    world_size: 总 GPU 数
    """
    # ── 初始化进程组 ──────────────────────────────────────
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_master = (rank == 0)     # 只有 rank-0 打印日志和保存模型

    if is_master:
        print(f"\n{'='*58}")
        print(f"  GAT DDP 训练  |  {world_size} GPUs  |  3080 x{world_size}")
        print(f"{'='*58}")
        print(f"  节点数: {data_cpu.num_nodes:,}  |  边数: {data_cpu.edge_index.shape[1]:,}")
        print(f"  细胞类型数: {n_classes}  |  batch_size/GPU: {cfg['batch_size']}")

    # ── 模型 ──────────────────────────────────────────────
    model = GATInductive(
        in_dim     = data_cpu.x.shape[1],
        hidden_dim = cfg["hidden_dim"],
        out_dim    = n_classes,
        heads      = cfg["heads"],
        dropout    = cfg["dropout"],
        proj_dim   = cfg["proj_dim"],
    ).to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    model_bare = model.module   # 底层模型（用于推断和保存）

    # ── 优化器 + 调度器 + AMP Scaler ─────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_factor"],
        patience=cfg["lr_patience"], min_lr=cfg["min_lr"]
    )
    scaler = GradScaler()   # 混合精度缩放器

    class_weights = class_weights_cpu.to(device)

    # ── NeighborLoader（每个 GPU 看不同的 batch）─────────
    # 训练 loader 只在有标签的 scRNA 节点上做种子
    train_idx = data_cpu.train_mask.nonzero(as_tuple=True)[0]
    val_idx   = data_cpu.val_mask.nonzero(as_tuple=True)[0]

    # 每个 rank 拿到不同的 shard（简单按 rank 切分种子）
    shard_size = (len(train_idx) + world_size - 1) // world_size
    local_train_idx = train_idx[rank * shard_size: (rank + 1) * shard_size]

    train_loader = NeighborLoader(
        data_cpu,
        num_neighbors=cfg["num_neighbors"],
        batch_size=cfg["batch_size"],
        input_nodes=local_train_idx,
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    # Xenium 节点 loader（用于采样 DA 损失）
    xen_idx = data_cpu.xenium_mask.nonzero(as_tuple=True)[0]
    xen_shard = xen_idx[rank * shard_size: (rank + 1) * shard_size]
    xen_loader = NeighborLoader(
        data_cpu,
        num_neighbors=cfg["num_neighbors"],
        batch_size=cfg["batch_size"],
        input_nodes=xen_shard,
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    # ── 训练循环 ──────────────────────────────────────────
    best_val_f1   = 0.0
    best_state    = None
    patience_cnt  = 0
    history       = []

    # 伪标签缓存（rank-0 广播给所有 rank）
    pseudo_labels_cpu = None
    pseudo_mask_cpu   = None

    xen_iter = iter(xen_loader)

    for epoch in range(1, cfg["n_epochs"] + 1):
        use_da = epoch > cfg["warmup_epochs"]

        # ── 更新伪标签（所有 rank 同步）──────────────────
        if use_da and (epoch - cfg["warmup_epochs"]) % cfg["pl_update_freq"] == 1:
            if is_master:
                lp_all, _ = full_graph_inference(model_bare, data_cpu, device, cfg, n_classes)
                xen_lp = lp_all[data_cpu.xenium_mask]
                conf, pred = xen_lp.exp().max(dim=1)
                mask = conf >= cfg["pl_threshold"]
                # 打包成 tensor 以便广播
                pl_tensor   = pred.contiguous()
                mask_tensor = mask.contiguous()
            else:
                pl_tensor   = torch.zeros(data_cpu.xenium_mask.sum(), dtype=torch.long)
                mask_tensor = torch.zeros(data_cpu.xenium_mask.sum(), dtype=torch.bool)

            dist.broadcast(pl_tensor,   src=0)
            dist.broadcast(mask_tensor, src=0)
            pseudo_labels_cpu = pl_tensor
            pseudo_mask_cpu   = mask_tensor

            if is_master:
                n_pl = mask_tensor.sum().item()
                print(f"  [Epoch {epoch:3d}] 伪标签更新: {n_pl:,} 个"
                      f" ({100*n_pl/data_cpu.xenium_mask.sum().item():.1f}%)")

        # ── 训练所有 batch ────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0

        xen_iter_local = iter(xen_loader)

        for batch in train_loader:
            batch = batch.to(device)
            # 对应的 Xenium batch（用于 DA 损失）
            try:
                xen_batch = next(xen_iter_local).to(device)
            except StopIteration:
                xen_iter_local = iter(xen_loader)
                xen_batch = next(xen_iter_local).to(device)

            optimizer.zero_grad()

            with autocast():   # 混合精度前向
                # scRNA batch
                h_src, lp_src = model_bare.encode(batch.x, batch.edge_index)
                # Xenium batch
                h_tgt, lp_tgt = model_bare.encode(xen_batch.x, xen_batch.edge_index)

                # 只取种子节点的预测（NeighborLoader 前 batch_size 个为种子）
                n_seed_src = batch.batch_size
                n_seed_tgt = xen_batch.batch_size

                # 有标签节点的 CE 损失
                has_label = batch.train_mask[:n_seed_src]
                if has_label.any():
                    loss_ce = F.nll_loss(
                        lp_src[:n_seed_src][has_label],
                        batch.y[:n_seed_src][has_label],
                        weight=class_weights,
                    )
                else:
                    loss_ce = torch.tensor(0.0, device=device)

                # MMD 损失
                loss_mmd = torch.tensor(0.0, device=device)
                if use_da:
                    loss_mmd = mmd_loss_batch(
                        h_src[:n_seed_src].detach(),   # 避免梯度流到 src
                        h_tgt[:n_seed_tgt],
                    )
                    # 熵正则化
                    loss_ent = entropy_loss_batch(lp_tgt[:n_seed_tgt])

                    # 伪标签损失
                    loss_pl = torch.tensor(0.0, device=device)
                    if pseudo_labels_cpu is not None and pseudo_mask_cpu is not None:
                        # 将 xen_batch 中的伪标签种子节点取出
                        xen_global_ids = xen_batch.n_id[:n_seed_tgt].cpu()
                        # 查找这些节点在 xenium_mask 内的位置
                        # data_cpu.xenium_mask_idx[i] 给出第i个xenium节点的全局id
                        # 用 xen_global_ids 对应到 pseudo 数组的下标
                        # 简化：从 xen_global_ids 对应到 n_flex 后的 xenium 偏移
                        n_flex = (~data_cpu.xenium_mask).sum().item()
                        xen_offset = xen_global_ids - n_flex
                        valid = (xen_offset >= 0) & (xen_offset < len(pseudo_labels_cpu))
                        if valid.any() and pseudo_mask_cpu[xen_offset[valid]].any():
                            pl_idx_local  = xen_offset[valid][pseudo_mask_cpu[xen_offset[valid]]]
                            # 对应的 tgt 位置（局部索引）
                            src_positions = valid.nonzero(as_tuple=True)[0]
                            src_positions = src_positions[pseudo_mask_cpu[xen_offset[valid]]]
                            pl_labels = pseudo_labels_cpu[pl_idx_local].to(device)
                            loss_pl = F.nll_loss(
                                lp_tgt[src_positions],
                                pl_labels,
                                weight=class_weights,
                            )
                else:
                    loss_ent = torch.tensor(0.0, device=device)
                    loss_pl  = torch.tensor(0.0, device=device)

                loss = (loss_ce
                        + cfg["lambda_mmd"] * loss_mmd
                        + cfg["lambda_ent"] * loss_ent
                        + cfg["lambda_pl"]  * loss_pl)

            # AMP 反向传播 + 梯度裁剪
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ── 评估（只在 rank-0 上做，每 5 epoch 一次）───────
        if is_master and (epoch % 5 == 0 or epoch <= 3):
            lp_all, _ = full_graph_inference(model_bare, data_cpu, device, cfg, n_classes)
            val_pred = lp_all[data_cpu.val_mask].argmax(dim=1).numpy()
            val_true = data_cpu.y[data_cpu.val_mask].numpy()
            val_f1   = f1_score(val_true, val_pred, average="macro", zero_division=0)
            val_loss = F.nll_loss(
                lp_all[data_cpu.val_mask].to(device),
                data_cpu.y[data_cpu.val_mask].to(device),
                weight=class_weights,
            ).item()

            scheduler.step(val_loss)

            log = {"epoch": epoch, "loss": avg_loss,
                   "val_f1_macro": val_f1, "val_loss": val_loss,
                   "lr": optimizer.param_groups[0]["lr"]}
            history.append(log)

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                best_state   = copy.deepcopy(model_bare.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 5

            print(f"  Ep {epoch:3d} | loss={avg_loss:.4f} "
                  f"| val_F1={val_f1:.4f} | lr={log['lr']:.2e}")

            if patience_cnt >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best F1={best_val_f1:.4f})")
                # 广播 early-stop 信号
                stop_flag = torch.tensor(1, device=device)
                dist.broadcast(stop_flag, src=0)
                break

        # 非 rank-0 的进程不做评估，只更新参数
        # 每 5 epoch 同步一次 stop 信号
        if epoch % 5 == 0:
            stop_flag = torch.tensor(0, device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break

    # ── rank-0 保存结果 ───────────────────────────────────
    if is_master and best_state is not None:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        model_path = os.path.join(cfg["output_dir"], "GAT_ddp_best.pt")
        torch.save(best_state, model_path)
        print(f"\n  模型已保存: {model_path}")
        print(f"  最优 Val F1-macro: {best_val_f1:.4f}")

        # 全图推断 Xenium
        model_bare.load_state_dict(best_state)
        lp_all, h_all = full_graph_inference(model_bare, data_cpu, device, cfg, n_classes)

        xen_lp    = lp_all[data_cpu.xenium_mask]
        xen_pred  = xen_lp.argmax(dim=1).numpy()
        xen_conf  = xen_lp.exp().max(dim=1).values.numpy()
        xen_labels = [cell_types[i] for i in xen_pred]

        result = {
            "model_name":    "GAT_DDP",
            "best_val_f1":   best_val_f1,
            "history":       history,
            "predictions":   xen_labels,
            "pred_indices":  xen_pred.tolist(),
            "confidence":    xen_conf.tolist(),
        }

        os.makedirs(os.path.dirname(cfg["result_json"]), exist_ok=True)
        with open(cfg["result_json"], "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  推断结果: {cfg['result_json']}")

    dist.destroy_process_group()
    return best_val_f1 if is_master else 0.0


# ══════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=None,
                        help="使用的 GPU 数量（默认: 全部可见 GPU）")
    args = parser.parse_args()

    # ── 加载图数据（在主进程加载，子进程共享内存）─────────
    if not os.path.exists(CFG["cache_file"]):
        print(f"[ERROR] 图缓存不存在: {CFG['cache_file']}")
        print("  请先运行 train.ipynb 的 Cell 3 生成图缓存。")
        sys.exit(1)

    print("正在加载图缓存...")
    ckpt = torch.load(CFG["cache_file"], weights_only=False)
    data_cpu         = ckpt["data"]
    class_weights_cpu = ckpt["class_weights"]
    split_info       = ckpt["split_info"]

    with open(CFG["scaler_file"], "rb") as f:
        fitted_scaler = pickle.load(f)

    # 确保 cell_types 可用（从 n_classes 反推或从元数据读取）
    n_classes  = int(data_cpu.y[data_cpu.train_mask].max().item()) + 1
    cell_types = [f"Type_{i}" for i in range(n_classes)]   # 占位符
    # 如果有实际名称文件，可在此读取：
    ct_file = "./data/cache/cell_types.json"
    if os.path.exists(ct_file):
        with open(ct_file) as f:
            cell_types = json.load(f)

    world_size = args.n_gpus or torch.cuda.device_count()
    if world_size == 0:
        print("[ERROR] 未检测到 GPU，请检查 CUDA 环境。")
        sys.exit(1)

    print(f"\n检测到 {torch.cuda.device_count()} 张 GPU，将使用 {world_size} 张")
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  {props.total_memory//1024**3} GB")

    # 预先诊断内存
    print(f"\n图信息: {data_cpu.num_nodes:,} 节点 / "
          f"{data_cpu.edge_index.shape[1]:,} 边 / "
          f"{data_cpu.x.shape[1]} 特征")

    # torchrun 通过环境变量传递 rank 信息，直接调用 train_worker
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_worker(local_rank, world_size, CFG,
                 data_cpu, class_weights_cpu, n_classes, cell_types)


if __name__ == "__main__":
    main()

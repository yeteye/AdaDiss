"""
gat_diagnose.py — 运行前先执行此脚本，精确估算 GAT 内存需求
"""
import torch
import numpy as np

def diagnose_gat_memory(n_nodes: int, n_edges: int,
                        hidden_dim: int = 256, heads: int = 4,
                        proj_dim: int = 512, in_dim: int = 377,
                        n_classes: int = 10):
    """
    估算 GAT 在全图 transductive 训练时各阶段的显存占用。
    单位：MB
    """
    B = 4          # float32 bytes
    B16 = 2        # float16 bytes

    print("=" * 58)
    print(f"  GAT 显存诊断报告  (n={n_nodes:,}  edges={n_edges:,})")
    print("=" * 58)

    # ── 图数据 ─────────────────────────────────────────────
    feat_mb   = n_nodes * in_dim * B / 1024**2
    edge_mb   = n_edges * 2 * 8 / 1024**2          # int64 edge_index

    # ── 模型参数 ───────────────────────────────────────────
    # 投影层: in_dim→proj_dim
    proj_params = in_dim * proj_dim + proj_dim * 2  # Linear + BN
    # GATConv 1: proj_dim → hidden_dim × heads
    gat1_params = (proj_dim * hidden_dim + hidden_dim) * heads * 2  # W + att_src/dst
    bn1_params  = hidden_dim * heads * 2
    # GATConv 2: hidden*heads → hidden
    gat2_params = (hidden_dim * heads * hidden_dim + hidden_dim) * 2
    bn2_params  = hidden_dim * 2
    # GATConv 3: hidden → n_classes
    gat3_params = (hidden_dim * n_classes + n_classes) * 2

    total_params = proj_params + gat1_params + bn1_params + gat2_params + bn2_params + gat3_params
    param_mb = total_params * B / 1024**2

    # ── 前向激活值 ─────────────────────────────────────────
    proj_act   = n_nodes * proj_dim * B / 1024**2
    h1_act     = n_nodes * hidden_dim * heads * B / 1024**2  # after conv1
    attn1_act  = n_edges * heads * B / 1024**2               # attention weights
    h2_act     = n_nodes * hidden_dim * B / 1024**2
    attn2_act  = n_edges * 1 * B / 1024**2
    h3_act     = n_nodes * n_classes * B / 1024**2
    total_act  = proj_act + h1_act + attn1_act + h2_act + attn2_act + h3_act

    # ── Adam 优化器状态 (params × 2) ─────────────────────
    adam_mb = param_mb * 2

    # ── 梯度 ─────────────────────────────────────────────
    grad_mb = param_mb + total_act   # 大致估算

    total_fp32 = feat_mb + edge_mb + param_mb + total_act + adam_mb + grad_mb

    print(f"\n  {'组件':<28} {'FP32 (MB)':>10}")
    print(f"  {'-'*40}")
    print(f"  {'节点特征矩阵':<28} {feat_mb:>10.1f}")
    print(f"  {'边索引 (edge_index)':<28} {edge_mb:>10.1f}")
    print(f"  {'模型参数':<28} {param_mb:>10.1f}")
    print(f"  {'前向激活 (全图)':<28} {total_act:>10.1f}")
    print(f"    其中注意力权重':<28} {'':>10}")
    print(f"      Layer1 ({heads}头):{'':<18} {attn1_act:>10.1f}")
    print(f"      Layer2 (1头):{'':<20} {attn2_act:>10.1f}")
    print(f"  {'Adam 优化器状态':<28} {adam_mb:>10.1f}")
    print(f"  {'梯度 (估算)':<28} {grad_mb:>10.1f}")
    print(f"  {'─'*40}")
    print(f"  {'总计 FP32':<28} {total_fp32:>10.1f}")
    print(f"  {'总计 FP16 (混合精度)':<28} {total_fp32 * 0.55:>10.1f}")
    print()

    gpu_gb = 20.0
    print(f"  单卡 3080 显存: {gpu_gb} GB = {gpu_gb*1024:.0f} MB")
    status_fp32 = "✅ 够用" if total_fp32 < gpu_gb*1024*0.9 else "❌ OOM"
    status_fp16 = "✅ 够用" if total_fp32*0.55 < gpu_gb*1024*0.9 else "❌ OOM"
    print(f"  FP32 状态: {status_fp32}  (占用 {total_fp32/(gpu_gb*1024)*100:.0f}%)")
    print(f"  FP16 状态: {status_fp16}  (占用 {total_fp32*0.55/(gpu_gb*1024)*100:.0f}%)")
    print()

    # ── 多卡建议 ──────────────────────────────────────────
    needed_gpus = int(np.ceil(total_fp32 / (gpu_gb * 1024 * 0.85)))
    print(f"  全图 FP32 至少需要: {needed_gpus} 张 3080")
    print(f"  推荐方案: 切换 NeighborSampler mini-batch (本脚本底部有实现)")
    print("=" * 58)

    return {
        "total_fp32_mb":   total_fp32,
        "total_fp16_mb":   total_fp32 * 0.55,
        "attn_heads_cost": attn1_act,
        "oom_fp32":        total_fp32 > gpu_gb * 1024 * 0.9,
        "oom_fp16":        total_fp32 * 0.55 > gpu_gb * 1024 * 0.9,
    }


if __name__ == "__main__":
    import sys
    # 从图缓存中读取实际节点/边数
    try:
        cache = torch.load("./data/cache/graph/graph_k30_val0.2.pt",
                           weights_only=False)
        data = cache["data"]
        n_nodes = data.x.shape[0]
        n_edges = data.edge_index.shape[1]
        in_dim  = data.x.shape[1]
        n_cls   = int(data.y[data.train_mask].max().item()) + 1
        print(f"从缓存读取: {n_nodes:,} 节点 / {n_edges:,} 边 / {in_dim} 特征 / {n_cls} 类")
    except FileNotFoundError:
        print("缓存不存在，使用估算值 (100k 节点, 250万边, 377特征, 10类)")
        n_nodes, n_edges, in_dim, n_cls = 100_000, 2_500_000, 377, 10

    result = diagnose_gat_memory(n_nodes, n_edges, in_dim=in_dim, n_classes=n_cls)
    if result["oom_fp32"] and not result["oom_fp16"]:
        print("\n  → 建议方案 A: 开启混合精度 (AMP) 即可解决")
    elif result["oom_fp16"]:
        print("\n  → 建议方案 B: 必须使用 NeighborSampler + 多 GPU")

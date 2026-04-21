"""
eval.py — 论文级评估 + 图表生成

图表清单（毕业论文必备）
-----------------------
Fig 1  : scRNA 参考集 UMAP（细胞类型着色）
Fig 2  : GNN 隐层嵌入 UMAP（细胞类型 + 数据域着色，两子图）
Fig 3  : 训练动态曲线（Loss / Val F1 随 epoch 变化）
Fig 4  : 方法定量对比（Acc / F1-macro / F1-weighted 柱状图）
Fig 5  : 每个方法的归一化混淆矩阵
Fig 6  : 空间预测图（4-panel：GCN / SAGE / GAT / TopACT）
Fig 7  : 细胞类型比例对比（scRNA vs 各模型预测，堆叠条形图）
Fig 8  : 逐类别 F1 热力图
Fig 9  : 预测置信度分布（小提琴图，按方法分组）
Fig 10 : Moran's I 空间自相关柱状图（按细胞类型）

重要原则
--------
- Cell 6 风险（P0-④）：Seurat 标签只用于对比分析，不参与调参或 early stopping
- 所有 F1 / Accuracy 基于 scRNA 验证集（真实 ground truth）
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    cohen_kappa_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# ── 全局绘图风格 ──────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.1,
})

# 颜色 palette（适合色盲友好，Nature Methods 常用）
CB_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]


# ══════════════════════════════════════════════════════════
# 0. 定量评估
# ══════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_classes: int | None = None,
) -> dict:
    """
    计算全套论文指标：Accuracy, F1 (macro/weighted/per-class),
    Cohen's Kappa, AUC (macro OVR, 可选).
    """
    n_classes = n_classes or len(np.unique(y_true))
    metrics   = {
        "accuracy":     accuracy_score(y_true, y_pred),
        "f1_macro":     f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None,       zero_division=0),
        "kappa":        cohen_kappa_score(y_true, y_pred),
    }
    if y_proba is not None and n_classes > 1:
        try:
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            metrics["auc_macro"] = roc_auc_score(
                y_bin, y_proba, average="macro", multi_class="ovr"
            )
        except Exception:
            metrics["auc_macro"] = float("nan")
    return metrics


def compare_all_methods(
    gnn_results: dict,
    topact_pred_idx: np.ndarray,
    topact_proba: np.ndarray,
    val_labels: np.ndarray,       # scRNA 验证集真实标签
    val_pred_per_method: dict,    # {method_name: val_pred_array}
    cell_types: list[str],
) -> pd.DataFrame:
    """
    汇总所有方法的定量指标，输出可直接放入论文表格的 DataFrame。

    注意：此处评估对象是 scRNA 验证集，不是 Xenium（避免 P0-④ 风险）。
    """
    rows = []
    for name, val_pred in val_pred_per_method.items():
        proba = None
        if name in gnn_results:
            # 从 history 中取最后验证集概率（可选）
            pass
        m = compute_metrics(val_labels, val_pred, proba, len(cell_types))
        rows.append({
            "Method":      name,
            "Accuracy":    f"{m['accuracy']:.4f}",
            "F1 Macro":    f"{m['f1_macro']:.4f}",
            "F1 Weighted": f"{m['f1_weighted']:.4f}",
            "Kappa":       f"{m['kappa']:.4f}",
        })
    df = pd.DataFrame(rows).set_index("Method")
    print("\n=== 定量对比（scRNA 验证集，论文 Table 1）===")
    print(df.to_string())
    return df


# ══════════════════════════════════════════════════════════
# 1. Fig 1 — scRNA 参考集 UMAP
# ══════════════════════════════════════════════════════════

def plot_scrna_umap(
    scrna_expr: np.ndarray,
    labels: np.ndarray,
    cell_types: list[str],
    output_path: Path,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
):
    """Fig 1：scRNA 参考集 UMAP（按细胞类型着色）。"""
    from umap import UMAP

    print("Computing UMAP for scRNA reference...")
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(scrna_expr)

    n_types = len(cell_types)
    cmap    = plt.cm.get_cmap("tab20", n_types)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, ct in enumerate(cell_types):
        mask = labels == i
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[cmap(i)], s=4, alpha=0.7, label=ct, rasterized=True
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("scRNA reference — cell type annotation")
    leg = ax.legend(
        markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=8, frameon=False
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")
    return embedding    # 返回供 Fig 2 复用


# ══════════════════════════════════════════════════════════
# 2. Fig 2 — GNN 隐层嵌入 UMAP（域 + 细胞类型）
# ══════════════════════════════════════════════════════════

def plot_embedding_umap(
    scrna_embeddings: np.ndarray,   # (n_scrna, h_dim)
    xenium_embeddings: np.ndarray,  # (n_xenium, h_dim)
    scrna_labels: np.ndarray,
    xenium_pred_idx: np.ndarray,
    cell_types: list[str],
    model_name: str,
    output_path: Path,
):
    """
    Fig 2：GNN 隐层嵌入 UMAP（两子图）。
    左图：按数据域着色（评估域适应效果）
    右图：按预测细胞类型着色（验证类型可分性）
    """
    from umap import UMAP

    all_emb   = np.vstack([scrna_embeddings, xenium_embeddings])
    n_scrna   = len(scrna_embeddings)
    n_xenium  = len(xenium_embeddings)
    n_types   = len(cell_types)

    print(f"  Computing embedding UMAP for {model_name}...")
    reducer   = UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    emb_2d    = reducer.fit_transform(all_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：域着色
    ax = axes[0]
    ax.scatter(emb_2d[:n_scrna, 0],  emb_2d[:n_scrna, 1],
               c=CB_PALETTE[0], s=2, alpha=0.5, label="scRNA (ref)", rasterized=True)
    ax.scatter(emb_2d[n_scrna:, 0],  emb_2d[n_scrna:, 1],
               c=CB_PALETTE[1], s=2, alpha=0.5, label="Xenium", rasterized=True)
    ax.set_title(f"{model_name} — domain alignment")
    ax.legend(markerscale=4, frameon=False)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # 右：细胞类型着色
    ax = axes[1]
    cmap = plt.cm.get_cmap("tab20", n_types)
    combined_labels = np.concatenate([scrna_labels, xenium_pred_idx])
    for i, ct in enumerate(cell_types):
        mask = combined_labels == i
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap(i)], s=2, alpha=0.5, label=ct, rasterized=True)
    ax.set_title(f"{model_name} — cell type clusters")
    ax.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, frameon=False)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    fig.suptitle(f"GNN embedding UMAP ({model_name})", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 3. Fig 3 — 训练动态曲线
# ══════════════════════════════════════════════════════════

def plot_training_curves(
    all_histories: dict,   # {model_name: [epoch_log_dict, ...]}
    output_path: Path,
):
    """
    Fig 3：各模型 Loss 分量 + Val F1 随 epoch 变化曲线。
    """
    n_models = len(all_histories)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 9))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, (name, history) in enumerate(all_histories.items()):
        epochs      = [h["epoch"]        for h in history]
        loss_ce     = [h["loss_ce"]      for h in history]
        loss_mmd    = [h["loss_mmd"]     for h in history]
        loss_ent    = [h.get("loss_ent", 0) for h in history]
        loss_pl     = [h.get("loss_pl",  0) for h in history]
        val_f1      = [h["val_f1_macro"] for h in history]
        train_f1    = [h["train_f1_macro"] for h in history]

        # 上：Loss 分量
        ax = axes[0, col]
        ax.plot(epochs, loss_ce,  label="CE loss",  color=CB_PALETTE[0])
        ax.plot(epochs, loss_mmd, label="MMD loss", color=CB_PALETTE[1], linestyle="--")
        ax.plot(epochs, loss_ent, label="Ent loss", color=CB_PALETTE[2], linestyle=":")
        ax.plot(epochs, loss_pl,  label="PL loss",  color=CB_PALETTE[3], linestyle="-.")
        ax.set_title(f"{name} — training losses")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8, frameon=False)

        # 下：F1 曲线
        ax = axes[1, col]
        ax.plot(epochs, train_f1, label="Train F1 (macro)", color=CB_PALETTE[0])
        ax.plot(epochs, val_f1,   label="Val F1 (macro)",   color=CB_PALETTE[1],
                linestyle="--")
        best_ep = epochs[np.argmax(val_f1)]
        ax.axvline(best_ep, color="gray", linestyle=":", alpha=0.6,
                   label=f"Best epoch ({best_ep})")
        ax.set_ylim(0, 1)
        ax.set_title(f"{name} — F1 curves")
        ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Macro")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Training dynamics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 4. Fig 4 — 方法定量对比柱状图
# ══════════════════════════════════════════════════════════

def plot_method_comparison(
    metrics_dict: dict,   # {method: {'accuracy':..., 'f1_macro':..., 'f1_weighted':...}}
    output_path: Path,
):
    """Fig 4：所有方法的 Accuracy / F1-macro / F1-weighted 对比。"""
    methods  = list(metrics_dict.keys())
    acc      = [metrics_dict[m]["accuracy"]    for m in methods]
    f1_mac   = [metrics_dict[m]["f1_macro"]    for m in methods]
    f1_wei   = [metrics_dict[m]["f1_weighted"] for m in methods]
    kappa    = [metrics_dict[m].get("kappa", 0) for m in methods]

    x     = np.arange(len(methods))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - 1.5*width, acc,    width, label="Accuracy",    color=CB_PALETTE[0])
    b2 = ax.bar(x - 0.5*width, f1_mac, width, label="F1 Macro",    color=CB_PALETTE[1])
    b3 = ax.bar(x + 0.5*width, f1_wei, width, label="F1 Weighted", color=CB_PALETTE[2])
    b4 = ax.bar(x + 1.5*width, kappa,  width, label="Cohen's κ",   color=CB_PALETTE[3])

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_title("Method comparison — scRNA validation set")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 5. Fig 5 — 归一化混淆矩阵
# ══════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cell_types: list[str],
    title: str,
    output_path: Path,
):
    """Fig 5：归一化混淆矩阵（行归一化 → 召回率视角）。"""
    cm      = confusion_matrix(y_true, y_pred, labels=np.arange(len(cell_types)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(10, len(cell_types) * 0.7),
                                    max(8, len(cell_types) * 0.65)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=cell_types, yticklabels=cell_types,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Recall (row-normalized)"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 6. Fig 6 — 空间预测图
# ══════════════════════════════════════════════════════════

def plot_spatial_predictions(
    xenium_coords: np.ndarray,    # (n_xenium, 2)
    all_pred_indices: dict,       # {method_name: (n_xenium,) int array}
    cell_types: list[str],
    output_path: Path,
    point_size: float = 1.5,
    alpha: float = 0.7,
):
    """Fig 6：四方法空间预测图（2×2 子图）。"""
    methods  = list(all_pred_indices.keys())
    n_methods = len(methods)
    n_cols   = min(n_methods, 2)
    n_rows   = (n_methods + n_cols - 1) // n_cols

    cmap    = plt.cm.get_cmap("tab20", len(cell_types))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(9 * n_cols, 8 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, method in enumerate(methods):
        r, c = divmod(idx, n_cols)
        ax   = axes[r, c]
        pred = all_pred_indices[method]

        sc = ax.scatter(
            xenium_coords[:, 0], xenium_coords[:, 1],
            c=pred, cmap="tab20", vmin=0, vmax=len(cell_types) - 1,
            s=point_size, alpha=alpha, rasterized=True
        )
        ax.set_title(method, fontsize=13)
        ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)")
        ax.set_aspect("equal")

    # 共享图例
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(i), markersize=8, label=ct)
        for i, ct in enumerate(cell_types)
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(cell_types), 4),
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Spatial cell type predictions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 7. Fig 7 — 细胞类型比例对比（堆叠条形图）
# ══════════════════════════════════════════════════════════

def plot_cell_type_proportions(
    scrna_labels: np.ndarray,
    all_pred_indices: dict,    # {method_name: (n_xenium,) int array}
    cell_types: list[str],
    output_path: Path,
):
    """
    Fig 7：scRNA 参考集 vs 各模型 Xenium 预测的细胞类型比例。
    堆叠条形图，论文中用于验证细胞类型比例的合理性。
    """
    n_types = len(cell_types)

    def proportions(labels):
        counts = np.bincount(labels, minlength=n_types)
        return counts / counts.sum()

    sources    = {"scRNA (ref)": proportions(scrna_labels)}
    sources.update({m: proportions(p) for m, p in all_pred_indices.items()})

    df = pd.DataFrame(sources, index=cell_types).T

    cmap = plt.cm.get_cmap("tab20", n_types)
    colors = [cmap(i) for i in range(n_types)]

    fig, ax = plt.subplots(figsize=(max(8, len(sources) * 1.5), 6))
    df.plot(kind="bar", stacked=True, color=colors, ax=ax,
            width=0.65, legend=False)

    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_title("Cell type proportion comparison")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[i], label=ct)
        for i, ct in enumerate(cell_types)
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 8. Fig 8 — 逐类别 F1 热力图
# ══════════════════════════════════════════════════════════

def plot_per_class_f1_heatmap(
    y_true: np.ndarray,
    all_val_preds: dict,   # {method_name: y_pred}
    cell_types: list[str],
    output_path: Path,
):
    """
    Fig 8：逐细胞类型 F1 热力图（方法 × 细胞类型）。
    用于展示各方法在稀有细胞类型上的性能差异。
    """
    rows = {}
    for method, y_pred in all_val_preds.items():
        f1_per = f1_score(y_true, y_pred, average=None,
                          labels=np.arange(len(cell_types)), zero_division=0)
        rows[method] = f1_per

    df = pd.DataFrame(rows, index=cell_types).T

    fig, ax = plt.subplots(figsize=(max(12, len(cell_types) * 0.85), 4))
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "F1 Score"},
        ax=ax,
    )
    ax.set_title("Per-class F1 score — scRNA validation set")
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Method")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 9. Fig 9 — 预测置信度分布
# ══════════════════════════════════════════════════════════

def plot_confidence_distribution(
    all_confidences: dict,   # {method_name: (n_xenium,) confidence array}
    output_path: Path,
):
    """
    Fig 9：各方法对 Xenium 预测的最大 softmax 置信度分布。
    置信度越高且分布越集中，说明域适应越有效。
    """
    data_list = []
    for method, conf in all_confidences.items():
        for c in conf:
            data_list.append({"Method": method, "Confidence": c})
    df = pd.DataFrame(data_list)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 小提琴图
    ax = axes[0]
    sns.violinplot(data=df, x="Method", y="Confidence",
                   palette=CB_PALETTE[:len(all_confidences)],
                   inner="box", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Prediction confidence distribution (Xenium)")
    ax.set_xlabel("Method"); ax.set_ylabel("Max softmax probability")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # 累积分布（CDF）
    ax = axes[1]
    for i, (method, conf) in enumerate(all_confidences.items()):
        sorted_c = np.sort(conf)
        cdf      = np.arange(1, len(sorted_c) + 1) / len(sorted_c)
        ax.plot(sorted_c, cdf, label=method, color=CB_PALETTE[i], linewidth=1.8)
    ax.axvline(0.9, color="gray", linestyle="--", alpha=0.7, label="threshold=0.9")
    ax.set_xlabel("Confidence"); ax.set_ylabel("CDF")
    ax.set_title("CDF of prediction confidence")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 10. Fig 10 — Moran's I 空间自相关
# ══════════════════════════════════════════════════════════

def plot_morans_i(
    morans_results: dict,   # {method_name: {cell_type: {'I':..., 'p_value':...}}}
    cell_types: list[str],
    output_path: Path,
):
    """
    Fig 10：各方法 Moran's I 逐细胞类型比较。
    用于验证空间预测的生物学合理性。
    """
    methods = list(morans_results.keys())
    n_types = len(cell_types)
    x       = np.arange(n_types)
    width   = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(12, n_types * 0.8), 5))

    for i, method in enumerate(methods):
        mi_vals = [morans_results[method].get(ct, {}).get("I", 0.0)
                   for ct in cell_types]
        p_vals  = [morans_results[method].get(ct, {}).get("p_value", 1.0)
                   for ct in cell_types]
        offset  = (i - len(methods) / 2 + 0.5) * width
        bars    = ax.bar(x + offset, mi_vals, width, label=method,
                         color=CB_PALETTE[i], alpha=0.85)

        # 显著性标记（* p<0.05, ** p<0.01）
        for bar, p in zip(bars, p_vals):
            mark = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            if mark:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005, mark,
                        ha="center", va="bottom", fontsize=8, color="black")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Moran's I")
    ax.set_title("Spatial autocorrelation (Moran's I) by cell type\n"
                 "(* p<0.05, ** p<0.01)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path}")


# ══════════════════════════════════════════════════════════
# 主入口：生成所有论文图表
# ══════════════════════════════════════════════════════════

def generate_all_thesis_figures(
    gnn_results: dict,
    topact_results: dict,
    scrna_expr_norm: np.ndarray,
    scrna_labels: np.ndarray,
    xenium_coords: np.ndarray,
    cell_types: list[str],
    val_labels: np.ndarray,
    val_pred_per_method: dict,
    morans_results: dict | None = None,
    output_dir: str = "./plots",
):
    """
    一键生成所有论文图表。

    Parameters
    ----------
    gnn_results         : run_experiment 的输出字典（可多个 model）
    topact_results      : TopACT.predict 的输出
    scrna_expr_norm     : 归一化后的 scRNA 特征（用于 UMAP）
    scrna_labels        : scRNA 整型标签
    xenium_coords       : Xenium 空间坐标 (n, 2)
    val_labels          : scRNA 验证集真实标签
    val_pred_per_method : {method_name: val_set 预测数组}
    morans_results      : {method_name: per_class Moran's I}
    output_dir          : 图表保存目录
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*55)
    print("  生成论文图表...")
    print("="*55)

    # 收集所有方法的预测
    all_pred_indices  = {}
    all_confidences   = {}
    all_histories     = {}
    all_gnn_emb       = {}

    for name, res in gnn_results.items():
        all_pred_indices[name] = res["pred_indices"]
        all_confidences[name]  = res["confidence"]
        all_histories[name]    = res["history"]
        all_gnn_emb[name]      = res["embeddings"]

    all_pred_indices["TopACT"] = topact_results["pred_indices"]
    all_confidences["TopACT"]  = topact_results["confidence"]

    # 汇总 metrics（基于 scRNA 验证集）
    metrics_dict = {}
    for method, val_pred in val_pred_per_method.items():
        m = compute_metrics(val_labels, val_pred, n_classes=len(cell_types))
        metrics_dict[method] = m

    # ── Fig 1 ────────────────────────────────────────────
    print("\n[Fig 1] scRNA UMAP...")
    try:
        plot_scrna_umap(
            scrna_expr_norm, scrna_labels, cell_types,
            out / "fig1_scrna_umap.pdf"
        )
    except ImportError:
        print("  ⚠ umap-learn not installed, skipping Fig 1")

    # ── Fig 2 ────────────────────────────────────────────
    print("\n[Fig 2] GNN embedding UMAP...")
    for name, emb in all_gnn_emb.items():
        n_scrna = len(scrna_labels)
        try:
            plot_embedding_umap(
                emb[:n_scrna], emb[n_scrna:],
                scrna_labels, all_pred_indices[name],
                cell_types, name,
                out / f"fig2_embedding_umap_{name}.pdf"
            )
        except ImportError:
            print("  ⚠ umap-learn not installed, skipping Fig 2")
            break

    # ── Fig 3 ────────────────────────────────────────────
    print("\n[Fig 3] Training curves...")
    plot_training_curves(all_histories, out / "fig3_training_curves.pdf")

    # ── Fig 4 ────────────────────────────────────────────
    print("\n[Fig 4] Method comparison...")
    plot_method_comparison(metrics_dict, out / "fig4_method_comparison.pdf")

    # ── Fig 5 ────────────────────────────────────────────
    print("\n[Fig 5] Confusion matrices...")
    for method, val_pred in val_pred_per_method.items():
        plot_confusion_matrix(
            val_labels, val_pred, cell_types,
            f"Confusion matrix — {method} (scRNA val set)",
            out / f"fig5_confusion_{method}.pdf"
        )

    # ── Fig 6 ────────────────────────────────────────────
    print("\n[Fig 6] Spatial prediction maps...")
    if xenium_coords is not None:
        plot_spatial_predictions(
            xenium_coords, all_pred_indices, cell_types,
            out / "fig6_spatial_predictions.pdf"
        )

    # ── Fig 7 ────────────────────────────────────────────
    print("\n[Fig 7] Cell type proportions...")
    plot_cell_type_proportions(
        scrna_labels, all_pred_indices, cell_types,
        out / "fig7_cell_type_proportions.pdf"
    )

    # ── Fig 8 ────────────────────────────────────────────
    print("\n[Fig 8] Per-class F1 heatmap...")
    plot_per_class_f1_heatmap(
        val_labels, val_pred_per_method, cell_types,
        out / "fig8_perclass_f1.pdf"
    )

    # ── Fig 9 ────────────────────────────────────────────
    print("\n[Fig 9] Confidence distribution...")
    plot_confidence_distribution(all_confidences, out / "fig9_confidence.pdf")

    # ── Fig 10 ───────────────────────────────────────────
    if morans_results:
        print("\n[Fig 10] Moran's I...")
        plot_morans_i(morans_results, cell_types, out / "fig10_morans_i.pdf")

    print(f"\n✅ 所有图表已保存到 {out.resolve()}/")
    print("   格式：PDF（300 dpi），可直接插入 LaTeX / Word 论文")

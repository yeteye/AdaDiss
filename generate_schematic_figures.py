"""
论文示意图生成脚本 — 图 3-2 / 图 4-1 / 图 4-2
===================================================

适用范围：本脚本独立运行（不依赖 notebook 中间变量），
直接生成三张论文示意图到 figures/ 目录。

依赖：matplotlib >= 3.5，numpy

运行方式：
    python generate_schematic_figures.py
或在 notebook 中分别执行三个 main 函数。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# 中文显示（如系统无 SimHei，可改 Microsoft YaHei / Noto Sans CJK）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                    'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'figures/'
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 图 3-2  转录本聚合示意图
# ============================================================
def make_fig_3_2(out_path: str = None):
    """
    左图：原始转录本点云（彩色散点，按基因类型上色，模拟 4 种 marker）；
    右图：15 μm bin 聚合后的 spot 网格，颜色映射每个 spot 的总转录本数。
    """
    if out_path is None:
        out_path = os.path.join(OUT_DIR, 'fig_3_2_aggregation_schematic.png')

    rng = np.random.RandomState(42)
    region_size = 60.0   # 显示区域 60 × 60 μm
    bin_size    = 15.0   # bin 尺寸 15 μm

    # 模拟 4 种基因 marker 的转录本，每种集中分布在不同区域
    centers = [(15, 15), (45, 15), (20, 45), (45, 40)]
    n_per_gene = [80, 60, 90, 70]
    colors_genes = ['#E64B35', '#4DBBD5', '#00A087', '#F39B7F']
    gene_labels  = ['Gene A', 'Gene B', 'Gene C', 'Gene D']

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=200)

    # ── 左图：原始点云 ────────────────────────────────────────
    ax = axes[0]
    for (cx, cy), n, c, lab in zip(centers, n_per_gene, colors_genes, gene_labels):
        sx = rng.normal(cx, 6.0, n)
        sy = rng.normal(cy, 6.0, n)
        # 截断到显示区域内
        keep = (sx >= 0) & (sx <= region_size) & (sy >= 0) & (sy <= region_size)
        ax.scatter(sx[keep], sy[keep], c=c, s=18, alpha=0.85,
                   edgecolors='white', linewidths=0.3, label=lab)

    ax.set_xlim(0, region_size)
    ax.set_ylim(0, region_size)
    ax.set_aspect('equal')
    ax.set_title('(a) 原始 Xenium 转录本点云', fontsize=13)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(False)

    # ── 右图：15 μm bin 聚合后 ───────────────────────────────
    ax = axes[1]
    n_bins = int(region_size / bin_size)
    counts = np.zeros((n_bins, n_bins))

    # 重新生成相同点云并落桶
    all_x, all_y = [], []
    for (cx, cy), n in zip(centers, n_per_gene):
        sx = rng.normal(cx, 6.0, n)
        sy = rng.normal(cy, 6.0, n)
        keep = (sx >= 0) & (sx < region_size) & (sy >= 0) & (sy < region_size)
        all_x.extend(sx[keep]); all_y.extend(sy[keep])
    rng = np.random.RandomState(42)  # reset for reproducibility
    for (cx, cy), n in zip(centers, n_per_gene):
        sx = rng.normal(cx, 6.0, n)
        sy = rng.normal(cy, 6.0, n)
        for x, y in zip(sx, sy):
            if 0 <= x < region_size and 0 <= y < region_size:
                bx, by = int(x // bin_size), int(y // bin_size)
                counts[by, bx] += 1

    im = ax.imshow(counts, origin='lower',
                   extent=[0, region_size, 0, region_size],
                   cmap='YlOrRd', vmin=0, vmax=counts.max())
    # 网格线
    for i in range(n_bins + 1):
        ax.axhline(i * bin_size, color='gray', linewidth=0.5, alpha=0.6)
        ax.axvline(i * bin_size, color='gray', linewidth=0.5, alpha=0.6)
    # 在每个 bin 中央标注转录本数
    for i in range(n_bins):
        for j in range(n_bins):
            v = int(counts[i, j])
            if v > 0:
                ax.text(j * bin_size + bin_size/2, i * bin_size + bin_size/2,
                        str(v), ha='center', va='center',
                        fontsize=11,
                        color='white' if v > counts.max()*0.55 else 'black',
                        fontweight='bold')

    ax.set_xlim(0, region_size); ax.set_ylim(0, region_size)
    ax.set_aspect('equal')
    ax.set_title(f'(b) {int(bin_size)} μm 方格聚合后的 spot 网格', fontsize=13)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label('每 spot 转录本数')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'✅ 图 3-2 已保存: {out_path}')


# ============================================================
# 图 4-1  方法总体框架图
# ============================================================
def make_fig_4_1(out_path: str = None):
    """
    水平四阶段流程图：数据预处理 → 联合图构建 → GNN 半监督训练 → 推断输出
    每个阶段一个圆角矩形，下方列出关键步骤；阶段之间用箭头连接。
    """
    if out_path is None:
        out_path = os.path.join(OUT_DIR, 'fig_4_1_framework.png')

    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 4 个阶段
    stages = [
        {
            'title': '① 数据预处理',
            'lines': ['Xenium 转录本 Q≥20 过滤',
                      'scRNA QC + 注释匹配',
                      '4748 共同基因对齐',
                      'log1p + PCA(50) 投影'],
            'color': '#FFE5B4',
            'border': '#E69138',
        },
        {
            'title': '② 联合图构建',
            'lines': ['scRNA 内 / spot 内特征 kNN',
                      '跨模态 mutual NN 边',
                      'spot 空间 kNN 边',
                      '类权重 [1.0, 5.0]'],
            'color': '#D9EAD3',
            'border': '#6AA84F',
        },
        {
            'title': '③ GNN 半监督训练',
            'lines': ['GCN / GraphSAGE / GAT',
                      'CE + MMD + Ent + PL',
                      '30 epoch warmup',
                      'Adam, AMP, 早停'],
            'color': '#CFE2F3',
            'border': '#3D85C6',
        },
        {
            'title': '④ 推断输出',
            'lines': ['spot 概率矩阵',
                      'argmax → 类型标签',
                      'Moran\'s I 评估',
                      '与 Seurat 对比'],
            'color': '#F4CCCC',
            'border': '#CC0000',
        },
    ]

    box_w, box_h = 3.4, 3.6
    gap = 0.4
    x0 = 0.5
    y_box = 1.2

    # 绘制阶段框
    centers_x = []
    for i, s in enumerate(stages):
        x = x0 + i * (box_w + gap)
        centers_x.append(x + box_w / 2)
        # 圆角矩形
        box = FancyBboxPatch((x, y_box), box_w, box_h,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             linewidth=2.0,
                             edgecolor=s['border'],
                             facecolor=s['color'],
                             alpha=0.95)
        ax.add_patch(box)
        # 标题
        ax.text(x + box_w / 2, y_box + box_h - 0.5,
                s['title'], ha='center', va='center',
                fontsize=14, fontweight='bold',
                color=s['border'])
        # 内容行
        for j, line in enumerate(s['lines']):
            ax.text(x + box_w / 2, y_box + box_h - 1.2 - j * 0.55,
                    '• ' + line, ha='center', va='center',
                    fontsize=10.5, color='#333333')

    # 阶段之间的箭头
    arrow_y = y_box + box_h / 2
    for i in range(len(stages) - 1):
        x_start = x0 + i * (box_w + gap) + box_w + 0.05
        x_end   = x0 + (i + 1) * (box_w + gap) - 0.05
        arrow = FancyArrowPatch((x_start, arrow_y), (x_end, arrow_y),
                                arrowstyle='->,head_width=0.35,head_length=0.4',
                                linewidth=2.5, color='#666666',
                                mutation_scale=20)
        ax.add_patch(arrow)

    # 整体标题
    ax.text(8, 5.4, '本文方法总体流程',
            ha='center', va='center', fontsize=15, fontweight='bold')

    # 底部数据集信息
    ax.text(8, 0.5,
            '数据集：FFPE Human Ovarian Cancer (Xenium Prime, 9065 panel) + '
            '17k Ovarian Cancer scFFPE (Flex)',
            ha='center', va='center', fontsize=10, style='italic',
            color='#555555')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'✅ 图 4-1 已保存: {out_path}')


# ============================================================
# 图 4-2  跨模态联合图构建示意图
# ============================================================
def make_fig_4_2(out_path: str = None):
    """
    左半：scRNA 细胞节点（蓝），有标签；右半：Xenium spot 节点（橙），无标签；
    三类边：scRNA 内 kNN（蓝色虚线）、spot 内特征 kNN（橙色虚线）、
            spot 空间 kNN（红色实线，仅在 spot 间）、跨模态 mutual NN（绿色实线）。
    """
    if out_path is None:
        out_path = os.path.join(OUT_DIR, 'fig_4_2_cross_modal_graph.png')

    rng = np.random.RandomState(7)

    # 节点位置
    n_sc = 8
    n_sp = 14

    # scRNA 细胞分布在左半圆区域
    sc_angles = rng.uniform(np.pi*0.5, np.pi*1.5, n_sc)
    sc_radius = rng.uniform(0.6, 2.4, n_sc)
    sc_x = sc_radius * np.cos(sc_angles) - 1.5
    sc_y = sc_radius * np.sin(sc_angles) + 4
    sc_pos = list(zip(sc_x, sc_y))

    # Xenium spot 分布在右半区域，更密集
    sp_x = rng.uniform(2.5, 6.5, n_sp)
    sp_y = rng.uniform(2.0, 6.0, n_sp)
    sp_pos = list(zip(sp_x, sp_y))

    fig, ax = plt.subplots(figsize=(13, 8), dpi=200)
    ax.set_xlim(-5, 8)
    ax.set_ylim(0, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')

    color_sc      = '#1F77B4'      # scRNA 节点 — 蓝
    color_sp      = '#FF7F0E'      # spot 节点 — 橙
    color_sc_knn  = '#1F77B4'      # scRNA 内 kNN — 蓝
    color_sp_feat = '#FF7F0E'      # spot 内特征 kNN — 橙
    color_sp_spat = '#D62728'      # spot 空间 kNN — 红
    color_cross   = '#2CA02C'      # 跨模态 mutual NN — 绿

    # ── 边：scRNA 内 kNN（每节点连最近 2 个）──────────────────
    for i, (x1, y1) in enumerate(sc_pos):
        d = [(j, np.hypot(x1 - x2, y1 - y2))
             for j, (x2, y2) in enumerate(sc_pos) if j != i]
        d.sort(key=lambda z: z[1])
        for j, _ in d[:2]:
            x2, y2 = sc_pos[j]
            ax.plot([x1, x2], [y1, y2], color=color_sc_knn,
                    linestyle=':', linewidth=1.0, alpha=0.55, zorder=1)

    # ── 边：spot 内特征 kNN（每节点连最近 2 个）──────────────
    for i, (x1, y1) in enumerate(sp_pos):
        d = [(j, np.hypot(x1 - x2, y1 - y2))
             for j, (x2, y2) in enumerate(sp_pos) if j != i]
        d.sort(key=lambda z: z[1])
        for j, _ in d[:2]:
            x2, y2 = sp_pos[j]
            ax.plot([x1, x2], [y1, y2], color=color_sp_feat,
                    linestyle=':', linewidth=1.0, alpha=0.45, zorder=1)

    # ── 边：spot 空间 kNN（仅最近 1 个的子集，避免太密）──────
    plotted = set()
    for i, (x1, y1) in enumerate(sp_pos):
        d = [(j, np.hypot(x1 - x2, y1 - y2))
             for j, (x2, y2) in enumerate(sp_pos) if j != i]
        d.sort(key=lambda z: z[1])
        j = d[0][0]
        key = tuple(sorted([i, j]))
        if key in plotted:
            continue
        plotted.add(key)
        x2, y2 = sp_pos[j]
        ax.plot([x1, x2], [y1, y2], color=color_sp_spat,
                linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)

    # ── 边：跨模态 mutual NN（精选少量代表性边）────────────
    cross_pairs = [(2, 3), (4, 7), (5, 11), (6, 9), (1, 0)]
    for sc_i, sp_i in cross_pairs:
        if sc_i >= n_sc or sp_i >= n_sp:
            continue
        x1, y1 = sc_pos[sc_i]
        x2, y2 = sp_pos[sp_i]
        ax.plot([x1, x2], [y1, y2], color=color_cross,
                linestyle='-', linewidth=2.0, alpha=0.8, zorder=2)

    # ── 节点 ─────────────────────────────────────────────────
    for x, y in sc_pos:
        circle = Circle((x, y), 0.22, facecolor=color_sc,
                        edgecolor='white', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
    for x, y in sp_pos:
        circle = Circle((x, y), 0.18, facecolor=color_sp,
                        edgecolor='white', linewidth=1.2, zorder=3)
        ax.add_patch(circle)

    # ── 区域标签 ──────────────────────────────────────────────
    ax.text(-1.8, 7.5, 'scRNA-seq 细胞（有标签）',
            ha='center', fontsize=13, fontweight='bold', color=color_sc)
    ax.text(-1.8, 7.0, '$|V_{sc}| = 16{,}569$',
            ha='center', fontsize=11, color=color_sc)
    ax.text(4.5, 7.5, 'Xenium spot（无标签）',
            ha='center', fontsize=13, fontweight='bold', color=color_sp)
    ax.text(4.5, 7.0, '$|V_{sp}| = 298{,}053$',
            ha='center', fontsize=11, color=color_sp)

    # 在两个域之间画一条隔离虚线
    ax.plot([0.8, 0.8], [1.2, 7.2], linestyle='--',
            color='#999999', linewidth=1.0, alpha=0.6)

    # ── 图例 ──────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='scRNA-seq 细胞节点',
               markerfacecolor=color_sc, markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Xenium spot 节点',
               markerfacecolor=color_sp, markersize=10),
        Line2D([0], [0], color=color_sc_knn, linestyle=':',
               linewidth=1.5, label='scRNA 内特征 kNN'),
        Line2D([0], [0], color=color_sp_feat, linestyle=':',
               linewidth=1.5, label='spot 内特征 kNN'),
        Line2D([0], [0], color=color_sp_spat, linestyle='-',
               linewidth=2.0, label='spot 空间 kNN'),
        Line2D([0], [0], color=color_cross, linestyle='-',
               linewidth=2.5, label='跨模态 mutual NN'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              ncol=3, bbox_to_anchor=(0.5, -0.05),
              fontsize=10, frameon=True, framealpha=0.95)

    # ── 标题 ──────────────────────────────────────────────────
    ax.text(1.5, 8.2, '跨模态联合异质图：四类边的协同作用',
            ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'✅ 图 4-2 已保存: {out_path}')


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    make_fig_3_2()
    make_fig_4_1()
    make_fig_4_2()
    print('\n全部示意图生成完毕。')

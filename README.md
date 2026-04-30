# 项目结构说明

```
project/
├── train.ipynb       # 主训练流程（10 个 Cell）
├── models.py         # GCN / GraphSAGE / GAT + 训练循环（含 DA）
├── topact.py         # TopACT baseline + Moran's I
├── eval.py           # Fig 1–10 论文图表生成
└── utils.py          # 预处理 / 图构建 / DA 损失函数
```

## 修复清单

| 级别 | 编号 | 问题 | 修复位置 |
|------|------|------|---------|
| P0 | ① | 双重独立归一化 | `utils.unified_normalize()` |
| P0 | ② | state_dict 浅拷贝 | `utils.save_best_state()` → deepcopy |
| P0 | ③ | 有向图未对称 | `utils.build_mutual_knn_graph()` + `to_undirected` |
| P0 | ④ | Seurat 标签评估风险 | `train.ipynb` Cell 10（独立可选 cell）|
| P1 | ⑤ | 缺 DA 机制 | `utils.mmd_loss` + `entropy_regularization` + 伪标签 |
| P1 | ⑥ | 无生物预处理 | `utils.log_normalize()` |
| P1 | ⑦ | 无类别权重 | `utils.build_combined_dataset()` → class_weights |
| P1 | ⑧ | 验证集未分层 | `StratifiedShuffleSplit` |
| P2 | ⑨ | 无 LR 调度器 | `ReduceLROnPlateau` in `models.run_experiment()` |
| P2 | ⑩ | 无梯度裁剪 | `clip_grad_norm_` in `models.train_epoch()` |
| P2 | ⑪ | 无 BatchNorm | `BatchNorm1d` in `GCN/GraphSAGE/GAT` |
| P3 | ⑫ | 无伪标签 | `utils.get_pseudo_labels()` in `models.train_epoch()` |
| P3 | ⑬ | 普通 kNN | Mutual kNN + `to_undirected` |
| P3 | ⑭ | 维度跨度大 | `proj_dim=512` 投影层（可选）|

## 论文图表清单

| 图号 | 内容 | 说明 |
|------|------|------|
| Fig 1 | scRNA UMAP | 参考集细胞类型分布 |
| Fig 2 | GNN 隐层 UMAP | 域适应效果 + 类型可分性 |
| Fig 3 | 训练曲线 | Loss 分量 + Val F1 |
| Fig 4 | 方法对比柱状图 | Acc / F1 / Kappa |
| Fig 5 | 混淆矩阵 | 每方法一张，行归一化 |
| Fig 6 | 空间预测图 | 4-panel 空间散点图 |
| Fig 7 | 细胞类型比例 | scRNA vs 各模型堆叠条形 |
| Fig 8 | 逐类 F1 热力图 | 稀有类型性能对比 |
| Fig 9 | 置信度分布 | 小提琴 + CDF |
| Fig 10 | Moran's I | 空间自相关按细胞类型 |

## 运行顺序

1. 先运行 `labelTransfer.ipynb`（R 预处理，生成 cache/）
2. 运行 `train.ipynb`（Cell 1 → Cell 9，Cell 10 可选）
~~~
# 1. 设置环境变量解决碎片问题（当前会话生效）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 永久写入 conda 环境（重启后也有效）
conda activate spatial_gnn
conda env config vars set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate spatial_gnn   # 重新激活使其生效
~~~
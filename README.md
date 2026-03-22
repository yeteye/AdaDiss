# AdaDiss

本科生毕业论文项目

## 实现思路

### 步骤1

下载人类肺部细胞 scRNA（有 cell type）

```bash
# 完整版
curl -O https://datasets.cellxgene.cziscience.com/dbb5ad81-1713-4aee-8257-396fbabe7c6e.h5ad

# 核心版
curl -O https://datasets.cellxgene.cziscience.com/4cb45d80-499a-48ae-a056-c71ac3552c94.h5ad
```

### 步骤2

下载人类肺癌数据集 Xenium（没有 cell type）

```bash
curl -O https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_V1_Human_Lung_Cancer_FFPE/Xenium_V1_Human_Lung_Cancer_FFPE_outs.zip
```

### 步骤3

找到两者共有的基因
因为两个数据的基因不完全一样，只能用共同基因训练模型

### 步骤4

在 scRNA 上训练分类器

### 步骤5

用分类器预测 Xenium 的 cell type

### 步骤6

得到 Xenium 的 pseudo label

### 步骤7

用 pseudo label 训练 GNN

# environment.yml
#
# 用法：
#   conda env create -f environment.yml
#   conda activate spatial_gnn
#
# 说明：
#   PyTorch Geometric 的 scatter/sparse 扩展通过 post-create 脚本安装（见下方）
#   R 包（Seurat, BPCells）同样在 post-create 步骤安装

name: spatial_gnn

channels:
  - conda-forge      # 优先级最高，覆盖 defaults
  - defaults

dependencies:
  # ── Python 版本 ──────────────────────────────────────────────────────
  - python=3.10

  # ── 科学计算核心 ──────────────────────────────────────────────────────
  - numpy=1.26
  - pandas=2.2
  - scipy=1.13
  - scikit-learn=1.5
  - joblib
  - tqdm

  # ── 可视化 ────────────────────────────────────────────────────────────
  - matplotlib=3.9
  - seaborn=0.13

  # ── 降维 / 聚类 ───────────────────────────────────────────────────────
  - umap-learn=0.5
  - leidenalg        # scRNA 聚类（可选，scanpy 需要）
  - python-igraph

  # ── 单细胞 / 空间数据格式 ────────────────────────────────────────────
  - h5py
  - hdf5
  - anndata=0.10

  # ── Jupyter ───────────────────────────────────────────────────────────
  - jupyterlab=4
  - notebook=7
  - ipykernel
  - ipywidgets
  - nbconvert

  # ── PyTorch（CPU 版本；GPU 见下方注释）───────────────────────────────
  # CPU 模式（无 GPU / macOS）:
  - pytorch=2.3
  - torchvision
  - cpuonly
  #
  # GPU 模式（CUDA 12.x）请注释掉上面 3 行，取消注释下面 3 行：
  # - pytorch=2.3
  # - torchvision
  # - pytorch-cuda=12.1

  # ── R + rpy2（%%R 魔法命令）──────────────────────────────────────────
  - rpy2=3.5
  - r-base=4.3
  - r-essentials=4.3   # R 常用包套装（dplyr, ggplot2, tidyr, etc.）
  - r-matrix
  - r-ggplot2
  - r-dplyr
  - r-tidyr
  - r-jsonlite
  - r-hdf5r            # 读取 .h5 文件（Flex 数据）
  - r-remotes          # 安装 GitHub 上的 R 包
  - r-arrow
  - r-irkernel         # R 的 Jupyter kernel
  - bioconductor-biocmanager

  # ── Conda 无法提供的包通过 pip 安装 ──────────────────────────────────
  - pip
  - pip:
    # scanpy（依赖 anndata，功能完整版）
    - scanpy==1.10
    # PyTorch Geometric 核心（版本需与 PyTorch 对应）
    # ⚠️  这里只装 torch-geometric 主包；
    #     scatter/sparse/cluster 等扩展需要匹配 CUDA 的 wheel，
    #     请运行 setup_env.sh 或 post_install.sh 完成安装。
    - torch-geometric==2.5
    # SeuratDisk（conda-forge 没有完整版）
    # 注：此处通过 pip 装 Python 绑定，R 端用 remotes::install_github
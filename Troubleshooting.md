#!/usr/bin/env bash
# =============================================================================
# post_install.sh  —  environment.yml 之后运行
#
# 安装两类 conda-forge 无法完整提供的依赖：
#   1. PyTorch Geometric 扩展（scatter / sparse / cluster）
#   2. R 包（Seurat 5, SeuratDisk, BPCells）
#
# 用法：
#   conda activate spatial_gnn
#   bash post_install.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; RESET='\033[0m'
step()    { echo -e "\n${CYAN}── $* ──────────────────────────────────────${RESET}"; }
success() { echo -e "${GREEN}  ✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}  ! $*${RESET}"; }
error()   { echo -e "${RED}  ✗ $*${RESET}"; exit 1; }

# 必须在激活的环境内运行
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    error "请先激活环境：conda activate spatial_gnn"
fi
echo "Running in conda env: ${CONDA_DEFAULT_ENV}"

# ── 1. 检测 PyTorch + CUDA 版本 ──────────────────────────────────────
step "1. 检测 PyTorch / CUDA 版本"

TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())")

if [[ "${CUDA_AVAIL}" == "True" ]]; then
    CUDA_VER=$(python -c "import torch; v=torch.version.cuda; \
        major,minor=v.split('.')[:2]; \
        print(f'cu{major}{minor}')")
    warn "CUDA 可用，版本标签：${CUDA_VER}"
else
    CUDA_VER="cpu"
    warn "无 GPU，使用 CPU 版本"
fi

PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html"
echo "  PyG wheel URL: ${PYG_WHEEL_URL}"

# ── 2. 安装 PyG 扩展 ─────────────────────────────────────────────────
step "2. 安装 PyTorch Geometric 扩展（pip）"

# 先卸载可能残留的旧版本
pip uninstall -y torch-scatter torch-sparse torch-cluster \
    torch-spline-conv 2>/dev/null || true

pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f "${PYG_WHEEL_URL}" \
    --quiet

# 验证
python -c "
import torch_scatter, torch_sparse, torch_cluster
print('  torch-scatter OK')
print('  torch-sparse  OK')
print('  torch-cluster OK')
"
success "PyG 扩展安装完成"

# ── 3. 安装 R 包 ─────────────────────────────────────────────────────
step "3. 安装 R 包（Seurat 5 / SeuratDisk / BPCells / IRkernel）"

Rscript - <<'RSCRIPT'
options(repos=c(CRAN="https://cran.r-project.org"))
options(warn=1)  # 立即打印 warning

# 3-1: BiocManager 初始化
if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager")
BiocManager::install(version="3.19", ask=FALSE, update=FALSE)

# 3-2: Seurat 5（conda-forge 的 r-seurat 可能是旧版，这里确保版本）
cat("Installing Seurat 5...\n")
if (packageVersion("Seurat") < "5.0.0") {
    install.packages("Seurat", repos=c(
        "https://satijalab.r-universe.dev",
        "https://cran.r-project.org"
    ))
}

# 3-3: SeuratObject（Seurat 5 依赖的底层对象系统）
if (!requireNamespace("SeuratObject", quietly=TRUE))
    install.packages("SeuratObject")

# 3-4: SeuratDisk（读写 h5Seurat 格式）
if (!requireNamespace("SeuratDisk", quietly=TRUE)) {
    cat("Installing SeuratDisk from GitHub...\n")
    remotes::install_github(
        "mojaveazure/seurat-disk",
        upgrade="never",
        quiet=TRUE,
        force=FALSE
    )
}

# 3-5: BPCells（大规模单细胞数据必须，支持 on-disk 矩阵）
if (!requireNamespace("BPCells", quietly=TRUE)) {
    cat("Installing BPCells from GitHub...\n")
    remotes::install_github(
        "bnprks/BPCells/r",
        upgrade="never",
        quiet=TRUE,
        force=FALSE
    )
}

# 3-6: IRkernel（将 R 注册为 Jupyter kernel）
if (!requireNamespace("IRkernel", quietly=TRUE))
    install.packages("IRkernel")

IRkernel::installspec(
    name="spatial_gnn_R",
    displayname="R 4.3 (spatial_gnn)"
)

# 3-7: 验证
pkgs <- c("Seurat","SeuratDisk","BPCells","IRkernel",
          "dplyr","ggplot2","Matrix","jsonlite","hdf5r","remotes")
for (pkg in pkgs) {
    if (requireNamespace(pkg, quietly=TRUE)) {
        v <- tryCatch(as.character(packageVersion(pkg)), error=function(e) "?")
        cat(sprintf("  ✓ %-15s %s\n", pkg, v))
    } else {
        cat(sprintf("  ✗ %-15s NOT FOUND\n", pkg))
    }
}
RSCRIPT

success "R 包安装完成"

# ── 4. 验证完整环境 ───────────────────────────────────────────────────
step "4. 完整环境验证"

python - <<'PYEOF'
import sys

checks = [
    ("numpy",          "import numpy as np; v=np.__version__"),
    ("pandas",         "import pandas as pd; v=pd.__version__"),
    ("scipy",          "import scipy; v=scipy.__version__"),
    ("scikit-learn",   "import sklearn; v=sklearn.__version__"),
    ("matplotlib",     "import matplotlib; v=matplotlib.__version__"),
    ("seaborn",        "import seaborn as sns; v=sns.__version__"),
    ("umap-learn",     "from umap import UMAP; v='OK'"),
    ("torch",          "import torch; v=f'{torch.__version__} CUDA={torch.cuda.is_available()}'"),
    ("torch_geometric","import torch_geometric as pyg; v=pyg.__version__"),
    ("torch_scatter",  "import torch_scatter; v='OK'"),
    ("torch_sparse",   "import torch_sparse; v='OK'"),
    ("torch_cluster",  "import torch_cluster; v='OK'"),
    ("rpy2",           "import rpy2; v=rpy2.__version__"),
    ("anndata",        "import anndata; v=anndata.__version__"),
    ("scanpy",         "import scanpy as sc; v=sc.__version__"),
]

failed = []
for name, expr in checks:
    try:
        ns = {}
        exec(expr, ns)
        print(f"  ✅ {name:<18} {ns.get('v','')}")
    except Exception as e:
        print(f"  ❌ {name:<18} {e}")
        failed.append(name)

if failed:
    print(f"\n⚠️  以下包安装失败，请检查：{failed}")
    sys.exit(1)
else:
    print("\n✅ 所有包验证通过！")
PYEOF

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  post_install.sh 完成！环境已就绪。${RESET}"
echo -e "${GREEN}═══════════════════════════════════════════════════${RESET}"
echo ""
echo "  启动 JupyterLab："
echo -e "    ${CYAN}jupyter lab${RESET}"
echo ""
echo "  Notebook 内 R 代码说明："
echo "    Cell 2 使用 '%%R' 魔法命令（Python kernel 调用 rpy2）"
echo "    不需要切换到 R kernel"
echo ""
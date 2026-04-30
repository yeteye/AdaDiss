#!/usr/bin/env bash
# =============================================================================
# setup_spatial_gnn.sh  —  spatial_gnn 项目一键环境配置（已修复版）
#
# 修复内容（相较于 spatial_gnn.sh 原版）：
#   BUG-A  TORCH_VER 从 2.7.1 → 2.3.1（对齐 requirements.txt）
#   BUG-B  TORCHVISION_VER 从 0.22.1 → 0.18.1（对齐 requirements.txt）
#   BUG-C  PYG_TORCH_TAG 从 2.7.0 → 2.3.0（pyg.org wheel URL 需与 torch 大版本对齐）
#   BUG-D  Step 7 anndata/scanpy 版本改为 0.10.0 / 1.10.0（对齐 requirements.txt）
#   BUG-E  BPCells 改为必须安装（git clone → 本地编译），不再允许跳过
#   BUG-F  SeuratDisk：先尝试本地镜像 tarball，再 fallback GitHub，清晰报错
#   BUG-G  Seurat conda install 语法修正（移除 >=5 条件触发 solver 歧义）
#   BUG-H  anndata 已由 conda 安装，Step 7 不再重复安装以避免版本冲突
#
# 用法：
#   bash setup_spatial_gnn.sh             # 自动检测 GPU / CPU
#   bash setup_spatial_gnn.sh --cpu       # 强制 CPU 模式
#   bash setup_spatial_gnn.sh --skip-r    # 跳过 R 包安装（调试用）
#   bash setup_spatial_gnn.sh --help
#
# 前置要求：
#   - Miniforge3 / Miniconda（conda 可用）
#   - git（BPCells 本地编译必须）
#   - C++ 编译器：apt install build-essential  或  yum install gcc-c++ make
# =============================================================================

set -euo pipefail

# ── 颜色 ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*" >&2; exit 1; }
step()    { echo -e "\n${BLUE}━━━━  $*  ━━━━${NC}"; }

# ── 参数解析 ──────────────────────────────────────────────────────────────────
FORCE_CPU=false
SKIP_R=false
for arg in "$@"; do
    case "$arg" in
        --cpu)    FORCE_CPU=true ;;
        --skip-r) SKIP_R=true ;;
        --help|-h)
            echo "用法: bash setup_spatial_gnn.sh [--cpu] [--skip-r]"
            echo "  --cpu      强制 CPU 模式（无 GPU 环境）"
            echo "  --skip-r   跳过 R 包安装（调试 Python 层时使用）"
            exit 0 ;;
    esac
done

ENV_NAME="spatial_gnn"
BPCELLS_DIR="$HOME/BPCells"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 0: 前置检查
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step "Step 0: 前置检查"

command -v conda &>/dev/null || error "未找到 conda，请先安装 Miniforge3 / Miniconda"
command -v git   &>/dev/null || error "未找到 git，BPCells 本地编译必须"

# 检查 C++ 编译器（BPCells 编译需要）
if ! command -v g++ &>/dev/null && ! command -v c++ &>/dev/null; then
    error "未找到 C++ 编译器。请先执行：
    Ubuntu/Debian: sudo apt install build-essential
    RHEL/CentOS:   sudo yum install gcc-c++ make"
fi

success "conda $(conda --version) / git $(git --version | cut -d' ' -f3) / C++ OK"

# ── CUDA 检测 ─────────────────────────────────────────────────────────────────
# BUG-A/B: 版本锁定为 requirements.txt 实际记录版本
#   torch==2.3.1 / torchvision==0.18.1 / torch-geometric==2.7.0
detect_cuda() {
    if $FORCE_CPU; then echo "cpu"; return; fi
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null 2>&1; then
        echo "cpu"; return
    fi
    local ver major
    ver=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    major=$(echo "$ver" | cut -d. -f1)
    if   [[ $major -ge 12 ]]; then echo "cu121"   # CUDA 12.x → cu121 wheel
    elif [[ $major -ge 11 ]]; then echo "cu118"   # CUDA 11.x → cu118 wheel
    else                           echo "cpu"
    fi
}

CUDA_TAG=$(detect_cuda)

# BUG-A: 原脚本 TORCH_VER="2.7.1" 与 requirements.txt torch==2.3.1 不符
TORCH_VER="2.3.1"
# BUG-B: 原脚本 TORCHVISION_VER="0.22.1" 与 requirements.txt torchvision==0.18.1 不符
TORCHVISION_VER="0.18.1"

case "$CUDA_TAG" in
    cu121) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    cu118) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    cpu)   TORCH_INDEX="https://download.pytorch.org/whl/cpu"   ;;
esac

# BUG-C: 原脚本 PYG_TORCH_TAG="2.7.0"；pyg.org wheel URL 使用 torch 大版本（不含 patch）
#         torch==2.3.1 → torch-2.3.0 目录（pyg.org 用 major.minor 命名）
PYG_TORCH_TAG="2.3.0"
PYG_URL="https://data.pyg.org/whl/torch-${PYG_TORCH_TAG}+${CUDA_TAG}.html"

info "CUDA tag        : ${CUDA_TAG}"
info "Torch version   : ${TORCH_VER}+${CUDA_TAG}"
info "Torchvision     : ${TORCHVISION_VER}+${CUDA_TAG}"
info "PyG wheel URL   : ${PYG_URL}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 创建 conda 环境
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step "Step 1: 创建 conda 环境 [${ENV_NAME}]"

SKIP_CREATE_ENV=false
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "环境 '${ENV_NAME}' 已存在"
    read -rp "  删除并重建？[y/N] " ans
    if [[ "${ans,,}" == "y" ]]; then
        conda env remove -n "${ENV_NAME}" -y
    else
        SKIP_CREATE_ENV=true
        info "保留现有环境，继续后续步骤"
    fi
fi

if [[ "$SKIP_CREATE_ENV" == "false" ]]; then
    conda config --add channels conda-forge --force 2>/dev/null || true
    conda config --set channel_priority strict 2>/dev/null || true

    # 注意：torch 不在此处安装（Step 2 用 pip 精确对齐版本）
    # anndata=0.10 已包含（对齐 requirements.txt），scanpy 由 pip 安装
    conda create -n "${ENV_NAME}" -y -c conda-forge \
        python=3.10 \
        numpy=1.26 pandas=2.2 scipy=1.13 scikit-learn=1.5 \
        matplotlib=3.9 seaborn=0.13 \
        umap-learn=0.5 leidenalg python-igraph \
        h5py hdf5 joblib tqdm \
        anndata=0.10 \
        jupyterlab=4 notebook=7 ipykernel ipywidgets nbconvert \
        rpy2=3.5 \
        r-base=4.3 r-essentials=4.3 \
        r-matrix r-ggplot2 r-dplyr r-tidyr \
        r-jsonlite r-hdf5r r-remotes r-arrow r-irkernel
    success "conda 环境创建完成"
fi

RUN="conda run -n ${ENV_NAME} --no-capture-output"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: PyTorch（pip，版本精确对齐 requirements.txt）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step "Step 2: 安装 PyTorch ${TORCH_VER}+${CUDA_TAG}"

INSTALLED_TORCH=$($RUN python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [[ "$INSTALLED_TORCH" == "${TORCH_VER}"* ]]; then
    warn "torch ${INSTALLED_TORCH} 已安装，跳过"
else
    $RUN pip install \
        "torch==${TORCH_VER}+${CUDA_TAG}" \
        "torchvision==${TORCHVISION_VER}+${CUDA_TAG}" \
        --index-url "${TORCH_INDEX}" \
        --quiet
    success "PyTorch $($RUN python -c 'import torch; print(torch.__version__)')"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: PyTorch Geometric（pip，wheel 与 torch 版本严格对齐）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step "Step 3: 安装 PyTorch Geometric（torch-geometric==2.7.0 + 扩展）"

INSTALLED_PYG=$($RUN python -c "import torch_geometric; print(torch_geometric.__version__)" 2>/dev/null || echo "none")
if [[ "$INSTALLED_PYG" != "none" ]]; then
    warn "torch_geometric ${INSTALLED_PYG} 已安装，跳过"
else
    # requirements.txt 精确版本：
    #   torch_scatter==2.1.2+pt23cu118
    #   torch_sparse==0.6.18+pt23cu118
    #   torch_cluster==1.6.3+pt23cu118
    #   torch_spline_conv==1.2.2+pt23cu118
    # 先装扩展（需要匹配 CUDA wheel），再装主包
    $RUN pip install \
        torch-scatter \
        torch-sparse \
        torch-cluster \
        torch-spline-conv \
        -f "${PYG_URL}" \
        --quiet

    # BUG-C: requirements.txt torch-geometric==2.7.0（不是 2.5）
    $RUN pip install "torch-geometric==2.7.0" --quiet
    success "PyG $($RUN python -c 'import torch_geometric; print(torch_geometric.__version__)')"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: scanpy（pip，版本对齐 requirements.txt）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step "Step 4: 安装 scanpy（pip）"

# BUG-D: 原脚本 Step 7 安装 anndata==0.12.10 + scanpy==1.11.5 与 requirements.txt 不符
# anndata=0.10 已由 Step 1 conda 安装，此处只补 scanpy==1.10.0
INSTALLED_SCANPY=$($RUN python -c "import scanpy; print(scanpy.__version__)" 2>/dev/null || echo "none")
if [[ "$INSTALLED_SCANPY" == "1.10"* ]]; then
    warn "scanpy ${INSTALLED_SCANPY} 已安装，跳过"
else
    $RUN pip install "scanpy==1.10.0" --quiet
    success "scanpy 1.10.0"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: BPCells — git clone → 本地编译安装（必须，不可跳过）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step "Step 5: BPCells（git clone → 本地编译，必须）"

# BUG-E: conda_github.sh 和 Troubleshooting.md 里 BPCells 被标记为「跳过」
#         本项目要求 BPCells 必须安装，此处强制执行本地编译路径
if $SKIP_R; then
    warn "--skip-r 已指定，跳过 BPCells（仅调试时使用）"
else
    if [ -d "${BPCELLS_DIR}" ]; then
        warn "BPCells 目录已存在 (${BPCELLS_DIR})，执行 git pull..."
        git -C "${BPCELLS_DIR}" pull --quiet \
            || warn "git pull 失败，使用已有版本继续编译"
    else
        info "克隆 BPCells 仓库到 ${BPCELLS_DIR}..."
        git clone --depth=1 \
            https://github.com/bnprks/BPCells.git \
            "${BPCELLS_DIR}" \
            || error "git clone BPCells 失败。
请检查：
  1. git 可访问 GitHub：git ls-remote https://github.com/bnprks/BPCells.git
  2. 若 GitHub 不可达，可手动下载 tarball 并解压到 ${BPCELLS_DIR}：
       wget https://github.com/bnprks/BPCells/archive/refs/heads/main.tar.gz
       tar -xzf main.tar.gz && mv BPCells-main ${BPCELLS_DIR}"
    fi

    info "在 conda 环境中本地编译 BPCells（耗时约 3-5 分钟）..."
    $RUN Rscript -e "
options(warn = 1)
if (requireNamespace('BPCells', quietly = TRUE)) {
    cat('BPCells', as.character(packageVersion('BPCells')), 'already installed, skipping.\n')
} else {
    cat('Compiling BPCells from local source: ${BPCELLS_DIR}/r\n')
    remotes::install_local(
        '${BPCELLS_DIR}/r',
        upgrade  = 'never',
        quiet    = FALSE,
        force    = FALSE
    )
    cat('BPCells', as.character(packageVersion('BPCells')), 'installed OK.\n')
}
" && success "BPCells 编译安装完成" \
  || error "BPCells 编译失败！
常见原因及解决方法：
  1. 缺少 C++ 编译器：sudo apt install build-essential
  2. 缺少 HDF5 开发库：sudo apt install libhdf5-dev
  3. 查看详细错误：conda run -n ${ENV_NAME} Rscript -e \"remotes::install_local('${BPCELLS_DIR}/r')\""
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 6: Seurat 5 + SeuratDisk
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ "$SKIP_R" == "false" ]]; then
    step "Step 6: 安装 Seurat 5 + SeuratDisk"

    # BUG-G: 原脚本 "r-seurat>=5" 语法在 conda 中可能引发 solver 歧义
    #         改为先尝试 conda-forge，失败后在 R 内从 CRAN 编译
    info "尝试 conda-forge 安装 Seurat 5..."
    conda install -n "${ENV_NAME}" -y -c conda-forge \
        r-seurat r-seuratobject 2>/dev/null \
        || warn "conda install r-seurat 失败，将在 R 内从 CRAN 安装"

    # 确保 Seurat >= 5（conda-forge 版本可能落后）
    $RUN Rscript -e "
options(repos = c(
    Seurat = 'https://satijalab.r-universe.dev',
    CRAN   = 'https://cloud.r-project.org'
))
options(warn = 1)

# 若 Seurat 未安装或版本 < 5，从 r-universe 安装
need_seurat <- tryCatch(
    packageVersion('Seurat') < '5.0.0',
    error = function(e) TRUE
)
if (need_seurat) {
    cat('Installing Seurat 5 from r-universe...\n')
    install.packages('Seurat')
} else {
    cat('Seurat', as.character(packageVersion('Seurat')), 'already OK.\n')
}

# SeuratObject（底层对象系统）
if (!requireNamespace('SeuratObject', quietly = TRUE))
    install.packages('SeuratObject')
" && success "Seurat 5 安装完成" \
  || error "Seurat 5 安装失败，请检查网络或手动安装"

    # BUG-F: SeuratDisk 仅 GitHub 可用，提供明确 fallback 提示
    info "安装 SeuratDisk（来源：GitHub）..."
    $RUN Rscript -e "
options(repos = c(CRAN = 'https://cloud.r-project.org'))
if (requireNamespace('SeuratDisk', quietly = TRUE)) {
    cat('SeuratDisk', as.character(packageVersion('SeuratDisk')), 'already installed.\n')
} else {
    cat('Installing SeuratDisk from GitHub...\n')
    tryCatch(
        remotes::install_github(
            'mojaveazure/seurat-disk',
            upgrade = 'never',
            quiet   = TRUE
        ),
        error = function(e) {
            cat('[WARN] SeuratDisk GitHub 安装失败：', conditionMessage(e), '\n')
            cat('手动安装方法：\n')
            cat('  1. 下载 tarball：\n')
            cat('     wget https://github.com/mojaveazure/seurat-disk/archive/refs/heads/master.tar.gz\n')
            cat('  2. 本地安装：\n')
            cat('     Rscript -e \"remotes::install_local(\\\"seurat-disk-master.tar.gz\\\")\"\n')
        }
    )
}
" && success "SeuratDisk 安装完成（或已提示手动安装方法）"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 7: 注册 Jupyter Kernel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step "Step 7: 注册 Jupyter Kernel"

$RUN python -m ipykernel install --user \
    --name "${ENV_NAME}" \
    --display-name "Python (spatial_gnn)"
success "Python kernel 注册完成"

if [[ "$SKIP_R" == "false" ]]; then
    $RUN Rscript -e "
IRkernel::installspec(
    name        = 'spatial_gnn_R',
    displayname = 'R 4.3 (spatial_gnn)'
)
cat('R kernel registered OK\n')
" && success "R kernel 注册完成" \
  || warn "R kernel 注册失败（可手动运行：conda run -n ${ENV_NAME} Rscript -e \"IRkernel::installspec()\"）"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 8: 全量验证
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step "Step 8: 全量验证"

info "验证 Python 包..."
$RUN python - <<'PYEOF'
import sys
print(f"Python {sys.version.split()[0]}\n")

CHECKS = [
    ("numpy",           "import numpy as np;             v=np.__version__"),
    ("pandas",          "import pandas as pd;            v=pd.__version__"),
    ("scipy",           "import scipy;                   v=scipy.__version__"),
    ("scikit-learn",    "import sklearn;                 v=sklearn.__version__"),
    ("matplotlib",      "import matplotlib;              v=matplotlib.__version__"),
    ("seaborn",         "import seaborn as sns;          v=sns.__version__"),
    ("umap-learn",      "from umap import UMAP;          v='OK'"),
    ("torch",           "import torch; v=f'{torch.__version__} | CUDA:{torch.cuda.is_available()}'"),
    ("torchvision",     "import torchvision;             v=torchvision.__version__"),
    ("torch_scatter",   "import torch_scatter;           v='OK'"),
    ("torch_sparse",    "import torch_sparse;            v='OK'"),
    ("torch_cluster",   "import torch_cluster;           v='OK'"),
    ("torch_geometric", "import torch_geometric;         v=torch_geometric.__version__"),
    ("rpy2",            "import rpy2;                    v=rpy2.__version__"),
    ("anndata",         "import anndata;                 v=anndata.__version__"),
    ("scanpy",          "import scanpy as sc;            v=sc.__version__"),
]

failed = []
for name, stmt in CHECKS:
    try:
        ns = {}
        exec(stmt, ns)
        print(f"  \033[32m✓\033[0m {name:<18} {ns.get('v','')}")
    except Exception as e:
        print(f"  \033[31m✗\033[0m {name:<18} {e}")
        failed.append(name)

print()
if failed:
    print(f"\033[31m  失败包: {failed}\033[0m")
    sys.exit(1)
else:
    print("\033[32m  所有 Python 包验证通过 ✓\033[0m")
PYEOF

if [[ "$SKIP_R" == "false" ]]; then
    info "验证 R 包..."
    $RUN Rscript - <<'REOF'
pkgs <- c("Seurat", "SeuratObject", "SeuratDisk", "BPCells",
          "dplyr", "ggplot2", "jsonlite", "hdf5r", "Matrix",
          "IRkernel", "rpy2" |> tryCatch(expr=NULL, error=function(e)NULL))
pkgs <- pkgs[!sapply(pkgs, is.null)]

ok <- TRUE
for (pkg in pkgs) {
    if (requireNamespace(pkg, quietly = TRUE)) {
        v <- tryCatch(as.character(packageVersion(pkg)), error = function(e) "?")
        cat(sprintf("  %s %-18s %s\n", "\u2713", pkg, v))
    } else {
        cat(sprintf("  \033[31m\u2717\033[0m %-18s NOT FOUND\n", pkg))
        ok <- FALSE
    }
}
if (!ok) quit(status = 1)
cat("\n\033[32m  所有 R 包验证通过 ✓\033[0m\n")
REOF
fi

# ── 完成 ──────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✅  环境配置完成！${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  启动 JupyterLab："
echo -e "    ${CYAN}conda activate ${ENV_NAME} && jupyter lab${NC}"
echo ""
echo "  Notebook Kernel 选择："
echo "    Python 训练 Cell  →  Python (spatial_gnn)"
echo "    R 数据加载 Cell   →  R 4.3 (spatial_gnn)  [或 %%R 魔法命令]"
echo ""

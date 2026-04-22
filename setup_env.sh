#!/usr/bin/env bash
# =============================================================================
# setup_env.sh  —  项目完整环境安装脚本
#
# 使用方式：
#   chmod +x setup_env.sh
#   bash setup_env.sh            # 自动检测 GPU / CPU
#   bash setup_env.sh --cpu      # 强制 CPU 模式
#
# 平台：Linux (Ubuntu 20.04+) / macOS (无 CUDA)
# 预计时间：20–35 分钟（首次，取决于网速）
# =============================================================================

set -euo pipefail

# ── 颜色输出 ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; RESET='\033[0m'
info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
step()    { echo -e "\n${CYAN}══════════════════════════════════════════${RESET}"; \
            echo -e "${CYAN}  $*${RESET}"; \
            echo -e "${CYAN}══════════════════════════════════════════${RESET}"; }

ENV_NAME="spatial_gnn"
PYTHON_VER="3.10"

# ── 解析参数 ──────────────────────────────────────────────────────────
FORCE_CPU=false
for arg in "$@"; do
    [[ "$arg" == "--cpu" ]] && FORCE_CPU=true
done

# ── 检测 CUDA ─────────────────────────────────────────────────────────
detect_cuda() {
    if $FORCE_CPU; then
        echo "cpu"; return
    fi
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local ver
        ver=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
        major=$(echo "$ver" | cut -d. -f1)
        if   [[ $major -ge 12 ]]; then echo "cu118"
        elif [[ $major -ge 11 ]]; then echo "cu118"
        else                           echo "cpu"
        fi
    else
        echo "cpu"
    fi
}

CUDA_TAG=$(detect_cuda)
info "Detected CUDA tag: ${CUDA_TAG}"

# ── 检测 conda ────────────────────────────────────────────────────────
step "Step 0: 检查 conda 安装"
if ! command -v conda &>/dev/null; then
    error "conda not found. 请先安装 Miniforge3（推荐）或 Miniconda：
  curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh
  bash miniforge.sh -b -p \$HOME/miniforge3
  source \$HOME/miniforge3/etc/profile.d/conda.sh"
fi
success "conda found: $(conda --version)"

# 确保 conda-forge 优先
conda config --add channels conda-forge --force 2>/dev/null || true
conda config --set channel_priority strict 2>/dev/null || true

# ── Step 1: 创建环境 ──────────────────────────────────────────────────
step "Step 1: 创建 conda 环境 [${ENV_NAME}]"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists."
    read -rp "  Remove and recreate? [y/N] " ans
    if [[ "${ans,,}" == "y" ]]; then
        conda env remove -n "${ENV_NAME}" -y
    else
        info "Skipping env creation, using existing."; goto_step2=true
    fi
fi

if [[ -z "${goto_step2:-}" ]]; then
    conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y \
        -c conda-forge \
        --override-channels
    success "Environment created."
fi

# ── 激活环境（脚本内通过 conda run 调用）─────────────────────────────
RUN="conda run -n ${ENV_NAME} --no-capture-output"

# ── Step 2: 科学计算基础包（全部 conda-forge）──────────────────────────
step "Step 2: 安装科学计算基础包（conda-forge）"

conda install -n "${ENV_NAME}" -y -c conda-forge \
    numpy=1.26 \
    pandas=2.2 \
    scipy=1.13 \
    scikit-learn=1.5 \
    matplotlib=3.9 \
    seaborn=0.13 \
    umap-learn=0.5 \
    joblib \
    tqdm \
    h5py \
    leidenalg \
    python-igraph

success "Scientific stack installed."

# ── Step 3: PyTorch（conda-forge 或 pytorch 频道）────────────────────
step "Step 3: 安装 PyTorch + torchvision"

if [[ "${CUDA_TAG}" == "cpu" ]]; then
    info "Installing CPU-only PyTorch..."
    conda install -n "${ENV_NAME}" -y -c conda-forge \
        pytorch=2.3 torchvision cpuonly
else
    info "Installing GPU PyTorch (${CUDA_TAG})..."
    # conda-forge 的 pytorch-gpu 包含 CUDA 库，无需额外安装 cudatoolkit
    conda install -n "${ENV_NAME}" -y \
        -c pytorch -c nvidia \
        pytorch=2.3 torchvision pytorch-cuda=11.8 
    # 如果上面失败，退回 pip 安装（见注释）
fi

success "PyTorch installed: $(${RUN} python -c 'import torch; print(torch.__version__)')"

# ── Step 4: PyG（必须 pip，conda-forge 版本不完整）───────────────────
step "Step 4: 安装 PyTorch Geometric (pip)"

# PyG 的 conda-forge 包缺少 torch_sparse 等扩展，必须用 pip
PYG_URL="https://data.pyg.org/whl/torch-2.3.0+${CUDA_TAG}.html"
info "PyG wheel URL: ${PYG_URL}"

${RUN} pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f "${PYG_URL}" \
    --quiet

${RUN} pip install torch-geometric --quiet

success "PyG installed: $(${RUN} python -c 'import torch_geometric; print(torch_geometric.__version__)')"

# ── Step 5: Jupyter 环境 ──────────────────────────────────────────────
step "Step 5: 安装 Jupyter + R 魔法命令"

conda install -n "${ENV_NAME}" -y -c conda-forge \
    jupyterlab=4 \
    notebook=7 \
    ipykernel \
    ipywidgets \
    rpy2=3.5 \
    r-base=4.3 \
    r-irkernel

success "Jupyter installed."

# 注册 Python kernel
${RUN} python -m ipykernel install --user \
    --name "${ENV_NAME}" \
    --display-name "Python (spatial_gnn)"

# ── Step 6: R 包 ──────────────────────────────────────────────────────
step "Step 6: 安装 R 包（conda-forge + R 源内安装）"

# 能用 conda-forge 的 R 包优先用 conda（避免编译）
conda install -n "${ENV_NAME}" -y -c conda-forge \
    r-seurat=5 \
    r-matrix \
    r-ggplot2 \
    r-dplyr \
    r-tidyr \
    r-jsonlite \
    r-hdf5r \
    r-remotes \
    r-arrow \
    bioconductor-biocmanager

info "Installing Seurat-related R packages from source..."
${RUN} Rscript - <<'RSCRIPT'
# SeuratDisk（CRAN 有，但推荐 GitHub 版）
if (!requireNamespace("SeuratDisk", quietly=TRUE)) {
    remotes::install_github("mojaveazure/seurat-disk", upgrade="never", quiet=TRUE)
}

# BPCells（大数据集必须）
if (!requireNamespace("BPCells", quietly=TRUE)) {
    remotes::install_github("bnprks/BPCells/r", upgrade="never", quiet=TRUE)
}

# IRkernel（R 核心在 Jupyter 中运行）
if (!requireNamespace("IRkernel", quietly=TRUE)) {
    install.packages("IRkernel", repos="https://cran.r-project.org", quiet=TRUE)
}
IRkernel::installspec(name="spatial_gnn_R", displayname="R (spatial_gnn)")

cat("R packages OK\n")
RSCRIPT

success "R packages installed."

# ── Step 7: 其他 pip 包 ───────────────────────────────────────────────
step "Step 7: 安装其他工具包（pip）"

${RUN} pip install \
    anndata==0.10 \
    scanpy==1.10 \
    --quiet

success "Additional packages installed."

# ── Step 8: 验证 ──────────────────────────────────────────────────────
step "Step 8: 环境验证"

${RUN} python - <<'PYEOF'
import sys
print(f"Python {sys.version}")

errors = []

def check(name, import_str):
    try:
        exec(import_str)
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        errors.append(name)

check("numpy",         "import numpy as np; print(f'     numpy {np.__version__}')")
check("pandas",        "import pandas as pd; print(f'     pandas {pd.__version__}')")
check("scipy",         "import scipy; print(f'     scipy {scipy.__version__}')")
check("scikit-learn",  "import sklearn; print(f'     sklearn {sklearn.__version__}')")
check("matplotlib",    "import matplotlib; print(f'     matplotlib {matplotlib.__version__}')")
check("seaborn",       "import seaborn; print(f'     seaborn {seaborn.__version__}')")
check("umap-learn",    "import umap; print(f'     umap-learn OK')")
check("torch",         "import torch; print(f'     torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')")
check("torch_geometric","import torch_geometric; print(f'     pyg {torch_geometric.__version__}')")
check("torch_scatter", "import torch_scatter; print(f'     torch_scatter OK')")
check("torch_sparse",  "import torch_sparse; print(f'     torch_sparse OK')")
check("rpy2",          "import rpy2; print(f'     rpy2 {rpy2.__version__}')")
check("anndata",       "import anndata; print(f'     anndata {anndata.__version__}')")
check("scanpy",        "import scanpy; print(f'     scanpy {scanpy.__version__}')")

if errors:
    print(f"\n⚠️  Failed: {errors}")
    sys.exit(1)
else:
    print("\n✅ All packages OK!")
PYEOF

success "Environment setup complete!"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  环境配置完成！${RESET}"
echo -e "${GREEN}═══════════════════════════════════════════════════${RESET}"
echo ""
echo "  激活环境："
echo -e "    ${CYAN}conda activate ${ENV_NAME}${RESET}"
echo ""
echo "  启动 JupyterLab："
echo -e "    ${CYAN}jupyter lab${RESET}"
echo ""
echo "  Notebook Kernel 选择："
echo "    Python kernel → 'Python (spatial_gnn)'"
echo "    R kernel      → 'R (spatial_gnn)'"
echo ""
echo "  注意：Cell 2 (R 数据加载) 使用 %%R 魔法命令"
echo "        需要在 Python kernel 下运行（rpy2 自动调 R）"
echo ""
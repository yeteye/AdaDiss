#!/usr/bin/env bash
# =============================================================================
# LabelTransfer 项目一键配置脚本
# 功能：创建 conda 环境、安装依赖、注册 R kernel、下载数据
# 用法：bash setup.sh [--skip-data] [--skip-env]
# =============================================================================

set -euo pipefail

# ─── 颜色输出 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ─── 参数解析 ─────────────────────────────────────────────────────────────────
SKIP_DATA=false
SKIP_ENV=false
for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --skip-env)  SKIP_ENV=true  ;;
        --help|-h)
            echo "用法: bash setup.sh [--skip-data] [--skip-env]"
            echo "  --skip-data   跳过数据下载步骤（已有数据时使用）"
            echo "  --skip-env    跳过 conda 环境安装步骤（已有环境时使用）"
            exit 0 ;;
    esac
done

# ─── 配置变量 ─────────────────────────────────────────────────────────────────
ENV_NAME="LabelTransfer"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BPCELLS_DIR="$HOME/BPCells"

# ─── 数据集配置（修改为 V1 Addon FFPE，版本 2.0.0）────────────────────────────
XENIUM_ZIP="Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs.zip"
XENIUM_DIR="Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs"
XENIUM_URL="https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE/${XENIUM_ZIP}"

FLEX_H5="17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
FLEX_H5_URL="https://cf.10xgenomics.com/samples/cell-exp/8.0.1/17k_Ovarian_Cancer_scFFPE/${FLEX_H5}"

ANNOT_CSV="FLEX_Ovarian_Barcode_Cluster_Annotation.csv"
ANNOT_URL="https://cf.10xgenomics.com/supp/cell-exp/${ANNOT_CSV}"

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   LabelTransfer 项目环境配置脚本${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  📁 项目根目录: ${PROJECT_ROOT}"
echo -e "  🐍 conda 环境: ${ENV_NAME}"
echo -e "  📦 Xenium 数据集: ${XENIUM_DIR}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# ─── 步骤 0：前置检查 ─────────────────────────────────────────────────────────
info "检查前置依赖..."
command -v conda  &>/dev/null || error "未找到 conda，请先安装 Anaconda / Miniconda"
command -v git    &>/dev/null || error "未找到 git，请先安装 git"
command -v aria2c &>/dev/null || error "未找到 aria2c，请先安装：
  Manjaro/Arch : sudo pacman -S aria2
  Ubuntu/Debian: sudo apt install aria2
  macOS        : brew install aria2"
success "前置依赖检查通过（conda / git / aria2c）"

# ─── 步骤 1：创建 conda 环境 ──────────────────────────────────────────────────
if [ "$SKIP_ENV" = false ]; then
    echo -e "\n${BLUE}[步骤 1/4]${NC} 创建 conda 环境..."

    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "环境 '${ENV_NAME}' 已存在，跳过创建"
        warn "如需重建请先运行: conda env remove -n ${ENV_NAME}"
    else
        info "创建 conda 环境（耗时约 5-15 分钟）..."
        conda create -y -n "${ENV_NAME}" -c conda-forge \
            r-base=4.3     \
            hdf5           \
            r-hdf5r        \
            r-seurat       \
            r-seuratobject \
            r-arrow        \
            r-tidyverse    \
            r-ggplot2      \
            r-ggpmisc      \
            r-cowplot      \
            r-gridextra    \
            r-viridis      \
            r-hrbrthemes   \
            r-jsonlite     \
            r-remotes      \
            r-irkernel     \
            jupyter        \
            notebook       \
            ipykernel
        success "conda 环境创建完成"
    fi

    CONDA_R=$(conda run -n "${ENV_NAME}" which R)
    info "R 路径: ${CONDA_R}"

    # ─── 步骤 2：安装 BPCells ────────────────────────────────────────────────
    echo -e "\n${BLUE}[步骤 2/4]${NC} 安装 BPCells..."

    if [ -d "${BPCELLS_DIR}" ]; then
        warn "BPCells 目录已存在，执行 git pull 更新..."
        git -C "${BPCELLS_DIR}" pull || warn "git pull 失败，继续使用现有版本"
    else
        info "克隆 BPCells 仓库..."
        git clone https://github.com/bnprks/BPCells.git "${BPCELLS_DIR}"
    fi

    info "在 conda 环境中编译安装 BPCells（耗时约 3-5 分钟）..."
    conda run -n "${ENV_NAME}" Rscript -e \
        "remotes::install_local('${BPCELLS_DIR}/r', upgrade = 'never')" \
        && success "BPCells 安装完成" \
        || error "BPCells 安装失败，请检查上方错误信息"

    # ─── 步骤 3：安装 SeuratDisk ─────────────────────────────────────────────
    info "安装 SeuratDisk..."
    conda run -n "${ENV_NAME}" Rscript -e \
        "if (!requireNamespace('SeuratDisk', quietly=TRUE)) {
            remotes::install_github('mojaveazure/seurat-disk', upgrade = 'never')
         } else { cat('SeuratDisk 已安装，跳过\n') }" \
        && success "SeuratDisk 安装完成" \
        || warn "SeuratDisk 安装失败（可选包，不影响核心流程）"

    # ─── 步骤 4：注册 Jupyter R kernel ───────────────────────────────────────
    info "注册 R Kernel 到 Jupyter..."
    conda run -n "${ENV_NAME}" Rscript -e \
        "IRkernel::installspec(name='LabelTransfer', displayname='R_LabelTransfer')" \
        && success "R Kernel 注册完成（内核名: R_LabelTransfer）" \
        || error "R Kernel 注册失败"

else
    warn "--skip-env 已启用，跳过环境安装步骤"
fi

# ─── 步骤 3：创建目录结构 ─────────────────────────────────────────────────────
echo -e "\n${BLUE}[步骤 3/4]${NC} 创建项目目录结构..."
cd "${PROJECT_ROOT}"

DIRS=(
    "data/raw/flex"
    "data/raw/xenium"
    "data/cache"
    "data/backup"
    "data/bpcells/ovarian_flex_counts"
    "data/bpcells/ovarian_xenium_counts"
    "results/predictions"
    "results/exports"
    "results/seurat"
    "reports/logs"
    "plots/01_quality_control"
    "plots/02_annotation"
    "plots/03_validation"
    "plots/04_spatial"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "${dir}" ]; then
        mkdir -p "${dir}"
        info "  📁 创建: ${dir}"
    fi
done
success "目录结构已就绪"

# ─── 步骤 4：下载数据 ─────────────────────────────────────────────────────────
if [ "$SKIP_DATA" = false ]; then
    echo -e "\n${BLUE}[步骤 4/4]${NC} 下载原始数据（使用 aria2c 多线程加速）..."

    # aria2c 公共参数
    ARIA2_OPTS="-x 16 -s 16 -k 1M -c --console-log-level=notice"

    cd "${PROJECT_ROOT}/data/raw"

    # ── Xenium 数据（V1 Addon FFPE，约 14 GB）────────────────────────────────
    echo -e "\n  ${CYAN}[Xenium]${NC} Xenium V1 Human Ovarian Cancer Addon FFPE (v2.0.0)"
    if [ -d "xenium/${XENIUM_DIR}" ]; then
        warn "Xenium 数据目录已存在，跳过下载"
        warn "路径: xenium/${XENIUM_DIR}"
    else
        info "下载 Xenium 数据（约 14 GB，16 线程）..."
        aria2c ${ARIA2_OPTS} \
            --dir="xenium" \
            --out="${XENIUM_ZIP}" \
            "${XENIUM_URL}"

        info "选择性解压（只提取必要文件，节省磁盘和时间）..."
        unzip -j "xenium/${XENIUM_ZIP}" \
            "*/transcripts.parquet" \
            "*/cells.csv.gz" \
            "*/cells.parquet" \
            "*/cell_feature_matrix.h5" \
            "*/gene_panel.json" \
            -d "xenium/${XENIUM_DIR}/"

        info "删除 ZIP 文件（已解压）..."
        rm -f "xenium/${XENIUM_ZIP}"

        success "Xenium 数据下载并解压完成"
        info "解压内容:"
        ls -lh "xenium/${XENIUM_DIR}/"
    fi

    # ── Flex scRNA-seq h5（约 1.3 GB）────────────────────────────────────────
    echo -e "\n  ${CYAN}[Flex]${NC} scRNA-seq 计数矩阵 (.h5)"
    if [ -f "flex/${FLEX_H5}" ]; then
        warn "Flex h5 文件已存在，跳过下载"
    else
        info "下载 Flex scRNA-seq h5（约 1.3 GB，16 线程）..."
        aria2c ${ARIA2_OPTS} \
            --dir="flex" \
            --out="${FLEX_H5}" \
            "${FLEX_H5_URL}"
        success "Flex h5 文件下载完成"
    fi

    # ── 细胞类型预注释 CSV（约 1 MB）─────────────────────────────────────────
    echo -e "\n  ${CYAN}[Flex]${NC} 细胞类型预注释文件 (.csv)"
    if [ -f "flex/${ANNOT_CSV}" ]; then
        warn "注释文件已存在，跳过下载"
    else
        info "下载细胞类型注释文件（约 1 MB）..."
        aria2c -x 4 -s 4 -c \
            --dir="flex" \
            --out="${ANNOT_CSV}" \
            "${ANNOT_URL}"
        success "注释文件下载完成"
    fi

    cd "${PROJECT_ROOT}"

    # ── 最终校验 ─────────────────────────────────────────────────────────────
    echo -e "\n${CYAN}[校验]${NC} 检查必要文件是否存在..."
    REQUIRED_FILES=(
        "data/raw/xenium/${XENIUM_DIR}/transcripts.parquet"
        "data/raw/xenium/${XENIUM_DIR}/cells.csv.gz"
        "data/raw/xenium/${XENIUM_DIR}/cell_feature_matrix.h5"
        "data/raw/flex/${FLEX_H5}"
        "data/raw/flex/${ANNOT_CSV}"
    )
    ALL_OK=true
    for f in "${REQUIRED_FILES[@]}"; do
        if [ -f "${f}" ]; then
            SIZE=$(du -sh "${f}" | cut -f1)
            success "  ✅ ${f}  (${SIZE})"
        else
            warn     "  ❌ 缺失: ${f}"
            ALL_OK=false
        fi
    done
    if [ "$ALL_OK" = false ]; then
        warn "部分文件缺失，请重新运行脚本或手动下载"
    fi

else
    warn "--skip-data 已启用，跳过数据下载步骤"
fi

# ─── 完成 ─────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   ✅ 配置完成！${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\n  启动 Jupyter Notebook："
echo -e "  ${CYAN}conda activate ${ENV_NAME} && jupyter notebook${NC}"
echo -e "\n  在 Jupyter 中选择内核: ${CYAN}R_LabelTransfer${NC}\n"

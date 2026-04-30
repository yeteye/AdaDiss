#!/usr/bin/env bash
# =============================================================================
# conda_mirror_setup.sh — conda / pip 国内镜像源一键配置
#
# 解决问题：conda.anaconda.org / repo.anaconda.com 连接超时
# 方案    ：全部替换为清华镜像（TUNA），覆盖 conda-forge
# 用法    ：bash conda_mirror_setup.sh [--test] [--reset]
#   --test   仅测试当前网络连通性，不修改配置
#   --reset  恢复官方源（删除镜像配置）
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── 参数 ─────────────────────────────────────────────────────────────────────
TEST_ONLY=false
RESET=false
for arg in "$@"; do
    case $arg in
        --test)  TEST_ONLY=true ;;
        --reset) RESET=true ;;
        --help|-h)
            echo "用法: bash conda_mirror_setup.sh [--test] [--reset]"
            echo "  --test   仅测试连通性"
            echo "  --reset  恢复官方源"
            exit 0 ;;
    esac
done

CONDARC="$HOME/.condarc"
BACKUP="$HOME/.condarc.bak.$(date +%Y%m%d_%H%M%S)"

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   conda 国内镜像源配置脚本${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# ─── 恢复官方源 ───────────────────────────────────────────────────────────────
if [ "$RESET" = true ]; then
    info "恢复官方源..."
    conda config --remove-key channels         2>/dev/null || true
    conda config --remove-key default_channels 2>/dev/null || true
    conda config --remove-key custom_channels  2>/dev/null || true
    conda config --set show_channel_urls false 2>/dev/null || true
    success "已恢复官方源，当前 .condarc:"
    cat "$CONDARC" 2>/dev/null || echo "  (文件为空)"
    exit 0
fi

# ─── 连通性测试函数 ───────────────────────────────────────────────────────────
check_url() {
    local name="$1"
    local url="$2"
    local timeout=8
    if curl -sf --connect-timeout "$timeout" --max-time "$timeout" \
            -o /dev/null "$url" 2>/dev/null; then
        success "  ✅ ${name}  (${url})"
        return 0
    else
        warn    "  ❌ ${name}  (${url})"
        return 1
    fi
}

# ─── 连通性测试 ───────────────────────────────────────────────────────────────
echo -e "${BLUE}[1/3]${NC} 测试源连通性..."
echo ""
echo "  官方源（通常被限制）："
check_url "conda.anaconda.org (conda-forge官方)" \
    "https://conda.anaconda.org/conda-forge/linux-64/repodata.json.zst" || true
check_url "repo.anaconda.com  (defaults官方)"    \
    "https://repo.anaconda.com/pkgs/main/linux-64/repodata.json.zst"    || true

echo ""
echo "  国内镜像源："
TUNA_OK=false
USTC_OK=false
check_url "清华 TUNA (main)"     \
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/repodata.json.zst"  \
    && TUNA_OK=true || true
check_url "清华 TUNA (conda-forge)" \
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/repodata.json.zst" \
    && TUNA_OK=true || true
check_url "中科大 USTC (main)"   \
    "https://mirrors.ustc.edu.cn/anaconda/pkgs/main/linux-64/repodata.json.zst" \
    && USTC_OK=true || true

if [ "$TEST_ONLY" = true ]; then
    echo ""
    info "--test 模式，不修改配置。"
    exit 0
fi

# ─── 选择最优镜像 ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}[2/3]${NC} 配置镜像源..."

# 根据连通性选择：优先清华，次选中科大
# 两者都不通则仍配置清华（在某些网络下延迟但能通）
if [ "$TUNA_OK" = true ]; then
    MIRROR_BASE="https://mirrors.tuna.tsinghua.edu.cn/anaconda"
    MIRROR_NAME="清华 TUNA"
elif [ "$USTC_OK" = true ]; then
    MIRROR_BASE="https://mirrors.ustc.edu.cn/anaconda"
    MIRROR_NAME="中科大 USTC"
else
    warn "两个镜像均无法连通，但仍配置清华镜像（可能是临时问题）"
    MIRROR_BASE="https://mirrors.tuna.tsinghua.edu.cn/anaconda"
    MIRROR_NAME="清华 TUNA（备用）"
fi

info "选用镜像: ${MIRROR_NAME}"

# ─── 备份旧配置 ───────────────────────────────────────────────────────────────
if [ -f "$CONDARC" ]; then
    cp "$CONDARC" "$BACKUP"
    info "已备份旧配置: ${BACKUP}"
fi

# ─── 写入新 .condarc ──────────────────────────────────────────────────────────
cat > "$CONDARC" << EOF
# conda 镜像配置 — 由 conda_mirror_setup.sh 生成于 $(date)
# 镜像源: ${MIRROR_NAME}
# 恢复官方源: bash conda_mirror_setup.sh --reset

channels:
  - defaults

# ── 覆盖 defaults 各子频道 ────────────────────────────────────────────────────
default_channels:
  - ${MIRROR_BASE}/pkgs/main
  - ${MIRROR_BASE}/pkgs/r
  - ${MIRROR_BASE}/pkgs/msys2

# ── conda-forge 镜像（关键：替换 conda.anaconda.org/conda-forge）────────────
custom_channels:
  conda-forge: ${MIRROR_BASE}/cloud
  bioconda:    ${MIRROR_BASE}/cloud
  pytorch:     ${MIRROR_BASE}/cloud
  nvidia:      ${MIRROR_BASE}/cloud

# ── 性能 & 稳定性设置 ────────────────────────────────────────────────────────
show_channel_urls: true     # 安装时显示来源 URL，便于确认是否命中镜像
channel_priority: flexible  # flexible 允许跨 channel 解析依赖（比 strict 更宽松）
auto_activate_base: false   # 不自动激活 base 环境

# ── 网络超时设置（容器/弱网环境适用）────────────────────────────────────────
remote_connect_timeout_secs:  10
remote_read_timeout_secs:     60
remote_max_retries:           3
ssl_verify: true              # 保持 SSL 验证（安全）
EOF

success ".condarc 写入完成: ${CONDARC}"
echo ""
echo "  当前配置内容："
echo "  ─────────────────────────────────────────────"
sed 's/^/  /' "$CONDARC"
echo "  ─────────────────────────────────────────────"

# ─── 配置 pip 镜像（顺带解决 pip 也超时的问题）──────────────────────────────
echo ""
echo -e "${BLUE}[3/3]${NC} 配置 pip 镜像..."

PIP_CONF_DIR="$HOME/.config/pip"
PIP_CONF="$PIP_CONF_DIR/pip.conf"
mkdir -p "$PIP_CONF_DIR"

cat > "$PIP_CONF" << EOF
# pip 镜像配置 — 由 conda_mirror_setup.sh 生成于 $(date)
[global]
index-url         = https://pypi.tuna.tsinghua.edu.cn/simple
extra-index-url   = https://pypi.mirrors.ustc.edu.cn/simple/
trusted-host       = pypi.tuna.tsinghua.edu.cn
                     pypi.mirrors.ustc.edu.cn
timeout            = 60
EOF

success "pip 配置写入完成: ${PIP_CONF}"

# ─── 清除 conda 缓存（旧的 repodata 可能缓存了超时错误）────────────────────
echo ""
info "清除 conda repodata 缓存（避免旧失败缓存干扰）..."
conda clean --index-cache -y 2>/dev/null && success "缓存清除完成" || warn "缓存清除失败（可忽略）"

# ─── 验证：用新源更新 conda 自身 ─────────────────────────────────────────────
echo ""
info "验证新配置（conda info）..."
conda info 2>/dev/null | grep -E "channel URLs|package cache|envs" || true

echo ""
info "测试从镜像拉取 conda-forge repodata..."
if conda search --channel conda-forge numpy --override-channels -q 2>/dev/null | head -3; then
    success "conda-forge 镜像源工作正常！"
else
    warn "conda-forge 测试未通过，请检查网络或稍后重试"
fi

# ─── 完成 ─────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   ✅ 镜像配置完成！${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  现在可以重新运行安装脚本："
echo -e "  ${CYAN}bash spatial_gnn.sh${NC}"
echo ""
echo -e "  如需恢复官方源："
echo -e "  ${CYAN}bash conda_mirror_setup.sh --reset${NC}"
echo ""
echo -e "  .condarc 备份位置:"
echo -e "  ${CYAN}${BACKUP:-（无旧配置）}${NC}"
echo ""

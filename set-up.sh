#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — 容器环境一键初始化
#
# 功能：系统依赖 → SSH Key → Git 克隆 → conda 镜像 → 完成
# 用法：bash bootstrap.sh [选项]
#   --skip-ssh     跳过 SSH key 生成（已有 key 时使用）
#   --skip-clone   跳过 git clone（已有仓库时使用）
#   --skip-mirror  跳过 conda 镜像配置
#   --email EMAIL  指定 git / SSH key 邮箱（默认交互式询问）
#   --name  NAME   指定 git 用户名（默认交互式询问）
#   --repo  URL    指定仓库 SSH URL（默认交互式询问）
#
# 示例（全自动无交互）：
#   bash bootstrap.sh \
#     --email 3089748482@qq.com \
#     --name bocchi \
#     --repo git@github.com:yeteye/AdaDiss.git
# =============================================================================

set -euo pipefail

# ── 颜色 ──────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}${BLUE}━━━━  $*  ━━━━${NC}"; }

# ── 参数解析 ──────────────────────────────────────────────────────────────────
SKIP_SSH=false
SKIP_CLONE=false
SKIP_MIRROR=false
GIT_EMAIL=""
GIT_NAME=""
REPO_URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-ssh)    SKIP_SSH=true ;;
        --skip-clone)  SKIP_CLONE=true ;;
        --skip-mirror) SKIP_MIRROR=true ;;
        --email)       GIT_EMAIL="$2"; shift ;;
        --name)        GIT_NAME="$2";  shift ;;
        --repo)        REPO_URL="$2";  shift ;;
        --help|-h)
            sed -n '3,20p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) error "未知参数: $1" ;;
    esac
    shift
done

# ── 收集缺少的必要参数（交互式） ──────────────────────────────────────────────
ask() {
    local var="$1" prompt="$2" default="${3:-}"
    if [[ -z "${!var}" ]]; then
        if [[ -n "$default" ]]; then
            read -rp "$(echo -e "${CYAN}?${NC} ${prompt} [${default}]: ")" val
            eval "$var=\"${val:-$default}\""
        else
            read -rp "$(echo -e "${CYAN}?${NC} ${prompt}: ")" val
            [[ -z "$val" ]] && error "$var 不能为空"
            eval "$var=\"$val\""
        fi
    fi
}

echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}   AdaDiss 容器环境初始化脚本${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

ask GIT_NAME  "git 用户名" "bocchi"
ask GIT_EMAIL "git 邮箱"   "3089748482@qq.com"
if [[ "$SKIP_CLONE" = false ]]; then
    ask REPO_URL "仓库 SSH URL" "git@github.com:yeteye/AdaDiss.git"
fi

echo ""
info "配置确认："
info "  用户名  : $GIT_NAME"
info "  邮箱    : $GIT_EMAIL"
[[ "$SKIP_CLONE" = false ]] && info "  仓库    : $REPO_URL"

# =============================================================================
# Step 1: 系统依赖
# =============================================================================
step "Step 1: 系统依赖安装"

export DEBIAN_FRONTEND=noninteractive

info "更新软件包列表..."
apt-get update -qq

info "安装基础工具（git / aria2 / build-essential）..."
apt-get install -y --no-install-recommends \
    git \
    aria2 \
    curl \
    wget \
    build-essential \
    cmake \
    ca-certificates \
    gnupg \
    2>/dev/null
success "基础工具安装完成"

info "安装 R 编译依赖..."
apt-get install -y --no-install-recommends \
    software-properties-common \
    dirmngr \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libhdf5-dev \
    2>/dev/null
success "R 编译依赖安装完成"

info "安装 R base..."
# 先尝试从系统源安装（容器里通常够用）
if apt-get install -y --no-install-recommends r-base r-base-dev 2>/dev/null; then
    R_VER=$(R --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' | head -1)
    success "R ${R_VER} 安装完成"
else
    # 回退：添加 CRAN 官方源再安装
    warn "系统源安装 R 失败，尝试 CRAN 源..."
    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "jammy")
    echo "deb https://cloud.r-project.org/bin/linux/ubuntu ${UBUNTU_CODENAME}-cran40/" \
        > /etc/apt/sources.list.d/cran.list
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/cran.gpg 2>/dev/null
    apt-get update -qq
    apt-get install -y --no-install-recommends r-base r-base-dev 2>/dev/null \
        || warn "R 安装失败，请手动安装。继续后续步骤..."
fi

# =============================================================================
# Step 2: git 全局配置
# =============================================================================
step "Step 2: git 全局配置"

git config --global user.name  "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

# 通用性能优化
git config --global http.postBuffer    524288000
git config --global core.compression   9
git config --global http.version       HTTP/1.1

success "git 配置完成"
git config --global --list | grep -E "user\." | sed 's/^/  /'

# =============================================================================
# Step 3: SSH Key
# =============================================================================
step "Step 3: SSH Key 配置"

SSH_KEY="$HOME/.ssh/id_ed25519"
mkdir -p ~/.ssh && chmod 700 ~/.ssh

if [[ "$SKIP_SSH" = true ]]; then
    warn "--skip-ssh 已指定，跳过 SSH key 生成"

elif [[ -f "$SSH_KEY" ]]; then
    warn "SSH key 已存在: ${SSH_KEY}"
    warn "如需重新生成请先删除: rm ~/.ssh/id_ed25519*"

else
    info "生成 Ed25519 SSH key..."
    ssh-keygen -t ed25519 -C "$GIT_EMAIL" -f "$SSH_KEY" -N "" -q
    success "SSH key 生成完成"
fi

# 启动 ssh-agent 并添加 key
if [[ -f "$SSH_KEY" ]]; then
    # 检查 ssh-agent 是否已运行
    if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
        eval "$(ssh-agent -s)" > /dev/null
    fi
    ssh-add "$SSH_KEY" 2>/dev/null || true

    # 配置 SSH 走 443 端口（穿透容器防火墙）
    SSH_CONF="$HOME/.ssh/config"
    if ! grep -q "Host github.com" "$SSH_CONF" 2>/dev/null; then
        cat >> "$SSH_CONF" << 'SSHCONF'

# GitHub SSH via port 443 (bypass firewall)
Host github.com
    HostName      ssh.github.com
    Port          443
    User          git
    IdentityFile  ~/.ssh/id_ed25519
    ServerAliveInterval 60
    TCPKeepAlive        yes
SSHCONF
        chmod 600 "$SSH_CONF"
        success "SSH config 已配置（443 端口穿透）"
    fi

    echo ""
    echo -e "${BOLD}${YELLOW}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${YELLOW}│  请将以下公钥添加到 GitHub SSH Keys：           │${NC}"
    echo -e "${BOLD}${YELLOW}│  Settings → SSH and GPG keys → New SSH key      │${NC}"
    echo -e "${BOLD}${YELLOW}└─────────────────────────────────────────────────┘${NC}"
    echo ""
    cat "${SSH_KEY}.pub"
    echo ""

    # 等待用户确认已添加公钥
    read -rp "$(echo -e "${CYAN}?${NC} 已将公钥添加到 GitHub？按回车继续，Ctrl+C 退出: ")" _

    info "测试 GitHub SSH 连接..."
    if ssh -T -o StrictHostKeyChecking=accept-new \
           -o ConnectTimeout=15 \
           git@github.com 2>&1 | grep -q "successfully authenticated"; then
        success "GitHub SSH 连接成功！"
    else
        warn "SSH 连接测试未通过（可能是 key 尚未生效，继续后续步骤）"
        warn "可手动测试: ssh -T git@github.com"
    fi
fi

# =============================================================================
# Step 4: 克隆仓库
# =============================================================================
step "Step 4: 克隆 Git 仓库"

REPO_DIR=$(basename "$REPO_URL" .git)

if [[ "$SKIP_CLONE" = true ]]; then
    warn "--skip-clone 已指定，跳过克隆"

elif [[ -d "$REPO_DIR/.git" ]]; then
    warn "仓库目录已存在: ${REPO_DIR}，执行 git pull..."
    git -C "$REPO_DIR" pull || warn "git pull 失败，继续后续步骤"

else
    info "克隆仓库: $REPO_URL"
    if git clone "$REPO_URL"; then
        success "仓库克隆完成: ${REPO_DIR}/"
    else
        error "git clone 失败。请检查：\n  1. 公钥已添加到 GitHub\n  2. ssh -T git@github.com 可以连通"
    fi
fi

# =============================================================================
# Step 5: conda 镜像配置
# =============================================================================
step "Step 5: conda 镜像配置"

if [[ "$SKIP_MIRROR" = true ]]; then
    warn "--skip-mirror 已指定，跳过镜像配置"

elif ! command -v conda &>/dev/null; then
    warn "conda 未安装，跳过镜像配置"
    warn "安装 conda 后运行: bash ${REPO_DIR}/conda_mirror_setup.sh"

else
    MIRROR_SCRIPT=""
    for candidate in \
        "${REPO_DIR}/conda_mirror_setup.sh" \
        "./conda_mirror_setup.sh" \
        "$HOME/conda_mirror_setup.sh"; do
        if [[ -f "$candidate" ]]; then
            MIRROR_SCRIPT="$candidate"
            break
        fi
    done

    if [[ -n "$MIRROR_SCRIPT" ]]; then
        info "运行镜像配置脚本: $MIRROR_SCRIPT"
        bash "$MIRROR_SCRIPT"
    else
        # 内嵌最小化镜像配置（不依赖外部脚本）
        info "未找到 conda_mirror_setup.sh，使用内嵌镜像配置..."
        cat > "$HOME/.condarc" << 'CONDARC'
channels:
  - defaults
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch:     https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
show_channel_urls: true
channel_priority: flexible
remote_connect_timeout_secs: 10
remote_read_timeout_secs: 60
CONDARC
        # pip 镜像
        mkdir -p "$HOME/.config/pip"
        cat > "$HOME/.config/pip/pip.conf" << 'PIPCONF'
[global]
index-url       = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host    = pypi.tuna.tsinghua.edu.cn
timeout         = 60
PIPCONF
        conda clean --index-cache -y 2>/dev/null || true
        success "conda + pip 镜像配置完成（清华 TUNA）"
    fi
fi

# =============================================================================
# 完成
# =============================================================================
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   ✅  环境初始化完成！${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  下一步："
if [[ "$SKIP_CLONE" = false && -d "$REPO_DIR" ]]; then
    echo -e "  ${CYAN}cd ${REPO_DIR}${NC}"
fi
if command -v conda &>/dev/null; then
    echo -e "  ${CYAN}bash spatial_gnn.sh${NC}    # 安装 Python GNN 环境"
    echo -e "  ${CYAN}bash label_transfer.sh${NC} # 安装 R LabelTransfer 环境"
fi
echo ""
echo "  常用命令："
echo -e "  ${CYAN}ssh -T git@github.com${NC}              # 测试 GitHub 连接"
echo -e "  ${CYAN}git config --global --list${NC}         # 查看 git 配置"
echo -e "  ${CYAN}cat ~/.ssh/id_ed25519.pub${NC}          # 查看公钥"
echo ""
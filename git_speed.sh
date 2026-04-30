#!/usr/bin/env bash
# =============================================================================
# git_speed.sh — git 国内加速配置
# 解决：git push/pull/clone 卡顿，GitHub 连接慢
#
# 用法：bash git_speed.sh [选项]
#   --diagnose   仅诊断当前连接速度，不修改配置
#   --proxy HOST:PORT  设置 SOCKS5/HTTP 代理（如已有代理）
#   --mirror     配置 GitHub 镜像加速（无代理时使用）
#   --ssh443     SSH 走 443 端口（穿透防火墙）
#   --reset      清除所有 git 加速配置
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── 参数解析 ─────────────────────────────────────────────────────────────────
DIAGNOSE=false
PROXY=""
SET_MIRROR=false
SSH_443=false
RESET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --diagnose)      DIAGNOSE=true ;;
        --proxy)         PROXY="$2"; shift ;;
        --mirror)        SET_MIRROR=true ;;
        --ssh443)        SSH_443=true ;;
        --reset)         RESET=true ;;
        --help|-h)
            echo "用法: bash git_speed.sh [选项]"
            echo "  --diagnose        诊断当前网速（不修改任何配置）"
            echo "  --proxy HOST:PORT 使用 SOCKS5 代理，如 --proxy 127.0.0.1:7890"
            echo "  --mirror          配置 GitHub 镜像加速（无代理时推荐）"
            echo "  --ssh443          SSH 走 443 端口穿透防火墙"
            echo "  --reset           清除所有加速配置"
            exit 0 ;;
        *) error "未知参数: $1"; exit 1 ;;
    esac
    shift
done

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   git 国内加速配置脚本${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# ─── 重置配置 ─────────────────────────────────────────────────────────────────
if [ "$RESET" = true ]; then
    info "清除所有 git 加速配置..."
    git config --global --unset http.proxy        2>/dev/null || true
    git config --global --unset https.proxy       2>/dev/null || true
    git config --global --unset http.https://github.com.proxy  2>/dev/null || true
    git config --global --unset url."https://ghproxy.com/https://github.com".insteadOf 2>/dev/null || true
    git config --global --unset url."https://gitclone.com/github.com".insteadOf 2>/dev/null || true

    # 清除 SSH 443 配置
    SSH_CONF="$HOME/.ssh/config"
    if [ -f "$SSH_CONF" ] && grep -q "github.com" "$SSH_CONF"; then
        warn "SSH config 含 GitHub 条目，请手动检查: $SSH_CONF"
    fi

    success "已清除 git 代理配置"
    echo ""
    info "当前 git 全局配置（http/https 相关）："
    git config --global --list | grep -E "http|url" || echo "  (无相关配置)"
    exit 0
fi

# ─── Step 1：诊断当前连接 ─────────────────────────────────────────────────────
echo -e "${BLUE}[Step 1]${NC} 诊断当前 GitHub 连接速度...\n"

measure_speed() {
    local name="$1"; local url="$2"; local timeout=15
    local result
    result=$(curl -o /dev/null -s -w "%{speed_download}|%{time_total}|%{http_code}" \
             --connect-timeout "$timeout" --max-time "$timeout" \
             -L "$url" 2>/dev/null || echo "0|99|000")
    local speed=$(echo "$result" | cut -d'|' -f1)
    local time=$(echo  "$result" | cut -d'|' -f2)
    local code=$(echo  "$result" | cut -d'|' -f3)

    local speed_kb=$(echo "$speed / 1024" | bc 2>/dev/null || echo "0")

    if   [ "$code" = "000" ];    then echo -e "  ❌ ${name}: 连接失败/超时"
    elif [ "$speed_kb" -gt 500 ] 2>/dev/null; then echo -e "  ${GREEN}✅ ${name}: ${speed_kb} KB/s  (快)${NC}"
    elif [ "$speed_kb" -gt 50  ] 2>/dev/null; then echo -e "  ${YELLOW}⚠️  ${name}: ${speed_kb} KB/s  (慢)${NC}"
    else echo -e "  ${RED}❌ ${name}: ${speed_kb} KB/s  (极慢/不可用)${NC}"
    fi
}

# GitHub 直连
measure_speed "GitHub 直连 (HTTPS)" \
    "https://github.com/bnprks/BPCells/archive/refs/heads/main.zip"

# 常用镜像
measure_speed "ghproxy.com 镜像" \
    "https://ghproxy.com/https://github.com/bnprks/BPCells/archive/refs/heads/main.zip"

measure_speed "gitclone.com 镜像" \
    "https://gitclone.com/github.com/bnprks/BPCells/archive/refs/heads/main.zip"

measure_speed "mirror.ghproxy.com 镜像" \
    "https://mirror.ghproxy.com/https://github.com/bnprks/BPCells/archive/refs/heads/main.zip"

# SSH 连通性
echo ""
info "测试 SSH 连接 GitHub（端口 22）..."
if ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
       git@github.com 2>&1 | grep -q "successfully authenticated"; then
    success "  SSH 22 端口可用"
else
    warn "  SSH 22 端口不可用或认证失败（正常，需要先配置 SSH key）"
fi

info "测试 SSH 走 443 端口（穿透 22 端口封锁）..."
if ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
       -p 443 ssh.github.com 2>&1 | grep -q "successfully authenticated"; then
    success "  SSH 443 端口可用"
else
    warn "  SSH 443 端口不可用（或 SSH key 未配置）"
fi

if [ "$DIAGNOSE" = true ]; then
    echo ""
    info "--diagnose 模式，不修改配置。"
    echo -e "\n  根据测速结果选择方案："
    echo -e "  - 镜像速度快  → ${CYAN}bash git_speed.sh --mirror${NC}"
    echo -e "  - 有本地代理  → ${CYAN}bash git_speed.sh --proxy 127.0.0.1:7890${NC}"
    echo -e "  - SSH443可用  → ${CYAN}bash git_speed.sh --ssh443${NC}"
    exit 0
fi

# ─── Step 2：应用选择的方案 ───────────────────────────────────────────────────
echo -e "\n${BLUE}[Step 2]${NC} 应用加速配置...\n"

APPLIED=false

# ── 方案 A：代理（最彻底，适合有科学上网的情况）────────────────────────────
if [ -n "$PROXY" ]; then
    info "配置代理: ${PROXY}"

    # 自动判断协议
    if echo "$PROXY" | grep -q "^socks5\|^http"; then
        PROXY_URL="$PROXY"
    else
        PROXY_URL="socks5h://${PROXY}"   # 默认 SOCKS5，h=DNS 也走代理
    fi

    # 只对 GitHub 生效，不影响国内源
    git config --global http.https://github.com.proxy  "$PROXY_URL"
    git config --global https.https://github.com.proxy "$PROXY_URL"

    success "已配置代理: ${PROXY_URL}"
    info "代理仅对 github.com 生效，其他 URL 不受影响"
    APPLIED=true
fi

# ── 方案 B：GitHub 镜像（无代理时的最佳选择）────────────────────────────────
if [ "$SET_MIRROR" = true ]; then
    info "配置 GitHub HTTPS 镜像加速..."

    # 先测哪个镜像最快，选速度最好的
    BEST_MIRROR=""
    BEST_SPEED=0

    for mirror_url in \
        "https://ghproxy.com/https://github.com" \
        "https://mirror.ghproxy.com/https://github.com" \
        "https://gitclone.com/github.com"; do

        mirror_host=$(echo "$mirror_url" | cut -d'/' -f3)
        speed=$(curl -o /dev/null -s -w "%{speed_download}" \
                --connect-timeout 8 --max-time 10 \
                "${mirror_url}/bnprks/BPCells/archive/refs/heads/main.zip" 2>/dev/null || echo "0")
        speed_kb=$(echo "$speed / 1024" | bc 2>/dev/null || echo "0")
        info "  ${mirror_host}: ${speed_kb} KB/s"

        if [ "$speed_kb" -gt "$BEST_SPEED" ] 2>/dev/null; then
            BEST_SPEED=$speed_kb
            BEST_MIRROR=$mirror_url
        fi
    done

    if [ -z "$BEST_MIRROR" ] || [ "$BEST_SPEED" -lt 10 ] 2>/dev/null; then
        warn "所有镜像速度均低于 10 KB/s，使用 ghproxy.com 作为默认"
        BEST_MIRROR="https://ghproxy.com/https://github.com"
    fi

    success "选用镜像: ${BEST_MIRROR} (${BEST_SPEED} KB/s)"

    # 清除旧镜像配置
    git config --global --unset url."https://ghproxy.com/https://github.com".insteadOf     2>/dev/null || true
    git config --global --unset url."https://mirror.ghproxy.com/https://github.com".insteadOf 2>/dev/null || true
    git config --global --unset url."https://gitclone.com/github.com".insteadOf            2>/dev/null || true

    # 设置新镜像
    git config --global url."${BEST_MIRROR}".insteadOf "https://github.com"

    success "GitHub HTTPS 已通过镜像加速"
    warn "注意：镜像加速后 git push 不可用（只读镜像）"
    warn "push 时请临时禁用：git config --global --unset url.*.insteadOf"
    APPLIED=true
fi

# ── 方案 C：SSH 走 443 端口（穿透防火墙，push/pull 都可用）────────────────
if [ "$SSH_443" = true ]; then
    SSH_CONF="$HOME/.ssh/config"
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"

    # 备份
    [ -f "$SSH_CONF" ] && cp "$SSH_CONF" "${SSH_CONF}.bak.$(date +%Y%m%d_%H%M%S)"

    # 检查是否已有 GitHub 条目
    if grep -q "Host github.com" "$SSH_CONF" 2>/dev/null; then
        warn "SSH config 已有 github.com 条目，跳过写入"
        warn "请手动检查: ${SSH_CONF}"
    else
        cat >> "$SSH_CONF" << 'SSHCONF'

# ── GitHub SSH 走 443 端口（by git_speed.sh）────────────────────────────────
# 解决：22 端口被防火墙封锁，或 SSH 连接极慢
# 原理：GitHub 在 ssh.github.com:443 上同样监听 SSH 协议
Host github.com
    HostName      ssh.github.com   # 实际连接的主机（支持 443）
    Port          443              # 用 443 替代 22
    User          git
    IdentityFile  ~/.ssh/id_ed25519  # 改为你的实际 key 文件名
    ServerAliveInterval 60
    TCPKeepAlive        yes
SSHCONF
        chmod 600 "$SSH_CONF"
        success "SSH config 已写入: ${SSH_CONF}"
    fi

    info "测试 SSH 443 连接..."
    if ssh -T -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
           git@github.com 2>&1 | grep -q "successfully authenticated"; then
        success "SSH 443 通道验证成功！push/pull/clone 均可使用"
    else
        warn "SSH 验证失败（可能是 key 未配置或 443 仍被封）"
        info "请确认 ~/.ssh/id_ed25519.pub 已添加到 GitHub SSH keys"
        info "GitHub → Settings → SSH and GPG keys → New SSH key"
    fi
    APPLIED=true
fi

# ── 未指定任何方案时，显示交互式提示 ─────────────────────────────────────────
if [ "$APPLIED" = false ]; then
    echo ""
    warn "未指定加速方案，请根据你的情况选择："
    echo ""
    echo -e "  ${BLUE}方案 1：镜像加速（无代理，clone/pull 快，push 不受影响）${NC}"
    echo -e "  ${CYAN}bash git_speed.sh --mirror${NC}"
    echo ""
    echo -e "  ${BLUE}方案 2：本地代理（clash/v2ray 等，push/pull 都快）${NC}"
    echo -e "  ${CYAN}bash git_speed.sh --proxy 127.0.0.1:7890${NC}  # SOCKS5"
    echo -e "  ${CYAN}bash git_speed.sh --proxy 127.0.0.1:7890${NC}  # HTTP 代理同样写法"
    echo ""
    echo -e "  ${BLUE}方案 3：SSH 走 443 端口（彻底方案，push/pull/clone 都快）${NC}"
    echo -e "  ${CYAN}bash git_speed.sh --ssh443${NC}"
    echo ""
    echo -e "  ${BLUE}先诊断再决定：${NC}"
    echo -e "  ${CYAN}bash git_speed.sh --diagnose${NC}"
    exit 0
fi

# ─── Step 3：附加通用优化 ─────────────────────────────────────────────────────
echo -e "\n${BLUE}[Step 3]${NC} 应用通用 git 性能优化...\n"

# HTTP/2 流水线（减少握手次数）
git config --global http.version HTTP/1.1    # 部分代理不支持 HTTP/2，用 1.1 更稳
git config --global http.postBuffer 524288000  # POST buffer 500MB（大文件 push）
git config --global core.compression 9          # 最大压缩（节省传输量）
git config --global pack.windowMemory "256m"    # pack 内存
git config --global pack.packSizeLimit "2g"     # 单 pack 文件上限

success "通用性能优化已应用"

# ─── 最终状态汇报 ─────────────────────────────────────────────────────────────
echo ""
info "当前 git 全局加速配置："
echo "  ─────────────────────────────────────────────────"
git config --global --list | grep -E "http\.proxy|https\.proxy|url\.|http\.post|core\.comp|pack\." \
    | sed 's/^/  /' || echo "  (空)"
echo "  ─────────────────────────────────────────────────"

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   ✅ 配置完成${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  清除所有配置：${CYAN}bash git_speed.sh --reset${NC}"
echo ""

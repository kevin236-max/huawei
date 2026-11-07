#!/usr/bin/env bash
# Build the checked-in Pi source in a machine-local runtime directory.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI_RUNTIME_ROOT="${PI_RUNTIME_ROOT:-$PROJECT_ROOT/.runtime}"
HUAWEI_PYTHON="${HUAWEI_PYTHON:-/kling_test/gen4/tangjiafu/miniconda3/envs/huawei/bin/python}"
NODE_VERSION=24.22.2

case "${1:-}" in
    -h|--help)
        cat <<'HELP'
Usage: bash install.sh [--install-system-deps]

Installs Python requirements and builds packages/pi in .runtime/pi.
Uses an existing Node >=22.19 with environment proxy support, or downloads Node 24.
The Pi build downloads npm dependencies and current model catalogs.
Requires Linux and an existing Python environment with pip.
Missing system tools are installed with apt-get (using sudo when needed).

PI_RUNTIME_ROOT changes the runtime directory (default: PROJECT/.runtime).
HUAWEI_PYTHON changes the Python interpreter used for requirements.
PI_BWRAP selects an existing bubblewrap executable.
HTTP_PROXY/HTTPS_PROXY/NO_PROXY (or lowercase) configure downloads and builds.
Node environment proxy support is enabled automatically.
--install-system-deps is accepted for compatibility; it is now the default.
HELP
        exit 0
        ;;
    --install-system-deps) ;;
    "") ;;
    *) echo 'Unexpected argument; use --help.' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Unexpected argument; use --help.' >&2; exit 2; }
# Normalize casing before any network operation. Lowercase takes precedence,
# as it does in curl; export both spellings for the different package managers.
for proxy_name in http_proxy https_proxy no_proxy; do
    proxy_upper="${proxy_name^^}"
    if [[ -v "$proxy_name" ]]; then
        export "$proxy_upper=${!proxy_name}"
        export "$proxy_name"
    elif [[ -v "$proxy_upper" ]]; then
        export "$proxy_name=${!proxy_upper}"
    fi
done
export NODE_USE_ENV_PROXY=1
[[ "$(uname -s)" == Linux ]] || { echo 'This installer requires Linux.' >&2; exit 1; }
[[ -x "$HUAWEI_PYTHON" ]] || {
    echo "Python not found: $HUAWEI_PYTHON. Set HUAWEI_PYTHON to your environment's Python." >&2
    exit 1
}

MISSING_TOOLS=()
for tool in git curl tar xz; do
    command -v "$tool" >/dev/null || MISSING_TOOLS+=("$tool")
done
BWRAP="${PI_BWRAP:-$(command -v bwrap || true)}"
[[ -z "${PI_BWRAP:-}" || -x "$PI_BWRAP" ]] || {
    echo 'PI_BWRAP must name an executable bubblewrap binary.' >&2
    exit 1
}
CA_CERTIFICATES_AVAILABLE=false
if command -v dpkg-query >/dev/null &&
    [[ "$(dpkg-query -W -f='${Status}' ca-certificates 2>/dev/null)" == 'install ok installed' ]]; then
    CA_CERTIFICATES_AVAILABLE=true
fi
SECCOMP_AVAILABLE=false
if "$HUAWEI_PYTHON" -c 'import ctypes; ctypes.CDLL("libseccomp.so.2")' >/dev/null 2>&1; then
    SECCOMP_AVAILABLE=true
fi
if [[ ${#MISSING_TOOLS[@]} -gt 0 || ! -x "$BWRAP" ]] || ! "$SECCOMP_AVAILABLE" || ! "$CA_CERTIFICATES_AVAILABLE"; then
    command -v apt-get >/dev/null || {
        echo 'Install git curl tar xz-utils bubblewrap libseccomp2 ca-certificates with your package manager.' >&2
        exit 1
    }
    APT=(apt-get)
    if [[ "$(id -u)" != 0 ]]; then
        command -v sudo >/dev/null || { echo 'Installing system dependencies requires root or sudo.' >&2; exit 1; }
        APT=(sudo --preserve-env=http_proxy,https_proxy,no_proxy,HTTP_PROXY,HTTPS_PROXY,NO_PROXY apt-get)
    fi
    "${APT[@]}" update
    "${APT[@]}" install -y git curl tar xz-utils bubblewrap libseccomp2 ca-certificates
    BWRAP="${PI_BWRAP:-$(command -v bwrap || true)}"
fi
[[ -x "$BWRAP" ]] || { echo 'PI_BWRAP must name an executable bubblewrap binary.' >&2; exit 1; }
BWRAP="$(readlink -f -- "$BWRAP")"
"$BWRAP" --version
"$HUAWEI_PYTHON" -c 'import ctypes; ctypes.CDLL("libseccomp.so.2")'

PI_SOURCE="$PROJECT_ROOT/packages/pi"
[[ -f "$PI_SOURCE/package.json" ]] || { echo 'Missing packages/pi source checkout.' >&2; exit 1; }
export GIT_OPTIONAL_LOCKS=0
[[ -z "$(git -C "$PI_SOURCE" status --porcelain)" ]] || {
    echo 'packages/pi has uncommitted changes; install a clean source checkout.' >&2
    exit 1
}
if [[ "$PI_RUNTIME_ROOT" != /* ]]; then PI_RUNTIME_ROOT="$PROJECT_ROOT/$PI_RUNTIME_ROOT"; fi
mkdir -p -- "$PI_RUNTIME_ROOT"
PI_RUNTIME_ROOT="$(cd -- "$PI_RUNTIME_ROOT" && pwd -P)"
case "$PI_RUNTIME_ROOT/" in
    "$PI_SOURCE/"*) echo 'PI_RUNTIME_ROOT must be outside packages/pi.' >&2; exit 1 ;;
esac
[[ "$PI_RUNTIME_ROOT/pi" != "$PI_SOURCE" && "$PI_RUNTIME_ROOT" != / ]] || {
    echo 'Choose a dedicated runtime directory outside packages/pi.' >&2
    exit 1
}
INSTALL_TEMP="$(mktemp -d "$PI_RUNTIME_ROOT/.install.XXXXXX")"
trap 'rm -rf -- "$INSTALL_TEMP"' EXIT

NODE="$(command -v node || true)"
node_is_compatible() {
    [[ -x "$1" && -x "$(dirname -- "$1")/npm" ]] && "$1" -e '
        const [major, minor] = process.versions.node.split(".").map(Number);
        const versionOK = major > 22 || (major === 22 && minor >= 19);
        process.exit(versionOK && process.allowedNodeEnvironmentFlags.has("--use-env-proxy") ? 0 : 1);
    '
}
if ! node_is_compatible "$NODE"; then
    NODE="$PI_RUNTIME_ROOT/node/bin/node"
    if ! node_is_compatible "$NODE"; then
        case "$(uname -m)" in
            x86_64) NODE_ARCH=x64 ;;
            aarch64|arm64) NODE_ARCH=arm64 ;;
            *) echo 'Automatic Node installation supports Linux x86_64 and aarch64; install Node >=22.19 first.' >&2; exit 1 ;;
        esac
        NODE_ARCHIVE="node-v$NODE_VERSION-linux-$NODE_ARCH.tar.xz"
        curl --fail --location --proto '=https' --tlsv1.2 --connect-timeout 15 --max-time 600 \
            "https://nodejs.org/dist/v$NODE_VERSION/$NODE_ARCHIVE" -o "$INSTALL_TEMP/node.tar.xz"
        mkdir -- "$INSTALL_TEMP/node"
        tar -xJf "$INSTALL_TEMP/node.tar.xz" --strip-components=1 -C "$INSTALL_TEMP/node"
        rm -rf -- "$PI_RUNTIME_ROOT/node"
        mv -- "$INSTALL_TEMP/node" "$PI_RUNTIME_ROOT/node"
    fi
fi
node_is_compatible "$NODE" || { echo 'Node must be >=22.19, support --use-env-proxy, and include npm.' >&2; exit 1; }
NODE="$(readlink -f -- "$NODE")"
export PATH="$(dirname -- "$NODE"):$PATH"
command -v npm >/dev/null || { echo 'npm is required alongside Node.' >&2; exit 1; }
export npm_config_cache="$PI_RUNTIME_ROOT/cache/npm"
export XDG_CACHE_HOME="$PI_RUNTIME_ROOT/cache"
export PI_CODING_AGENT_DIR="$INSTALL_TEMP/pi-check"

"$HUAWEI_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt"
mkdir -- "$INSTALL_TEMP/pi"
git -C "$PI_SOURCE" archive HEAD | tar -x -C "$INSTALL_TEMP/pi"
(
    cd -- "$INSTALL_TEMP/pi"
    npm ci --ignore-scripts
    npm run build
)
"$NODE" "$INSTALL_TEMP/pi/packages/coding-agent/dist/bundle/cli.js" --version

# Keep the active build until its replacement has been built successfully.
rm -rf -- "$PI_RUNTIME_ROOT/pi.previous"
if [[ -e "$PI_RUNTIME_ROOT/pi" ]]; then mv -- "$PI_RUNTIME_ROOT/pi" "$PI_RUNTIME_ROOT/pi.previous"; fi
mv -- "$INSTALL_TEMP/pi" "$PI_RUNTIME_ROOT/pi"
"$NODE" - "$INSTALL_TEMP/runtime.json" "$NODE" "$PI_RUNTIME_ROOT/pi" "$BWRAP" <<'JS'
const fs = require('node:fs');
const [file, node, pi_root, bwrap] = process.argv.slice(2);
fs.writeFileSync(file, JSON.stringify({node, pi_root, bwrap}, null, 2) + '\n');
JS
mv -f -- "$INSTALL_TEMP/runtime.json" "$PI_RUNTIME_ROOT/runtime.json"
rm -rf -- "$PI_RUNTIME_ROOT/pi.previous"
printf 'Installed runtime: %s\n' "$PI_RUNTIME_ROOT/runtime.json"
printf 'Python: %s\n' "$HUAWEI_PYTHON"

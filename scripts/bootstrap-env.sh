#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_version="0.9.5"
bootstrap_dir="${repo_root}/.tools/uv"
uv_bin="${bootstrap_dir}/bin/uv"
python_bin="${PYTHON:-python3.10}"
export UV_CACHE_DIR="${repo_root}/.tools/uv-cache"
export UV_PYTHON_DOWNLOADS="never"

"${python_bin}" -c 'import platform, sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"; assert sys.platform == "linux" and platform.machine() == "x86_64", "M0 supports Linux x86_64 CPU only"'
if [[ ! -x "${uv_bin}" ]] || [[ "$("${uv_bin}" --version)" != "uv ${uv_version}" ]]; then
  "${python_bin}" -m pip install --disable-pip-version-check \
    --cache-dir "${repo_root}/.tools/pip-cache" \
    --index-url "https://pypi.org/simple" --only-binary=:all: \
    --target "${bootstrap_dir}" --upgrade "uv==${uv_version}"
fi

"${uv_bin}" sync --project "${repo_root}/training" --locked --group dev --python "${python_bin}"
"${repo_root}/training/.venv/bin/python" "${repo_root}/scripts/check-environment.py"

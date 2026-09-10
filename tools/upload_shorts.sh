#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_dir}"

PY="python3"
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi

PYTHONPATH=src "${PY}" -m streamslice.retry_youtube_uploads \
  --config config/local-render.yaml \
  --output-root ~/tiktoks/output \
  "$@"

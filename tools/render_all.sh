#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_dir}"

PY="python3"
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi

SECRETS_FILE="${HOME}/.config/streamslice/client_secrets.json"
TOKEN_FILE="${HOME}/.config/streamslice/youtube_token.json"

if [[ ! -f "${TOKEN_FILE}" && ! -f "${SECRETS_FILE}" && ! -f "client_secrets.json" ]]; then
  echo "================================================================="
  echo "ℹ️  Авторизация YouTube API:"
  echo "Для автопубликации скачайте OAuth 2.0 Client ID (Desktop App) из"
  echo "Google Cloud Console и сохраните в: ${SECRETS_FILE}"
  echo "(или в корень проекта: client_secrets.json)"
  echo "Затем выполните 1 раз: ${PY} -m streamslice.cli youtube-login"
  echo "================================================================="
fi

PYTHONPATH=src "${PY}" -m streamslice.cli \
  --config config/local-render.yaml render-queue

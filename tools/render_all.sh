#!/usr/bin/env bash
# Render the whole queue on this machine.
#
# Gemini runs on the host, not here: the queue consumer calls super_curator and
# clip_reviewer, both of which need the model. Instead of installing CLIProxyAPI
# locally, this opens an SSH tunnel to the host's proxy and points the config at
# it (see the proxy section of config/local-render.yaml).
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_dir}"

PY="python3"
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi

CONFIG="${STREAMSLICE_CONFIG:-config/local-render.yaml}"
ENV_FILE="${STREAMSLICE_ENV:-${HOME}/.config/streamslice/streamslice.env}"
PROXY_HOST="${STREAMSLICE_PROXY_HOST:-root@2.26.2.207}"
PROXY_LOCAL_PORT="${STREAMSLICE_PROXY_PORT:-18081}"
PROXY_REMOTE_PORT="${STREAMSLICE_PROXY_REMOTE_PORT:-8081}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

if [[ -z "${CLIPROXY_API_KEY:-}" ]]; then
  echo "CLIPROXY_API_KEY is not set and ${ENV_FILE} does not define it." >&2
  echo "Copy it from the host: ssh ${PROXY_HOST} 'grep CLIPROXY_API_KEY /etc/streamslice.env'" >&2
  exit 1
fi

proxy_healthy() {
  curl -fsS -o /dev/null --max-time 5 \
    -H "Authorization: Bearer ${CLIPROXY_API_KEY}" \
    "http://127.0.0.1:${PROXY_LOCAL_PORT}/v1/models"
}

if proxy_healthy; then
  echo "Proxy tunnel on :${PROXY_LOCAL_PORT} already up."
else
  echo "Opening SSH tunnel :${PROXY_LOCAL_PORT} -> ${PROXY_HOST}:${PROXY_REMOTE_PORT}..."
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -f -N \
    -L "${PROXY_LOCAL_PORT}:127.0.0.1:${PROXY_REMOTE_PORT}" "${PROXY_HOST}"
  for _ in $(seq 10); do
    sleep 1
    if proxy_healthy; then break; fi
  done
  if ! proxy_healthy; then
    echo "Tunnel is up but the proxy did not answer. Check CLIProxyAPI on ${PROXY_HOST}:" >&2
    echo "  ssh ${PROXY_HOST} 'systemctl status cliproxy'" >&2
    exit 1
  fi
  echo "Tunnel ready."
fi

SECRETS_FILE="${HOME}/.config/streamslice/client_secrets.json"
TOKEN_FILE="${HOME}/.config/streamslice/youtube_token.json"
if [[ ! -f "${TOKEN_FILE}" && ! -f "${SECRETS_FILE}" && ! -f "client_secrets.json" ]]; then
  echo "YouTube is not authorized. Run once: PYTHONPATH=src ${PY} -m streamslice.cli youtube-login"
fi

exec env PYTHONPATH=src "${PY}" -m streamslice.cli --config "${CONFIG}" render-queue

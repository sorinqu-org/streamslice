#!/usr/bin/env bash
set -euo pipefail

if [[ -e /opt/streamslice ]]; then
  echo "/opt/streamslice already exists; refusing to overwrite it" >&2
  exit 2
fi

install -d -m 0755 \
  /opt/streamslice \
  /opt/cliproxy \
  /var/lib/streamslice/output \
  /var/lib/streamslice/work \
  /var/lib/streamslice/render-queue \
  /var/log/cliproxy

tar -xzf /tmp/streamslice-deploy.tar.gz -C /opt/streamslice
install -m 0755 /tmp/cli-proxy-api /opt/cliproxy/cli-proxy-api
install -m 0600 /tmp/config.yaml /opt/cliproxy/config.yaml

if ! grep -Eq '^host:[[:space:]]*"127\.0\.0\.1"' /opt/cliproxy/config.yaml; then
  echo "CLIProxyAPI must bind to 127.0.0.1" >&2
  exit 3
fi

install -m 0644 \
  /opt/streamslice/deploy/cliproxy.service \
  /etc/systemd/system/cliproxy.service
install -d -m 0755 /etc/systemd/system/twitch-recorder.service.d
{
  echo "[Service]"
  echo "EnvironmentFile=/etc/streamslice.env"
  echo "ReadWritePaths=/var/lib/streamslice"
} > /etc/systemd/system/twitch-recorder.service.d/streamslice-env.conf

python3 -m venv /opt/streamslice/.venv
/opt/streamslice/.venv/bin/pip install \
  --disable-pip-version-check \
  -e /opt/streamslice
chmod +x /opt/streamslice/tools/process_chunk_for_queue.py

python3 - <<'PY'
import pathlib
import re

config = pathlib.Path("/opt/cliproxy/config.yaml").read_text(encoding="utf-8")
match = re.search(r'api-keys:\s*\n\s*-\s*["\']?([^"\'\s]+)', config)
if not match:
    raise SystemExit("No API key found in CLIProxyAPI config")
pathlib.Path("/etc/streamslice.env").write_text(
    f"CLIPROXY_API_KEY={match.group(1)}\n",
    encoding="utf-8",
)
PY
chmod 0600 /etc/streamslice.env

systemctl daemon-reload
/opt/streamslice/.venv/bin/python -c \
  'import streamslice; print("StreamSlice:", streamslice.__file__)'
sha256sum /opt/cliproxy/cli-proxy-api
echo "DEPLOY_OK"

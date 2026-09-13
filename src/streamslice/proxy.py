from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .process import require_binary

LOGGER = logging.getLogger(__name__)


class ProxyError(RuntimeError):
    pass


def api_key(config: dict[str, Any]) -> str:
    env_name = config["proxy"]["api_key_env"]
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ProxyError(f"Set {env_name} before running StreamSlice")
    return value


def list_models(config: dict[str, Any], timeout: float = 3) -> list[str]:
    url = config["proxy"]["base_url"].rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key(config)}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ProxyError(f"Proxy returned HTTP {response.status}")
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ProxyError(f"Proxy health check failed: {exc}") from exc
    return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]


def start_proxy(config: dict[str, Any]) -> None:
    proxy = config["proxy"]
    if not proxy.get("binary"):
        # The proxy runs on another machine (see proxy.base_url), typically
        # reached over an SSH tunnel. Trying to launch a local binary here would
        # fail with a misleading "executable not found" instead of telling the
        # operator that the tunnel is down.
        raise ProxyError(
            f"CLIProxyAPI at {proxy.get('base_url')} is unreachable and no local "
            "binary is configured (proxy.binary is empty). If it runs on another "
            "host, check the SSH tunnel."
        )
    binary = require_binary(proxy["binary"])
    config_file = Path(proxy["config"]).expanduser().resolve()
    work_dir = Path(proxy["working_dir"]).expanduser().resolve()
    log_path = Path(proxy["log_file"]).expanduser().resolve()
    if not config_file.is_file() or not work_dir.is_dir():
        raise ProxyError("CLIProxyAPI binary directory or config file is missing")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    LOGGER.info("Starting CLIProxyAPI")
    subprocess.Popen(
        [binary, "-config", str(config_file)],
        cwd=work_dir,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def configured_proxy_models(config: dict[str, Any]) -> set[str]:
    """Names of models routed through CLIProxyAPI (the default provider).

    Models routed to other providers (e.g. openrouter) are validated by their
    own provider and must not be checked against the local proxy catalog.
    """
    names: set[str] = set()
    for spec in (config.get("models") or {}).values():
        if isinstance(spec, dict):
            provider = str(spec.get("provider") or "cliproxy").strip() or "cliproxy"
            if provider != "cliproxy":
                continue
            name = str(spec.get("model") or "").strip()
        else:
            name = str(spec).strip()
        if name:
            names.add(name)
    return names


def ensure_proxy(config: dict[str, Any]) -> list[str]:
    try:
        models = list_models(config)
    except ProxyError:
        start_proxy(config)
        deadline = time.monotonic() + float(config["proxy"]["startup_timeout_seconds"])
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            time.sleep(1)
            try:
                models = list_models(config)
                break
            except ProxyError as exc:
                last_error = exc
        else:
            raise ProxyError(f"CLIProxyAPI did not become healthy: {last_error}")

    configured = configured_proxy_models(config)
    missing = configured.difference(models)
    if missing:
        raise ProxyError(
            f"Configured models are unavailable: {sorted(missing)}. Available: {sorted(models)}"
        )
    return models


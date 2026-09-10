from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class ProcessError(RuntimeError):
    pass


def require_binary(name_or_path: str) -> str:
    value = str(Path(name_or_path).expanduser()) if "/" in name_or_path else name_or_path
    resolved = value if "/" in value and Path(value).is_file() else shutil.which(value)
    if not resolved:
        raise ProcessError(f"Required executable not found: {name_or_path}")
    if not os.access(resolved, os.X_OK):
        raise ProcessError(f"File is not executable: {resolved}")
    return str(Path(resolved).resolve())


def run(
    args: Iterable[str | Path],
    *,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(item) for item in args]
    LOGGER.debug("Running: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise ProcessError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{detail[-4000:]}"
        )
    return result


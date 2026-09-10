from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Config does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML value must be a mapping")
    parent_name = data.pop("extends", None)
    if parent_name:
        parent_path = (config_path.parent / parent_name).resolve()
        data = _deep_merge(load_config(parent_path), data)
    _validate(data)
    data["_config_path"] = str(config_path)
    data["_project_root"] = str(config_path.parent.parent.resolve())
    return data


def _validate(config: dict[str, Any]) -> None:
    required = (
        "proxy",
        "models",
        "sync",
        "transcription",
        "audio_analysis",
        "selection",
        "layout",
        "subtitles",
        "render",
        "output",
        "runtime",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")
    selection = config["selection"]
    if selection["min_duration_seconds"] < 20:
        raise ConfigError("selection.min_duration_seconds must be at least 20")
    if selection["max_duration_seconds"] > 90:
        raise ConfigError("selection.max_duration_seconds must be at most 90")
    if selection["final_count"] < 1 or selection["final_count"] > 10:
        raise ConfigError("selection.final_count must be between 1 and 10")
    render = config["render"]
    if (render["width"], render["height"], render["fps"]) != (1080, 1920, 60):
        raise ConfigError("render must be 1080x1920 at 60 FPS")


def project_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()

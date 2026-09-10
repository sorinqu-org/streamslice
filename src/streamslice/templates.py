"""Frame layout templates: loading, validation and lookup for the FFmpeg renderer."""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"[a-z0-9-]+")
_HEX_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")
_SOURCES = ("webcam", "gameplay", "full")
_FITS = ("cover", "contain", "blur_fill")
_PRIMARY_LAYOUT_NAMES = ("split", "webcam_full", "gameplay_full")
_HEIGHT_SUM_TOLERANCE = 1e-3

_WARNING_PREFIX = "Warning: "


class TemplateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Band:
    """One horizontal band of a vertical frame."""

    source: str
    height: float
    fit: str = "cover"
    align_x: float = 0.5
    align_y: float = 0.5
    zoom: float = 1.0
    track: bool = False
    max_upscale: float = 2.0
    divider_px: int = 0
    divider_color: str = "#000000"


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    name: str
    bands: tuple[Band, ...]


@dataclass(frozen=True, slots=True)
class Template:
    name: str
    description: str = ""
    width: int = 1080
    height: int = 1920
    fps: int = 60
    layouts: Mapping[str, LayoutSpec] = field(default_factory=dict)
    subtitles: Mapping[str, Any] = field(default_factory=dict)
    motion: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def layout(self, name: str) -> LayoutSpec:
        """Return the requested layout, falling back to "split", then the first one."""
        if name in self.layouts:
            return self.layouts[name]
        if "split" in self.layouts:
            return self.layouts["split"]
        return next(iter(self.layouts.values()))

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the same mapping shape accepted by validate_template_mapping."""
        return {
            "name": self.name,
            "description": self.description,
            "canvas": {"width": self.width, "height": self.height, "fps": self.fps},
            "layouts": {
                layout_name: {
                    "bands": [
                        {
                            "source": band.source,
                            "height": band.height,
                            "fit": band.fit,
                            "align_x": band.align_x,
                            "align_y": band.align_y,
                            "zoom": band.zoom,
                            "track": band.track,
                            "max_upscale": band.max_upscale,
                            "divider_px": band.divider_px,
                            "divider_color": band.divider_color,
                        }
                        for band in layout_spec.bands
                    ]
                }
                for layout_name, layout_spec in self.layouts.items()
            },
            "subtitles": dict(self.subtitles),
            "motion": dict(self.motion),
        }


def builtin_templates_dir() -> Path:
    """Return the repository's bundled `templates/` directory."""
    return Path(__file__).resolve().parents[2] / "templates"


def template_search_paths(config: dict[str, Any]) -> list[Path]:
    """Directories searched for templates, in priority order (first match wins)."""
    render_cfg = config.get("render") or {}
    paths: list[Path] = []
    templates_dir = render_cfg.get("templates_dir")
    if templates_dir:
        paths.append(Path(templates_dir).expanduser().resolve())
    paths.append(Path.home() / ".config" / "streamslice" / "templates")
    paths.append(builtin_templates_dir())
    return paths


def default_template_name(config: dict[str, Any]) -> str:
    render_cfg = config.get("render") or {}
    return str(render_cfg.get("template", "classic-split"))


def _discover_template_files(config: dict[str, Any]) -> dict[str, Path]:
    """Map template name (file stem) -> path, resolved by search path priority."""
    resolved: dict[str, Path] = {}
    for directory in template_search_paths(config):
        if not directory.is_dir():
            continue
        for file_path in sorted(directory.glob("*.yaml")):
            resolved.setdefault(file_path.stem, file_path)
    return resolved


def list_templates(config: dict[str, Any]) -> list[Template]:
    """Load every template visible from the configured search paths, sorted by name."""
    templates = [_load_template_file(path) for path in _discover_template_files(config).values()]
    return sorted(templates, key=lambda template: template.name)


def load_template(name: str, config: dict[str, Any]) -> Template:
    for directory in template_search_paths(config):
        candidate = directory / f"{name}.yaml"
        if candidate.is_file():
            return _load_template_file(candidate)
    available = sorted(_discover_template_files(config))
    names = ", ".join(available) if available else "none"
    raise TemplateError(f"Template '{name}' not found. Available templates: {names}")


def _load_template_file(path: Path) -> Template:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise TemplateError(f"Failed to parse template YAML '{path}': {exc}") from exc

    messages = validate_template_mapping(raw)
    errors = [message for message in messages if not message.startswith(_WARNING_PREFIX)]
    warnings = [message for message in messages if message.startswith(_WARNING_PREFIX)]
    for warning in warnings:
        LOGGER.warning("%s: %s", path, warning)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise TemplateError(f"Invalid template '{path}':\n{details}")

    return _build_template(raw, source_path=path)


def _build_template(data: dict[str, Any], source_path: Path | None) -> Template:
    canvas = data.get("canvas") or {}
    layouts_raw = data.get("layouts") or {}
    layouts = {
        layout_name: LayoutSpec(
            name=layout_name,
            bands=tuple(_build_band(band) for band in (layout_value.get("bands") or [])),
        )
        for layout_name, layout_value in layouts_raw.items()
    }
    return Template(
        name=str(data.get("name", "")),
        description=str(data.get("description", "")),
        width=int(canvas.get("width", 1080)),
        height=int(canvas.get("height", 1920)),
        fps=int(canvas.get("fps", 60)),
        layouts=MappingProxyType(layouts),
        subtitles=MappingProxyType(dict(data.get("subtitles") or {})),
        motion=MappingProxyType(dict(data.get("motion") or {})),
        source_path=source_path,
    )


def _build_band(raw: dict[str, Any]) -> Band:
    return Band(
        source=str(raw["source"]),
        height=float(raw["height"]),
        fit=str(raw.get("fit", "cover")),
        align_x=float(raw.get("align_x", 0.5)),
        align_y=float(raw.get("align_y", 0.5)),
        zoom=float(raw.get("zoom", 1.0)),
        track=bool(raw.get("track", False)),
        max_upscale=float(raw.get("max_upscale", 2.0)),
        divider_px=int(raw.get("divider_px", 0)),
        divider_color=str(raw.get("divider_color", "#000000")),
    )


def validate_template_mapping(data: Any) -> list[str]:
    """Validate a raw template mapping (as parsed from YAML).

    Returns a list of human-readable messages. Messages that represent mere
    forward-compatibility warnings (unknown keys) are prefixed with
    "Warning: "; every other message is a hard error. An empty list, or a
    list containing only "Warning: " entries, means the template is valid.
    """
    if not isinstance(data, dict):
        return ["Top-level template value must be a mapping"]

    messages: list[str] = []
    known_root_keys = {"name", "description", "canvas", "layouts", "subtitles", "motion"}
    for key in data:
        if key not in known_root_keys:
            messages.append(f"{_WARNING_PREFIX}unknown top-level key '{key}'")

    name = data.get("name")
    if not isinstance(name, str) or not name:
        messages.append("Template 'name' is required and must be a non-empty string")
    elif not _NAME_PATTERN.fullmatch(name):
        messages.append(f"Template name '{name}' must match pattern [a-z0-9-]+")

    messages.extend(_validate_canvas(data.get("canvas")))

    layouts = data.get("layouts")
    if not isinstance(layouts, dict) or not layouts:
        messages.append("'layouts' is required and must be a non-empty mapping")
    else:
        for layout_name, layout_value in layouts.items():
            messages.extend(_validate_layout(layout_name, layout_value))
        if not any(layout_name in layouts for layout_name in _PRIMARY_LAYOUT_NAMES):
            messages.append(
                "layouts must include at least one of: "
                + ", ".join(_PRIMARY_LAYOUT_NAMES)
            )

    for section in ("subtitles", "motion"):
        value = data.get(section)
        if value is not None and not isinstance(value, dict):
            messages.append(f"'{section}' must be a mapping")

    return messages


def _validate_canvas(canvas: Any) -> list[str]:
    if canvas is None:
        canvas = {}
    if not isinstance(canvas, dict):
        return ["'canvas' must be a mapping"]

    messages: list[str] = []
    known_canvas_keys = {"width", "height", "fps"}
    for key in canvas:
        if key not in known_canvas_keys:
            messages.append(f"{_WARNING_PREFIX}unknown canvas key '{key}'")

    width = canvas.get("width", 1080)
    height = canvas.get("height", 1920)
    fps = canvas.get("fps", 60)
    for field_name, value in (("width", width), ("height", height), ("fps", fps)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            messages.append(f"canvas.{field_name} must be a positive integer")

    if all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (width, height)):
        ratio = width / height
        if abs(ratio - 9.0 / 16.0) > 1e-2:
            messages.append(
                f"{_WARNING_PREFIX}canvas aspect ratio {width}x{height} is not 9:16"
            )
    return messages


def _validate_layout(layout_name: str, layout_value: Any) -> list[str]:
    if not isinstance(layout_value, dict):
        return [f"Layout '{layout_name}' must be a mapping"]

    messages: list[str] = []
    for key in layout_value:
        if key != "bands":
            messages.append(f"{_WARNING_PREFIX}unknown key '{key}' in layout '{layout_name}'")

    bands = layout_value.get("bands")
    if not isinstance(bands, list) or not bands:
        messages.append(f"Layout '{layout_name}' must have a non-empty 'bands' list")
        return messages

    height_sum = 0.0
    for index, band in enumerate(bands):
        band_messages, band_height = _validate_band(layout_name, index, band)
        messages.extend(band_messages)
        height_sum += band_height

    if abs(height_sum - 1.0) > _HEIGHT_SUM_TOLERANCE:
        messages.append(
            f"Layout '{layout_name}' band heights must sum to 1.0, got {height_sum:.4f}"
        )
    return messages


def _validate_band(layout_name: str, index: int, band: Any) -> tuple[list[str], float]:
    label = f"Layout '{layout_name}' band[{index}]"
    if not isinstance(band, dict):
        return [f"{label} must be a mapping"], 0.0

    messages: list[str] = []
    known_band_keys = {
        "source",
        "height",
        "fit",
        "align_x",
        "align_y",
        "zoom",
        "track",
        "max_upscale",
        "divider_px",
        "divider_color",
    }
    for key in band:
        if key not in known_band_keys:
            messages.append(f"{_WARNING_PREFIX}unknown key '{key}' in {label}")

    source = band.get("source")
    if source not in _SOURCES:
        messages.append(f"{label} source must be one of {_SOURCES}, got {source!r}")

    height_value = band.get("height")
    height_num = 0.0
    if not isinstance(height_value, (int, float)) or isinstance(height_value, bool):
        messages.append(f"{label} height must be a number")
    else:
        height_num = float(height_value)

    fit = band.get("fit", "cover")
    if fit not in _FITS:
        messages.append(f"{label} fit must be one of {_FITS}, got {fit!r}")

    for field_name, default in (("align_x", 0.5), ("align_y", 0.5)):
        value = band.get(field_name, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not (0.0 <= float(value) <= 1.0)
        ):
            messages.append(f"{label} {field_name} must be within [0, 1]")

    zoom = band.get("zoom", 1.0)
    if not isinstance(zoom, (int, float)) or isinstance(zoom, bool) or float(zoom) < 1.0:
        messages.append(f"{label} zoom must be >= 1.0")

    max_upscale = band.get("max_upscale", 2.0)
    if (
        not isinstance(max_upscale, (int, float))
        or isinstance(max_upscale, bool)
        or float(max_upscale) < 1.0
    ):
        messages.append(f"{label} max_upscale must be >= 1.0")

    divider_px = band.get("divider_px", 0)
    if not isinstance(divider_px, int) or isinstance(divider_px, bool) or divider_px < 0:
        messages.append(f"{label} divider_px must be a non-negative integer")

    divider_color = band.get("divider_color", "#000000")
    if not isinstance(divider_color, str) or not _HEX_COLOR_PATTERN.fullmatch(divider_color):
        messages.append(f"{label} divider_color must be a #RRGGBB hex string")

    return messages, height_num

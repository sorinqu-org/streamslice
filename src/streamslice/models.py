from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AudioPeak:
    time: float
    score: float
    rms: float
    peak: float
    crest: float
    onset: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Candidate:
    start_time: float
    end_time: float
    highlight_reason: str
    emotion_score: float
    camera_layout_recommendation: dict[str, Any] = field(default_factory=dict)
    context_score: float = 0.0
    context_summary: str = ""
    self_contained: bool = False
    quality_score: float = 0.0
    quality_summary: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> Candidate:
        recommendation = item.get("camera_layout_recommendation", {})
        if isinstance(recommendation, str):
            recommendation = {"layout": recommendation}
        return cls(
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            highlight_reason=str(item.get("highlight_reason", "")).strip(),
            emotion_score=max(0.0, min(10.0, float(item.get("emotion_score", 0)))),
            camera_layout_recommendation=dict(recommendation),
            context_score=max(0.0, min(10.0, float(item.get("context_score", 0)))),
            context_summary=str(item.get("context_summary", "")).strip(),
            self_contained=bool(item.get("self_contained", False)),
            quality_score=max(0.0, min(10.0, float(item.get("quality_score", 0)))),
            quality_summary=str(item.get("quality_summary", "")).strip(),
        )

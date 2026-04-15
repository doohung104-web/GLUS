from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FrameScores:
    fused_scores: List[float]
    text_scores: Optional[List[float]] = None
    motion_scores: Optional[List[float]] = None


@dataclass
class SamplePlan:
    context_indices: List[int]
    query_indices: List[int]
    scores: Optional[FrameScores] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseScoreBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def score(
        self,
        video_frames: List[str],
        expressions: List[str],
        video_meta: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
    ) -> FrameScores:
        raise NotImplementedError


class BaseContextSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def sample(
        self,
        scores: FrameScores,
        video_length: int,
        context_frame_num: int,
        is_train: bool = True,
    ) -> List[int]:
        raise NotImplementedError


class BaseQuerySampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def sample(
        self,
        scores: FrameScores,
        video_length: int,
        question_frame_num: int,
        is_train: bool = True,
    ) -> List[int]:
        raise NotImplementedError

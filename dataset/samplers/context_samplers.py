from __future__ import annotations

import random

import numpy as np

from .base import FrameScores
from .registry import register_context_sampler


def _split_segments(video_length: int, num_segments: int):
    boundaries = np.linspace(0, video_length, num_segments + 1, dtype=int)
    segments = []
    for i in range(num_segments):
        left, right = int(boundaries[i]), int(boundaries[i + 1])
        if right <= left:
            right = min(video_length, left + 1)
        segments.append((left, right))
    return segments


@register_context_sampler("uniform_random")
class UniformRandomContextSampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, scores: FrameScores, video_length: int, context_frame_num: int, is_train: bool = True):
        segments = _split_segments(video_length, context_frame_num)
        indices = []
        for left, right in segments:
            if right - left <= 1:
                indices.append(left)
            else:
                indices.append(random.randint(left, right - 1))
        return indices


@register_context_sampler("uniform_center")
class UniformCenterContextSampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, scores: FrameScores, video_length: int, context_frame_num: int, is_train: bool = True):
        segments = _split_segments(video_length, context_frame_num)
        return [min(video_length - 1, (left + right - 1) // 2) for left, right in segments]


@register_context_sampler("segment_top1")
class SegmentTop1ContextSampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, scores: FrameScores, video_length: int, context_frame_num: int, is_train: bool = True):
        fused = np.asarray(scores.fused_scores)
        segments = _split_segments(video_length, context_frame_num)
        indices = []
        for left, right in segments:
            local = fused[left:right]
            if len(local) == 0:
                indices.append(min(video_length - 1, left))
            else:
                indices.append(left + int(np.argmax(local)))
        return indices


@register_context_sampler("segment_topk_random")
class SegmentTopKRandomContextSampler:
    def __init__(self, topk: int = 3, **kwargs):
        self.topk = max(1, int(topk))

    def sample(self, scores: FrameScores, video_length: int, context_frame_num: int, is_train: bool = True):
        fused = np.asarray(scores.fused_scores)
        segments = _split_segments(video_length, context_frame_num)
        indices = []
        for left, right in segments:
            local = fused[left:right]
            if len(local) == 0:
                indices.append(min(video_length - 1, left))
                continue
            k = min(self.topk, len(local))
            top_local = np.argsort(-local)[:k]
            chosen = int(random.choice(top_local)) if is_train and k > 1 else int(top_local[0])
            indices.append(left + chosen)
        return indices

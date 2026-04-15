from __future__ import annotations

import random

import numpy as np

from .base import FrameScores
from .registry import register_query_sampler


@register_query_sampler("random_contiguous")
class RandomContiguousQuerySampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, scores: FrameScores, video_length: int, question_frame_num: int, is_train: bool = True):
        if video_length <= question_frame_num:
            return list(range(video_length))
        start = random.randint(0, video_length - question_frame_num)
        return list(range(start, start + question_frame_num))


@register_query_sampler("centered_anchor")
class CenteredAnchorQuerySampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, scores: FrameScores, video_length: int, question_frame_num: int, is_train: bool = True):
        if video_length <= question_frame_num:
            return list(range(video_length))
        fused = np.asarray(scores.fused_scores)
        anchor = int(np.argmax(fused))
        half = question_frame_num // 2
        start = anchor - half
        start = max(0, min(start, video_length - question_frame_num))
        return list(range(start, start + question_frame_num))


@register_query_sampler("top_window")
class TopWindowQuerySampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, scores: FrameScores, video_length: int, question_frame_num: int, is_train: bool = True):
        if video_length <= question_frame_num:
            return list(range(video_length))
        fused = np.asarray(scores.fused_scores, dtype=np.float32)
        best_start, best_score = 0, None
        window_sum = fused[:question_frame_num].sum()
        best_start, best_score = 0, float(window_sum)
        for start in range(1, video_length - question_frame_num + 1):
            window_sum += fused[start + question_frame_num - 1] - fused[start - 1]
            if float(window_sum) > best_score:
                best_score = float(window_sum)
                best_start = start
        return list(range(best_start, best_start + question_frame_num))

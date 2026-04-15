from __future__ import annotations

from .base import SamplePlan
from .registry import build_context_sampler, build_query_sampler, build_score_backend


class FrameSamplerPolicy:
    def __init__(self, score_backend, context_sampler, query_sampler):
        self.score_backend = score_backend
        self.context_sampler = context_sampler
        self.query_sampler = query_sampler

    def sample(
        self,
        video_frames,
        expressions,
        video_meta,
        context_frame_num,
        question_frame_num,
        is_train=True,
    ):
        scores = self.score_backend.score(
            video_frames=video_frames,
            expressions=expressions,
            video_meta=video_meta,
            is_train=is_train,
        )
        context_indices = self.context_sampler.sample(
            scores=scores,
            video_length=len(video_frames),
            context_frame_num=context_frame_num,
            is_train=is_train,
        )
        query_indices = self.query_sampler.sample(
            scores=scores,
            video_length=len(video_frames),
            question_frame_num=question_frame_num,
            is_train=is_train,
        )
        return SamplePlan(
            context_indices=context_indices,
            query_indices=query_indices,
            scores=scores,
            meta={
                "score_backend": self.score_backend.__class__.__name__,
                "context_sampler": self.context_sampler.__class__.__name__,
                "query_sampler": self.query_sampler.__class__.__name__,
            },
        )


def build_frame_sampler(
    score_backend="none",
    context_sampler="uniform_random",
    query_sampler="random_contiguous",
    **kwargs,
):
    return FrameSamplerPolicy(
        score_backend=build_score_backend(score_backend, **kwargs),
        context_sampler=build_context_sampler(context_sampler, **kwargs),
        query_sampler=build_query_sampler(query_sampler, **kwargs),
    )

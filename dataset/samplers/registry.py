from __future__ import annotations

from typing import Any, Dict


_SCORE_BACKENDS: Dict[str, type] = {}
_CONTEXT_SAMPLERS: Dict[str, type] = {}
_QUERY_SAMPLERS: Dict[str, type] = {}


def register_score_backend(name: str):
    def decorator(cls):
        _SCORE_BACKENDS[name] = cls
        return cls
    return decorator


def register_context_sampler(name: str):
    def decorator(cls):
        _CONTEXT_SAMPLERS[name] = cls
        return cls
    return decorator


def register_query_sampler(name: str):
    def decorator(cls):
        _QUERY_SAMPLERS[name] = cls
        return cls
    return decorator


def _build(name: str, registry: Dict[str, type], **kwargs: Any):
    if name not in registry:
        raise KeyError(f"Unknown plugin '{name}'. Available: {sorted(registry.keys())}")
    return registry[name](**kwargs)


def build_score_backend(name: str, **kwargs: Any):
    return _build(name, _SCORE_BACKENDS, **kwargs)


def build_context_sampler(name: str, **kwargs: Any):
    return _build(name, _CONTEXT_SAMPLERS, **kwargs)


def build_query_sampler(name: str, **kwargs: Any):
    return _build(name, _QUERY_SAMPLERS, **kwargs)

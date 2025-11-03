"""Compatibility helpers bridging Pydantic v1 and v2 APIs."""

from __future__ import annotations

from typing import Any, Dict


def model_dump(model: Any, **kwargs: Any) -> Dict[str, Any]:
    """Return model data as a dict across Pydantic versions."""
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)  # type: ignore[call-arg]
    return model.dict(**kwargs)  # type: ignore[attr-defined]


def model_copy(model: Any, **kwargs: Any):
    """Return a (potentially deep) copy across Pydantic versions."""
    copy_method = getattr(model, "model_copy", None)
    if copy_method:
        return copy_method(**kwargs)
    return model.copy(**kwargs)  # type: ignore[attr-defined]

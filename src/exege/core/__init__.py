"""Durable core: the errors, the adapter registry, the hint for a missing extra.

Depends on the standard library and nothing else, and knows nothing about the
subpackages built on it.
"""

from __future__ import annotations

from exege.core.errors import (
    AdapterError,
    ExegeError,
    MissingExtraError,
    RequestError,
)

__all__ = [
    "AdapterError",
    "MissingExtraError",
    "RequestError",
    "ExegeError",
]

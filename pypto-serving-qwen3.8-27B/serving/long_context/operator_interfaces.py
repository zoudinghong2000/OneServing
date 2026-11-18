"""Compile-time model extensions. No per-layer Python callbacks run on the NPU path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InlineOperators:
    """Definitions for a declared model boundary, in the model's symbol namespace.

    The serving compiler checks the root signatures against its model contract
    before linking. Operators do not supply entrypoints, scheduling or allocation.
    """

    source: str
    identity: str


class AttentionBackend(Protocol):
    def bind(self, role: str) -> InlineOperators | None: ...


class LegacyHeadPagePolicy(AttentionBackend, Protocol):
    def windowed(self, role: str, position: int) -> bool: ...
    def full_history_end(self, request, scheduler) -> int: ...
    def retire(self, head: int, block: int, position: int, page_tokens: int) -> bool: ...


class FFNBackend(Protocol):
    # enabled is a runtime policy flag; it does not choose a second full network.
    enabled: bool
    def bind(self, role: str) -> InlineOperators | None: ...


@dataclass(frozen=True)
class ModelExtensions:
    family: str
    roles: tuple[str, ...]
    attention: AttentionBackend
    ffn: FFNBackend

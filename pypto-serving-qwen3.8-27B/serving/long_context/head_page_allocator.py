"""Serving-owned physical KV layout and page lifecycle for four head classes."""
from __future__ import annotations

from dataclasses import dataclass
import os

from .operator_interfaces import LegacyHeadPagePolicy


@dataclass(frozen=True)
class KVLayout:
    context: int = 32768
    page_tokens: int = 128
    head_classes: int = 4
    full_layers: int = 16
    head_dim: int = 256
    physical_pages: int | None = None

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (
                self.context, self.page_tokens, self.head_classes,
                self.full_layers, self.head_dim)):
            raise ValueError("KV geometry must consist of positive integers")
        if self.context <= 0 or self.context % self.page_tokens:
            raise ValueError("Context must be a positive multiple of page_tokens")
        if self.physical_pages is None:
            object.__setattr__(self, "physical_pages", self.required_pages)
        if type(self.physical_pages) is not int or self.physical_pages <= 0:
            raise ValueError("Physical KV pages must be a positive integer")
        if self.physical_pages < self.required_pages:
            raise ValueError(f"KV pool needs at least {self.required_pages} pages for full-history fallback")

    @property
    def logical_pages(self):
        return self.context // self.page_tokens

    @property
    def required_pages(self):
        return 1 + self.head_classes * self.logical_pages

    @property
    def bytes_per_page(self):
        return 2 * self.full_layers * self.page_tokens * self.head_dim * 2

    @property
    def pool_shape(self):
        return (self.full_layers, 1, self.physical_pages, self.page_tokens, self.head_dim)

    @classmethod
    def from_environment(cls):
        pages, budget = os.environ.get("ONESERVING_KV_PAGES"), os.environ.get("ONESERVING_KV_BUDGET_MIB")
        if pages is not None and budget is not None:
            raise ValueError("Specify ONESERVING_KV_PAGES or ONESERVING_KV_BUDGET_MIB")
        base = cls()
        count = int(pages) if pages is not None else base.physical_pages
        if budget is not None:
            count = int(budget) * (1 << 20) // base.bytes_per_page
        return cls(physical_pages=count)


class HeadPageAllocator:
    def __init__(self, layout, policy: LegacyHeadPagePolicy):
        self.layout, self.policy = layout, policy
        self.request_id = None

    def reset(self, request_id, full_history_end):
        import torch
        self.request_id = request_id
        self.full_history_end = full_history_end
        self.table = torch.zeros((self.layout.head_classes, self.layout.logical_pages), dtype=torch.int32)
        self.mapping = {}
        self.free = list(range(self.layout.physical_pages - 1, 0, -1))
        self.retired = set()

    def prepare(self, slot, target, role):
        from .schedule import RequestCapacityError
        pos, page = slot.position, self.layout.page_tokens
        end = pos + len(slot.token_ids)
        if self.request_id != slot.request.request_id:
            raise RuntimeError("Head page owner differs from the active request")
        if not 0 <= pos < end <= self.layout.context:
            raise ValueError("KV write exceeds the serving context limit")
        windowed = self.policy.windowed(role, pos)
        if self.retired and not windowed:
            raise RuntimeError("Full-history consumer after local page retirement")
        if windowed and pos >= self.full_history_end:
            for head, block in list(self.mapping):
                if self.policy.retire(head, block, pos, page):
                    key = head, block
                    self.free.append(self.mapping.pop(key))
                    self.retired.add(key)
                    self.table[key] = 0
        needed = [(head, block)
                  for block in range(pos // page, (end + page - 1) // page)
                  for head in range(self.layout.head_classes)
                  if (head, block) not in self.mapping]
        # Capacity failure is transactional; do not leave a partially allocated batch.
        if len(needed) > len(self.free):
            raise RequestCapacityError(slot.request, "Insufficient physical head pages")
        for key in needed:
            self.mapping[key] = self.free.pop()
            self.table[key] = self.mapping[key]
        count = self.layout.head_classes * self.layout.logical_pages
        if target.numel() < count:
            raise ValueError("Page table ABI does not cover the serving KV layout")
        target[:count].copy_(self.table.reshape(-1))

"""Qwen3.8 BF16 32K, single-active-request profile. Share unchanged modules with optimized."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent.parent.parent/"optimized/kernels"))

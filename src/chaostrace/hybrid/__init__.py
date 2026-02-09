"""Hybrid components for ChaosTrace.

This package provides optional components that can be fused with the chaos-based score:
- Matrix Profile (optional dependency: stumpy)
- Causal drift proxy (numpy-only)
- Fusion utilities and evaluation metrics
"""

from .fusion import FusionOutput, fuse_scores, default_weights  # noqa: F401

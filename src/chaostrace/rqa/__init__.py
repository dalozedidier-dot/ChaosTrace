"""Recurrence Quantification Analysis (RQA) utilities.

This subpackage is intentionally lightweight by default.

Core implementation:
- NumPy/SciPy only.

Optional extras:
- `chaostrace[rqa]` can bring in pyrqa/pyunicorn for extended capabilities.

All functions degrade gracefully when optional dependencies are missing.
"""

from __future__ import annotations

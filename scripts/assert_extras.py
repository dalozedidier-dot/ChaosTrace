"""Assert that optional dependency extras exist in pyproject.toml.

Why this exists
- GitHub Actions workflows may request `pip install -e '.[dl]'` or similar.
- If the repo is in an inconsistent state (old pyproject, partial patch), pip only
  emits a warning and continues, which later fails far from the root cause.

Usage
  python scripts/assert_extras.py dl
  python scripts/assert_extras.py mp
  python scripts/assert_extras.py viz
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _load_pyproject(pyproject: Path) -> dict[str, Any]:
    if not pyproject.exists():
        raise FileNotFoundError(f"pyproject.toml not found at: {pyproject}")

    try:
        import tomllib  # py311+

        return tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to parse pyproject.toml via tomllib") from e


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python scripts/assert_extras.py <extra_name>")
        return 2

    want = argv[0].strip()
    if not want:
        print("ERROR: empty extra name")
        return 2

    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    doc = _load_pyproject(pyproject)

    project = doc.get("project") or {}
    opt = project.get("optional-dependencies") or {}

    if not isinstance(opt, dict):
        print("ERROR: [project.optional-dependencies] is missing or invalid")
        return 1

    extras = sorted(str(k) for k in opt.keys())
    if want not in opt:
        print(f"ERROR: extra '{want}' not found in pyproject.toml")
        print("Available extras:")
        for k in extras:
            print(f"  - {k}")
        return 1

    deps = opt.get(want)
    if not isinstance(deps, list) or not deps:
        print(f"ERROR: extra '{want}' exists but has no dependencies")
        return 1

    print(f"OK: extra '{want}' exists with {len(deps)} dependency entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

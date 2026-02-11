from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from chaostrace.utils.manifest import write_manifest


def _parse_kv(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid --param '{it}', expected key=value")
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid --param '{it}', empty key")
        # Best-effort JSON decode, else keep string
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit a manifest.json for an output directory.")
    ap.add_argument("--out-dir", required=True, help="Output directory to write manifest.json into")
    ap.add_argument(
        "--files",
        nargs="+",
        default=[],
        help="Relative file paths (within out-dir) to hash and list in the manifest",
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra params key=value. Value can be JSON (e.g. number, true, {...})",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = _parse_kv(list(args.param))

    # Auto-add useful CI context if present
    for k, envk in [
        ("git_sha", "GITHUB_SHA"),
        ("git_ref", "GITHUB_REF"),
        ("workflow", "GITHUB_WORKFLOW"),
        ("run_id", "GITHUB_RUN_ID"),
        ("run_number", "GITHUB_RUN_NUMBER"),
    ]:
        if k not in params and os.environ.get(envk):
            params[k] = os.environ[envk]

    write_manifest(out_dir, params=params, files=[Path(f) for f in args.files])


if __name__ == "__main__":
    main()

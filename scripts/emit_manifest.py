from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _github_ctx() -> dict[str, str]:
    keys = [
        "GITHUB_REPOSITORY",
        "GITHUB_SHA",
        "GITHUB_REF",
        "GITHUB_WORKFLOW",
        "GITHUB_JOB",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_NUMBER",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_ACTOR",
        "RUNNER_OS",
        "RUNNER_ARCH",
    ]
    out: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v is not None:
            out[k.lower()] = v
    fixed: dict[str, str] = {}
    for k, v in out.items():
        fixed[k] = v
    return fixed


def _parse_params(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--param expects key=value, got: {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit a stable CI manifest (sha256, sizes, context).")
    ap.add_argument("--out-dir", required=True, help="Directory to write manifest_ci.json into")
    ap.add_argument("--files", nargs="+", default=[], help="Files relative to out-dir (or absolute)")
    ap.add_argument("--param", action="append", default=[], help="Extra key=value params")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for rel in args.files:
        p = Path(rel)
        if not p.is_absolute():
            p = (out_dir / rel).resolve()
        exists = p.exists() and p.is_file()
        files.append(
            {
                "path": rel,
                "exists": bool(exists),
                "bytes": int(p.stat().st_size) if exists else 0,
                "sha256": _sha256(p) if exists else None,
            }
        )

    payload = {
        "schema_version": 1,
        "out_dir": str(out_dir),
        "github": _github_ctx(),
        "params": _parse_params(list(args.param)),
        "files": files,
    }

    (out_dir / "manifest_ci.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

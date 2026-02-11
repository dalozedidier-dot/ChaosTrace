from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _gh_meta() -> Dict[str, Any]:
    keys = [
        "GITHUB_REPOSITORY",
        "GITHUB_SHA",
        "GITHUB_REF",
        "GITHUB_WORKFLOW",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_NUMBER",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_JOB",
        "GITHUB_ACTOR",
        "RUNNER_OS",
        "RUNNER_ARCH",
    ]
    return {k.lower(): os.environ.get(k, "") for k in keys if os.environ.get(k) is not None}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit a stable manifest.json for CI artifacts.")
    ap.add_argument("--out-dir", required=True, help="Directory containing artifacts.")
    ap.add_argument(
        "--files",
        nargs="+",
        default=[],
        help="List of filenames (relative to out-dir) to record in the manifest.",
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra key=value pairs to include as manifest.params (repeatable).",
    )
    ap.add_argument(
        "--manifest-name",
        default="manifest_ci.json",
        help="Output manifest file name (written inside out-dir).",
    )
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, str] = {}
    for item in args.param:
        if "=" in item:
            k, v = item.split("=", 1)
            params[k.strip()] = v.strip()
        else:
            params[item.strip()] = ""

    records: List[Dict[str, Any]] = []
    for rel in args.files:
        rel = str(rel).strip()
        if not rel:
            continue
        p = out_dir / rel
        if p.exists() and p.is_file():
            records.append(
                {
                    "path": rel,
                    "exists": True,
                    "bytes": int(p.stat().st_size),
                    "sha256": _sha256_file(p),
                }
            )
        else:
            records.append({"path": rel, "exists": False, "bytes": 0, "sha256": ""})

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "out_dir": str(out_dir.as_posix()),
        "params": params,
        "github": _gh_meta(),
        "files": records,
    }

    (out_dir / str(args.manifest_name)).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

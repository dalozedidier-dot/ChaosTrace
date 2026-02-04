from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .hashing import sha256_file

def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj

def write_manifest(out_dir: str | Path, params: dict[str, Any], files: list[str | Path]) -> Path:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    file_hashes = []
    for f in files:
        fp = outp / f if not Path(f).is_absolute() else Path(f)
        if fp.exists() and fp.is_file():
            file_hashes.append({"file": str(fp.relative_to(outp)), "sha256": sha256_file(fp)})
        else:
            file_hashes.append({"file": str(f), "sha256": None})

    manifest = {
        "schema": "chaostrace_manifest_v1",
        "python": sys.version,
        "platform": {"system": platform.system(), "release": platform.release()},
        "params": {k: _jsonable(v) for k, v in params.items()},
        "files": file_hashes,
    }
    mpath = outp / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return mpath

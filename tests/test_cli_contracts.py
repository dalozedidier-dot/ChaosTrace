from __future__ import annotations

import json
def test_run_hybrid_cli_contract() -> None:
    from chaostrace.cli.run_hybrid import build_parser

    p = build_parser()
    opts = {a.dest for a in p._actions}  # noqa: SLF001

    # Workflows rely on these options.
    assert "pick" in opts
    assert "require_mp" in opts
    assert "require_dl" in opts
    assert "enable_mp" in opts


def test_rqa_multiscale_default_is_auto() -> None:
    from chaostrace.cli.rqa_multiscale import build_parser

    p = build_parser()
    ns = p.parse_args(["--input", "test_data/sample_timeseries.csv", "--out", "_tmp"])
    assert str(ns.scales).lower() == "auto"


def test_manifest_writer_contract(tmp_path) -> None:
    from chaostrace.utils.manifest import write_manifest

    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "a.txt").write_text("hello", encoding="utf-8")

    mpath = write_manifest(out, params={"tool": "test"}, files=["a.txt"])
    data = json.loads(mpath.read_text(encoding="utf-8"))

    assert data["schema"] == "chaostrace_manifest_v1"
    assert "python" in data
    assert data["params"]["tool"] == "test"
    assert data["files"][0]["file"] == "a.txt"
    assert isinstance(data["files"][0]["sha256"], str)

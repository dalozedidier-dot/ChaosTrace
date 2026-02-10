from __future__ import annotations


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

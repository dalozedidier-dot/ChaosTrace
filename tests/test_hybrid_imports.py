from __future__ import annotations

def test_import_hybrid_cli_without_optionals() -> None:
    # Importing CLI modules must not require optional extras (torch/stumpy).
    import chaostrace.cli.run_hybrid  # noqa: F401
    import chaostrace.cli.train_hybrid  # noqa: F401

def test_import_hybrid_modules_without_optionals() -> None:
    import chaostrace.hybrid.fusion  # noqa: F401
    import chaostrace.hybrid.causal_var  # noqa: F401
    import chaostrace.hybrid.metrics  # noqa: F401
    # matrix_profile module imports without stumpy, runtime error only when called.
    import chaostrace.hybrid.matrix_profile  # noqa: F401

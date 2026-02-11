This patch only changes artifact names to include the GitHub run id and attempt,
to avoid confusion between different runs when downloading logs/artifacts.

It updates:
- .github/workflows/ci.yml
- .github/workflows/hybrid_dl.yml

New naming pattern examples:
- ci_out_py3.12_run1234567890_a1
- hybrid_core_out_run1234567890_a1
- hybrid_dl_out_run1234567890_a1

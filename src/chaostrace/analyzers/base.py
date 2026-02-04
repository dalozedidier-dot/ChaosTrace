from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import pandas as pd

@dataclass(frozen=True)
class AnalyzerResult:
    name: str
    metrics: dict[str, float]
    timeline: pd.DataFrame

class Analyzer(Protocol):
    def run(self, df: pd.DataFrame) -> AnalyzerResult: ...

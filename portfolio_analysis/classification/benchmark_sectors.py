"""
Benchmark sector retrieval.
Returns the GICS sector weights of major benchmark ETFs for side-by-side comparison.
"""

from typing import Dict
from portfolio_analysis.data.classification import HoldingClassifier

def get_benchmark_sector_weights(benchmark_ticker: str) -> Dict[str, float]:
    """
    Get the GICS sector weights for a benchmark ETF (like SPY, VOO, VT, BND).

    Parameters
    ----------
    benchmark_ticker : str
        Benchmark ticker symbol

    Returns
    -------
    Dict[str, float]
        Sector names mapped to weight percentages (0-100)
    """
    return HoldingClassifier.get_sector_breakdown(benchmark_ticker)

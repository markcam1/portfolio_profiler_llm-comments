"""
GICS sector allocation and rollup.
Computes portfolio-level GICS sector weights based on asset-level breakdown.
"""

from typing import Dict
from portfolio_analysis.data.classification import HoldingClassifier

def get_sector_weights_rollup(portfolio_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Rolls up individual asset GICS sector weights into a single portfolio breakdown.

    Parameters
    ----------
    portfolio_weights : Dict[str, float]
        Dictionary mapping tickers to their weights in the portfolio (0.0 to 1.0 or scaled 0 to 100)

    Returns
    -------
    Dict[str, float]
        Dictionary mapping sector names to their aggregate percentage in the portfolio (0-100)
    """
    sector_weights: Dict[str, float] = {}
    
    # Check if weights sum to ~1.0 or ~100 to handle scaling gracefully
    total_wt = sum(portfolio_weights.values())
    is_fractional = total_wt <= 1.05
    
    for ticker, weight in portfolio_weights.items():
        # Scale to 0-100 percentage weight
        weight_pct = weight * 100.0 if is_fractional else weight
        
        classification = HoldingClassifier.get_classification(ticker)
        breakdown = classification.get("sector_breakdown", {})
        
        for sec, sec_wt in breakdown.items():
            # sec_wt is already 0-100 (e.g. 31.0 for VTI Technology)
            contribution = (sec_wt / 100.0) * weight_pct
            sector_weights[sec] = sector_weights.get(sec, 0.0) + contribution
            
    return sector_weights

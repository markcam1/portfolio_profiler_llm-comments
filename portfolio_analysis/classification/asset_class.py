"""
Asset class and region classification.
Determines if a ticker belongs to domestic/foreign equity/bond classes.
"""

from portfolio_analysis.data.classification import HoldingClassifier

def get_asset_class_and_region(ticker: str) -> dict[str, str]:
    """
    Get the asset class and regional classification for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol

    Returns
    -------
    dict
        Dictionary containing 'asset_class' and 'region'
    """
    classification = HoldingClassifier.get_classification(ticker)
    return {
        "asset_class": classification["asset_class"],
        "region": classification["region"],
    }

"""
Tests for the HoldingClassifier module.
"""

from portfolio_analysis.data.classification import HoldingClassifier


def test_static_lookups():
    """Verify that common static tickers are classified correctly without API calls."""
    # VTI (Domestic Equity ETF)
    vti = HoldingClassifier.get_classification("VTI")
    assert vti["asset_class"] == "equity"
    assert vti["region"] == "domestic"
    assert vti["sector"] is None
    assert "Technology" in vti["sector_breakdown"]
    assert vti["sector_breakdown"]["Technology"] == 31.0

    # BND (Domestic Bond ETF)
    bnd = HoldingClassifier.get_classification("BND")
    assert bnd["asset_class"] == "bond"
    assert bnd["region"] == "domestic"
    assert bnd["sector"] == "Fixed Income"
    assert bnd["sector_breakdown"] == {"Fixed Income": 100.0}

    # VXUS (International Equity ETF)
    vxus = HoldingClassifier.get_classification("VXUS")
    assert vxus["asset_class"] == "equity"
    assert vxus["region"] == "foreign"
    assert vxus["sector_breakdown"]["Financial Services"] == 19.5

    # AAPL (Single stock)
    aapl = HoldingClassifier.get_classification("AAPL")
    assert aapl["asset_class"] == "equity"
    assert aapl["region"] == "domestic"
    assert aapl["sector"] == "Technology"
    assert aapl["sector_breakdown"] == {"Technology": 100.0}


def test_fallback_lookup():
    """Verify that unknown/garbage tickers fall back gracefully instead of crashing."""
    garbage = HoldingClassifier.get_classification("XYZ123ABC")
    assert garbage["asset_class"] == "equity"
    assert garbage["region"] == "domestic"
    assert garbage["sector"] is None
    assert "Technology" in garbage["sector_breakdown"]


def test_sector_breakdown_convenience():
    """Verify the convenience getter for sector breakdown."""
    breakdown = HoldingClassifier.get_sector_breakdown("AAPL")
    assert breakdown == {"Technology": 100.0}

"""
Tests for the new AI Analysis schema module.
"""

from portfolio_analysis.ai_analysis.schema import ProfileItem, ProfileSnapshot, ItemComment

def test_profile_item_dataclass():
    """Verify ProfileItem initialization and attributes."""
    item = ProfileItem(
        item_id="annual_volatility",
        item_type="metric",
        title="Annual Volatility",
        data={"value_pct": 10.57},
        phase="v1"
    )
    assert item.item_id == "annual_volatility"
    assert item.item_type == "metric"
    assert item.title == "Annual Volatility"
    assert item.data == {"value_pct": 10.57}
    assert item.phase == "v1"

def test_profile_snapshot_dataclass():
    """Verify ProfileSnapshot initialization and attributes."""
    item = ProfileItem(
        item_id="annual_volatility",
        item_type="metric",
        title="Annual Volatility",
        data={"value_pct": 10.57}
    )
    snapshot = ProfileSnapshot(
        as_of_date="2026-05-26",
        base_currency="USD",
        items=[item],
        portfolio_summary="Balanced portfolio"
    )
    assert snapshot.as_of_date == "2026-05-26"
    assert snapshot.base_currency == "USD"
    assert len(snapshot.items) == 1
    assert snapshot.items[0].item_id == "annual_volatility"
    assert snapshot.portfolio_summary == "Balanced portfolio"

def test_item_comment_dataclass():
    """Verify ItemComment initialization and attributes."""
    comment = ItemComment(
        item_id="annual_volatility",
        comment="This is a test comment.",
        status="ok"
    )
    assert comment.item_id == "annual_volatility"
    assert comment.comment == "This is a test comment."
    assert comment.status == "ok"

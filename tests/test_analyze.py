"""
Tests for the analyze module of the AI Analysis layer.
Uses mocking to avoid live network requests or credential requirements.
"""

import pytest
from unittest.mock import MagicMock, patch
from portfolio_analysis.ai_analysis.schema import ProfileItem, ProfileSnapshot, ItemComment
from portfolio_analysis.ai_analysis.analyze import analyze_profile, clear_commentary_cache

@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure a clean commentary cache before every test run."""
    clear_commentary_cache()

@patch("portfolio_analysis.ai_analysis.analyze.get_client")
def test_analyze_profile_basic(mock_get_client):
    """Verify standard commentary generation works with correct mapping and JSON parsing."""
    # 1. Setup mock client response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        '[\n'
        '  {"item_id": "annual_return", "comment": "Excellent return of 7.35%."},\n'
        '  {"item_id": "annual_volatility", "comment": "Moderate volatility of 10.57%."}\n'
        ']'
    )
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    # 2. Build snapshot
    item1 = ProfileItem(item_id="annual_return", item_type="metric", title="Annual Return", data={"value_pct": 7.35})
    item2 = ProfileItem(item_id="annual_volatility", item_type="metric", title="Annual Volatility", data={"value_pct": 10.57})
    snapshot = ProfileSnapshot(
        as_of_date="2026-05-26",
        base_currency="USD",
        items=[item1, item2],
        portfolio_summary="Balanced portfolio"
    )

    # 3. Trigger analysis
    comments = analyze_profile(snapshot)

    assert len(comments) == 2
    assert comments[0].item_id == "annual_return"
    assert comments[0].comment == "Excellent return of 7.35%."
    assert comments[0].status == "ok"
    
    assert comments[1].item_id == "annual_volatility"
    assert comments[1].comment == "Moderate volatility of 10.57%."
    assert comments[1].status == "ok"

    # Verify exactly one call is made
    mock_client.models.generate_content.assert_called_once()

@patch("portfolio_analysis.ai_analysis.analyze.get_client")
def test_analyze_profile_caching(mock_get_client):
    """Verify that calling analyze_profile with an unchanged snapshot serves from cache."""
    # 1. Setup mock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '[{"item_id": "annual_return", "comment": "Cached return."}]'
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    item = ProfileItem(item_id="annual_return", item_type="metric", title="Annual Return", data={"value_pct": 7.35})
    snapshot = ProfileSnapshot(
        as_of_date="2026-05-26",
        base_currency="USD",
        items=[item],
        portfolio_summary="Balanced portfolio"
    )

    # Call twice
    res1 = analyze_profile(snapshot)
    res2 = analyze_profile(snapshot)

    assert res1 == res2
    assert res1[0].comment == "Cached return."
    
    # Assert only one call was actually dispatched to Gemini
    mock_client.models.generate_content.assert_called_once()

@patch("portfolio_analysis.ai_analysis.analyze.get_client")
def test_analyze_profile_missing_items(mock_get_client):
    """Verify that missing items in the LLM response get synthesized error comments."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # LLM only returns comment for annual_return, missing annual_volatility
    mock_response.text = '[{"item_id": "annual_return", "comment": "Return comment."}]'
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    item1 = ProfileItem(item_id="annual_return", item_type="metric", title="Annual Return", data={"value_pct": 7.35})
    item2 = ProfileItem(item_id="annual_volatility", item_type="metric", title="Annual Volatility", data={"value_pct": 10.57})
    snapshot = ProfileSnapshot(
        as_of_date="2026-05-26",
        base_currency="USD",
        items=[item1, item2],
        portfolio_summary="Balanced portfolio"
    )

    comments = analyze_profile(snapshot)
    assert len(comments) == 2
    assert comments[0].item_id == "annual_return"
    assert comments[0].status == "ok"
    
    assert comments[1].item_id == "annual_volatility"
    assert comments[1].status == "error"
    assert "Error:" in comments[1].comment

@patch("portfolio_analysis.ai_analysis.analyze.get_client")
def test_analyze_profile_json_retry(mock_get_client):
    """Verify that JSON parse failure triggers a retry and handles secondary failures gracefully."""
    mock_client = MagicMock()
    
    # First response is invalid prose
    mock_response1 = MagicMock()
    mock_response1.text = "This is raw text instead of JSON!"
    
    # Second response (retry) is valid JSON
    mock_response2 = MagicMock()
    mock_response2.text = '[{"item_id": "annual_return", "comment": "Recovered."}]'
    
    mock_client.models.generate_content.side_effect = [mock_response1, mock_response2]
    mock_get_client.return_value = mock_client

    item = ProfileItem(item_id="annual_return", item_type="metric", title="Annual Return", data={"value_pct": 7.35})
    snapshot = ProfileSnapshot(
        as_of_date="2026-05-26",
        base_currency="USD",
        items=[item],
        portfolio_summary="Balanced portfolio"
    )

    comments = analyze_profile(snapshot)
    assert len(comments) == 1
    assert comments[0].comment == "Recovered."
    assert comments[0].status == "ok"
    
    # Assert retry was called
    assert mock_client.models.generate_content.call_count == 2

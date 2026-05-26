"""
Data schemas for the AI Analysis commentary layer.
Defines the clean data contracts between deterministic finance calculations and LLMs.
"""

from dataclasses import dataclass, field
from typing import Any, List

@dataclass
class ProfileItem:
    """One commentable element on the profile page."""
    item_id: str               # stable unique id, e.g. "annual_volatility"
    item_type: str             # "metric" | "chart" | "table"
    title: str                 # human label, e.g. "Annual Volatility"
    data: dict[str, Any]       # the numbers behind this item
    phase: str = "v1"          # "v1" or "p2" (Phase 2)

@dataclass
class ProfileSnapshot:
    """Everything the AI layer needs for one batched call."""
    as_of_date: str
    base_currency: str          # default "USD"
    items: List[ProfileItem]
    portfolio_summary: str      # short plain-text whole-portfolio context

@dataclass
class ItemComment:
    """One comment, returned per item."""
    item_id: str               # must match a ProfileItem.item_id
    comment: str               # the natural-language commentary
    status: str = "ok"         # "ok" | "skipped_no_data" | "error"

"""
Classification and metadata package.
Provides asset class, region, and GICS sector rollups for portfolios.
"""

from portfolio_analysis.classification.asset_class import get_asset_class_and_region
from portfolio_analysis.classification.sectors import get_sector_weights_rollup
from portfolio_analysis.classification.benchmark_sectors import get_benchmark_sector_weights

__all__ = [
    "get_asset_class_and_region",
    "get_sector_weights_rollup",
    "get_benchmark_sector_weights",
]

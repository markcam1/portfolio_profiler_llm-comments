"""
Holding-level classification and metadata retrieval.
Supports asset class, sector, and region lookups for tickers, with dynamic
yfinance fetching and a robust static database fallback.
"""

from typing import Optional, Dict
import yfinance as yf

# Static database of common tickers for fast, offline, and rate-limit-resistant lookup
STATIC_CLASSIFICATIONS = {
    # Broad Market Equity ETFs
    "VTI": {
        "name": "Vanguard Total Stock Market ETF",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {
            "Technology": 31.0,
            "Financial Services": 13.0,
            "Healthcare": 12.0,
            "Consumer Cyclical": 10.0,
            "Industrials": 9.5,
            "Communication Services": 8.5,
            "Consumer Defensive": 6.0,
            "Energy": 4.0,
            "Real Estate": 2.5,
            "Utilities": 2.2,
            "Basic Materials": 1.3
        }
    },
    "VOO": {
        "name": "Vanguard S&P 500 ETF",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {
            "Technology": 32.0,
            "Financial Services": 12.5,
            "Healthcare": 11.5,
            "Consumer Cyclical": 10.5,
            "Industrials": 9.0,
            "Communication Services": 9.0,
            "Consumer Defensive": 6.0,
            "Energy": 3.8,
            "Real Estate": 2.3,
            "Utilities": 2.2,
            "Basic Materials": 1.2
        }
    },
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {
            "Technology": 32.0,
            "Financial Services": 12.5,
            "Healthcare": 11.5,
            "Consumer Cyclical": 10.5,
            "Industrials": 9.0,
            "Communication Services": 9.0,
            "Consumer Defensive": 6.0,
            "Energy": 3.8,
            "Real Estate": 2.3,
            "Utilities": 2.2,
            "Basic Materials": 1.2
        }
    },
    "ITOT": {
        "name": "iShares Core S&P Total U.S. Stock Market ETF",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {
            "Technology": 31.0,
            "Financial Services": 13.0,
            "Healthcare": 12.0,
            "Consumer Cyclical": 10.0,
            "Industrials": 9.5,
            "Communication Services": 8.5,
            "Consumer Defensive": 6.0,
            "Energy": 4.0,
            "Real Estate": 2.5,
            "Utilities": 2.2,
            "Basic Materials": 1.3
        }
    },
    # International Equity ETFs
    "VXUS": {
        "name": "Vanguard Total International Stock ETF",
        "asset_class": "equity",
        "region": "foreign",
        "sector_breakdown": {
            "Financial Services": 19.5,
            "Technology": 13.5,
            "Industrials": 13.0,
            "Consumer Cyclical": 11.5,
            "Healthcare": 9.0,
            "Consumer Defensive": 7.5,
            "Basic Materials": 7.0,
            "Communication Services": 5.5,
            "Energy": 5.5,
            "Utilities": 3.0,
            "Real Estate": 2.5,
            "Other": 2.5
        }
    },
    "VEA": {
        "name": "Vanguard FTSE Developed Markets ETF",
        "asset_class": "equity",
        "region": "foreign",
        "sector_breakdown": {
            "Financial Services": 20.0,
            "Industrials": 14.5,
            "Technology": 12.5,
            "Consumer Cyclical": 11.5,
            "Healthcare": 10.0,
            "Consumer Defensive": 8.0,
            "Basic Materials": 6.5,
            "Energy": 5.0,
            "Communication Services": 4.5,
            "Utilities": 3.5,
            "Real Estate": 2.5,
            "Other": 1.5
        }
    },
    "VWO": {
        "name": "Vanguard FTSE Emerging Markets ETF",
        "asset_class": "equity",
        "region": "foreign",
        "sector_breakdown": {
            "Financial Services": 22.0,
            "Technology": 21.0,
            "Consumer Cyclical": 12.0,
            "Communication Services": 9.0,
            "Basic Materials": 7.0,
            "Energy": 6.0,
            "Industrials": 6.0,
            "Consumer Defensive": 5.0,
            "Utilities": 3.0,
            "Healthcare": 3.0,
            "Real Estate": 2.0,
            "Other": 4.0
        }
    },
    # Bond ETFs
    "BND": {
        "name": "Vanguard Total Bond Market ETF",
        "asset_class": "bond",
        "region": "domestic",
        "sector_breakdown": {
            "Fixed Income": 100.0
        }
    },
    "AGG": {
        "name": "iShares Core U.S. Aggregate Bond ETF",
        "asset_class": "bond",
        "region": "domestic",
        "sector_breakdown": {
            "Fixed Income": 100.0
        }
    },
    "BNDX": {
        "name": "Vanguard Total International Bond ETF",
        "asset_class": "bond",
        "region": "foreign",
        "sector_breakdown": {
            "Fixed Income": 100.0
        }
    },
    # Common Equities
    "AAPL": {
        "name": "Apple Inc.",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Technology": 100.0}
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Technology": 100.0}
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Technology": 100.0}
    },
    "AMZN": {
        "name": "Amazon.com, Inc.",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Consumer Cyclical": 100.0}
    },
    "GOOG": {
        "name": "Alphabet Inc.",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Communication Services": 100.0}
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Communication Services": 100.0}
    },
    "META": {
        "name": "Meta Platforms, Inc.",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Communication Services": 100.0}
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Consumer Cyclical": 100.0}
    },
    "JNJ": {
        "name": "Johnson & Johnson",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Healthcare": 100.0}
    },
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "asset_class": "equity",
        "region": "domestic",
        "sector_breakdown": {"Financial Services": 100.0}
    }
}

class HoldingClassifier:
    """Classifies tickers to determine asset class, sector, region, and name."""
    
    @staticmethod
    def get_classification(ticker: str) -> Dict[str, any]:
        """
        Classifies a ticker using static database or dynamic yfinance fetch.
        
        Returns:
            Dict containing:
                - name: str (Long company or ETF name)
                - asset_class: str ("equity" | "bond" | "cash" | "other")
                - region: str ("domestic" | "foreign")
                - sector: Optional[str] (Primary sector if a single equity/bond)
                - sector_breakdown: Dict[str, float] (Sector name -> percentage mapping)
        """
        ticker_upper = ticker.upper().strip()
        
        # 1. Static Fallback Check (extremely fast & reliable)
        if ticker_upper in STATIC_CLASSIFICATIONS:
            data = STATIC_CLASSIFICATIONS[ticker_upper].copy()
            # Derive primary sector from breakdown
            breakdown = data.get("sector_breakdown", {})
            if len(breakdown) == 1:
                data["sector"] = list(breakdown.keys())[0]
            else:
                data["sector"] = None
            return data
            
        # 2. Dynamic Fetch via yfinance
        try:
            yt = yf.Ticker(ticker_upper)
            info = yt.info
            
            # Name
            name = info.get("longName") or info.get("shortName") or ticker_upper
            
            # Asset Class
            quote_type = info.get("quoteType", "").upper()
            if "BOND" in quote_type or "FIXEDINCOME" in quote_type:
                asset_class = "bond"
            elif "EQUITY" in quote_type:
                asset_class = "equity"
            elif "ETF" in quote_type:
                # Handle ETF classification by name or defaults
                name_upper = name.upper()
                if "BOND" in name_upper or "AGGREGATE" in name_upper or "TREASURY" in name_upper:
                    asset_class = "bond"
                else:
                    asset_class = "equity"
            else:
                asset_class = "equity" # Default fallback
                
            # Region (US-based is domestic, others are foreign)
            country = info.get("country", "").upper()
            if country in ["UNITED STATES", "US", "USA"] or not country:
                region = "domestic"
            else:
                region = "foreign"
                
            # Sector
            sector = info.get("sector")
            if not sector and asset_class == "bond":
                sector = "Fixed Income"
                
            # Sector Breakdown
            if sector:
                sector_breakdown = {sector: 100.0}
            elif asset_class == "bond":
                sector_breakdown = {"Fixed Income": 100.0}
            else:
                # Average equity ETF fallback if we can't get dynamic holdings
                sector_breakdown = STATIC_CLASSIFICATIONS["VTI"]["sector_breakdown"].copy()
                
            return {
                "name": name,
                "asset_class": asset_class,
                "region": region,
                "sector": sector,
                "sector_breakdown": sector_breakdown
            }
            
        except Exception:
            # Safe catch-all fallback
            is_bond = any(kw in ticker_upper for kw in ["BND", "AGG", "BOND", "BIL", "SHY"])
            is_intl = any(kw in ticker_upper for kw in ["VXUS", "VEA", "VWO", "EFA", "IEFA"])
            
            asset_class = "bond" if is_bond else "equity"
            region = "foreign" if is_intl else "domestic"
            sector = "Fixed Income" if is_bond else None
            
            if is_bond:
                sector_breakdown = {"Fixed Income": 100.0}
            elif is_intl:
                sector_breakdown = STATIC_CLASSIFICATIONS["VXUS"]["sector_breakdown"].copy()
            else:
                sector_breakdown = STATIC_CLASSIFICATIONS["VTI"]["sector_breakdown"].copy()
                
            return {
                "name": f"{ticker_upper} Asset",
                "asset_class": asset_class,
                "region": region,
                "sector": sector,
                "sector_breakdown": sector_breakdown
            }
            
    @staticmethod
    def get_sector_breakdown(ticker: str) -> Dict[str, float]:
        """Convenience function to get the sector breakdown for a ticker."""
        return HoldingClassifier.get_classification(ticker)["sector_breakdown"]

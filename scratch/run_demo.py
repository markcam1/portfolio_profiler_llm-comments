# /// script
# dependencies = [
#   "google-genai>=2.6.0",
#   "yfinance>=0.2.38",
#   "pandas>=2.0.3",
#   "numpy>=1.25.2",
#   "scipy>=1.7.0",
#   "matplotlib>=3.10.9",
#   "pandas-datareader>=0.10.0"
# ]
# ///

"""
Interactive terminal application to test the natural-language Commentary & Q&A Layer.
Allows custom portfolio input, live API key configuration, and real-time interactive Q&A.
"""

import os
import sys
# Enforce parent directory to sys.path so local package is always discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timedelta
import pandas as pd

# Core toolkit imports
from portfolio_analysis.data.loader import DataLoader
from portfolio_analysis.analysis.portfolio import PortfolioAnalysis

# LLM Layer imports
from portfolio_analysis.data.classification import HoldingClassifier
from portfolio_analysis.llm.schema import (
    Holding,
    PeriodReturn,
    SectorComparison,
    PortfolioMetrics,
)
from portfolio_analysis.llm.config import is_api_configured
from portfolio_analysis.llm.commentary import generate_commentary
from portfolio_analysis.llm.qa import answer_question


def get_interactive_portfolio():
    """Prompts the user to enter custom tickers and weights or use the default."""
    print("\n--- Configure Portfolio ---")
    choice = input("Use standard 3-Fund Portfolio (60% VTI, 30% VXUS, 10% BND)? [Y/n]: ").strip().lower()
    
    if choice in ["", "y", "yes"]:
        return {"VTI": 0.60, "VXUS": 0.30, "BND": 0.10}
        
    print("\nEnter tickers and weight percentages (e.g. AAPL 40, MSFT 30, BND 30).")
    print("When finished, leave empty and press Enter.")
    
    portfolio = {}
    while True:
        line = input(f"Add asset (Ticker %): ").strip()
        if not line:
            if not portfolio:
                print("Portfolio cannot be empty! Defaulting to standard 60/30/10.")
                return {"VTI": 0.60, "VXUS": 0.30, "BND": 0.10}
            break
            
        parts = line.split()
        if len(parts) != 2:
            print("Invalid format. Please enter 'TICKER PERCENTAGE', e.g., 'AAPL 40'")
            continue
            
        ticker, pct_str = parts[0].upper(), parts[1]
        try:
            pct = float(pct_str)
            if pct <= 0 or pct > 100:
                print("Percentage must be between 0 and 100.")
                continue
            portfolio[ticker] = pct / 100.0
        except ValueError:
            print("Percentage must be a numeric value.")
            
    # Normalize weights to sum to 1.0
    total_wt = sum(portfolio.values())
    if not np_is_close(total_wt, 1.0):
        print(f"\nWeights sum to {total_wt*100:.1f}%. Normalizing to 100.0%...")
        portfolio = {t: w / total_wt for t, w in portfolio.items()}
        
    return portfolio


def np_is_close(a, b, tol=0.0001):
    return abs(a - b) < tol


def setup_api_key():
    """Validates or prompts for the Gemini API key."""
    if is_api_configured():
        return True
        
    print("\n" + "!" * 50)
    print("GEMINI API KEY REQUIRED")
    print("!" * 50)
    print("To run the Commentary & Q&A features, a Gemini API Key is required.")
    print("You can get a free key from Google AI Studio (https://aistudio.google.com/).")
    print("-" * 50)
    
    key = input("Please paste your GEMINI_API_KEY (or press Enter to skip to math-only): ").strip()
    if key:
        os.environ["GEMINI_API_KEY"] = key
        return True
    return False


def main():
    print("=" * 60)
    print("PORTFOLIO PROFILER - INTERACTIVE TESTING TERMINAL")
    print("=" * 60)
    
    # 1. Setup API key
    has_api = setup_api_key()
    
    # 2. Get portfolio configuration
    portfolio_weights = get_interactive_portfolio()
    tickers = list(portfolio_weights.keys())
    weights_list = [portfolio_weights[t] for t in tickers]
    
    print(f"\nTarget Portfolio: " + ", ".join([f"{t}: {w*100:.1f}%" for t, w in portfolio_weights.items()]))
    
    # 3. Fetch Pricing Data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    
    print(f"\nFetching daily pricing data for {', '.join(tickers)}...")
    loader = DataLoader(tickers=tickers, start_date=start_date, end_date=end_date)
    
    try:
        data = loader.fetch_data(progress=False)
        print(f"Loaded {len(data)} trading days of historical data.")
    except Exception as e:
        print(f"Error downloading pricing data: {e}")
        print("Please check your internet connection or ticker symbols.")
        sys.exit(1)
        
    # 4. Compute Financial Metrics
    print("\nCalculating portfolio returns and risk metrics...")
    pa = PortfolioAnalysis(data, weights_list)
    summary = pa.get_summary()
    pa.print_summary()
    
    # 5. Classify and Aggregate Metadata
    print("\nAggregating asset classes, regions, and GICS sectors...")
    holdings = []
    asset_allocation = {}
    domestic_foreign_split = {}
    sector_weights = {}
    
    for ticker, weight in portfolio_weights.items():
        weight_pct = weight * 100.0
        classification = HoldingClassifier.get_classification(ticker)
        
        # Create Holding
        h = Holding(
            ticker=ticker,
            name=classification["name"],
            weight_pct=weight_pct,
            asset_class=classification["asset_class"],
            sector=classification["sector"],
            region=classification["region"]
        )
        holdings.append(h)
        
        # Accumulate asset class
        ac = classification["asset_class"]
        asset_allocation[ac] = asset_allocation.get(ac, 0.0) + weight_pct
        
        # Accumulate region
        reg = classification["region"]
        domestic_foreign_split[reg] = domestic_foreign_split.get(reg, 0.0) + weight_pct
        
        # Accumulate GICS sector breakdown
        for sec, sec_wt in classification["sector_breakdown"].items():
            contribution = (sec_wt / 100.0) * weight_pct
            sector_weights[sec] = sector_weights.get(sec, 0.0) + contribution
            
    # Calculate GICS comparison vs benchmark VOO
    benchmark_sectors = HoldingClassifier.get_sector_breakdown("VOO")
    sector_comparison = []
    all_sectors = set(list(sector_weights.keys()) + list(benchmark_sectors.keys()))
    for sector in all_sectors:
        p_wt = sector_weights.get(sector, 0.0)
        b_wt = benchmark_sectors.get(sector, 0.0)
        if p_wt > 0 or b_wt > 0:
            sector_comparison.append(SectorComparison(
                sector=sector,
                portfolio_weight_pct=p_wt,
                benchmark_weight_pct=b_wt,
                benchmark_name="Vanguard S&P 500 ETF (VOO)"
            ))
            
    # Format returns
    period_returns = [
        PeriodReturn(label="1Y", return_pct=summary["annual_return"] * 100.0)
    ]
    
    # 6. Instantiate Data Contract Metrics
    metrics = PortfolioMetrics(
        as_of_date=datetime.today().strftime("%Y-%m-%d"),
        total_value=100000.0,
        base_currency="USD",
        asset_allocation=asset_allocation,
        domestic_foreign_split=domestic_foreign_split,
        sector_weights=sector_weights,
        sector_comparison=sector_comparison,
        holdings=holdings,
        period_returns=period_returns,
        volatility_pct=summary["annual_volatility"] * 100.0,
        max_drawdown_pct=summary["max_drawdown"] * 100.0,
        top_concentration=sorted(holdings, key=lambda x: x.weight_pct, reverse=True)
    )
    
    # 7. Natural Language Actions
    if not has_api:
        print("\n" + "=" * 50)
        print("API Key not set. Math-only analysis complete.")
        print("To test the LLM features, restart the script and enter an API Key.")
        print("=" * 50)
        return
        
    print("\n--- Generating Portfolio Commentary ---")
    print("Generating review... please wait...")
    try:
        review = generate_commentary(metrics)
        print("\n=== Executive Written Review ===")
        print(review)
        print("=" * 30)
    except Exception as e:
        print(f"Error generating commentary: {e}")
        
    # Interactive Q&A loop
    print("\n" + "=" * 50)
    print("INTERACTIVE Q&A SHELL")
    print("=" * 50)
    print("You can now ask natural-language questions about this portfolio!")
    print("The agent will automatically call tools (e.g. get holdings, check sectors) to answer.")
    print("Examples:")
    print("  - 'What is my asset class weights and volatility?'")
    print("  - 'Show me my holdings in domestic equities.'")
    print("  - 'What is my biggest sector overweight compared to the S&P 500?'")
    print("Type 'exit' or 'quit' to terminate.")
    print("-" * 50)
    
    while True:
        try:
            q = input("\nUser > ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit"]:
                print("\nExiting shell. Goodbye!")
                break
                
            print("Agent is thinking & running tools...")
            ans = answer_question(q, metrics)
            print(f"\nAgent > {ans}")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\nExiting shell. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please check your API key, connection, or try a different question.")


if __name__ == "__main__":
    main()

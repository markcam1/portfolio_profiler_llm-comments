"""
Engineer Investor Portfolio Analyzer - Streamlit Web Application

A professional portfolio analysis tool built with transparency and user value in mind.

Author: Engineer Investor (@egr_investor)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Import AI Analysis modules
from portfolio_analysis.ai_analysis import ProfileItem, ProfileSnapshot, analyze_profile
from portfolio_analysis.ai_analysis.config import is_api_configured

# Import Classification modules
from portfolio_analysis.classification import (
    get_asset_class_and_region,
    get_sector_weights_rollup,
    get_benchmark_sector_weights
)
from portfolio_analysis.data.classification import HoldingClassifier

# Page configuration
st.set_page_config(
    page_title="Engineer Investor Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TRADING_DAYS = 252

PRESET_PORTFOLIOS = {
    'Custom': {},
    '60/40 Traditional': {'VTI': 0.60, 'BND': 0.40},
    'Three-Fund Portfolio': {'VTI': 0.40, 'VXUS': 0.20, 'BND': 0.40},
    'All-Weather (Ray Dalio)': {'VTI': 0.30, 'TLT': 0.40, 'IEF': 0.15, 'GLD': 0.075, 'DBC': 0.075},
    'Golden Butterfly': {'VTI': 0.20, 'VBR': 0.20, 'TLT': 0.20, 'SHY': 0.20, 'GLD': 0.20},
    'Aggressive Growth': {'VTI': 0.50, 'VGT': 0.25, 'VXUS': 0.25},
    'Conservative Income': {'VTI': 0.20, 'BND': 0.50, 'VTIP': 0.15, 'VNQ': 0.15},
    'S&P 500 Only': {'SPY': 1.0},
    'Total World Stock': {'VT': 1.0},
}

BENCHMARKS = {
    'SPY': 'S&P 500',
    'VTI': 'Total US Market',
    'VT': 'Total World',
    'BND': 'US Bonds',
    'QQQ': 'NASDAQ 100',
}


# ============================================
# Data Functions
# ============================================

@st.cache_data(ttl=3600)
def fetch_data(tickers, start_date, end_date):
    """Fetch adjusted price data from Yahoo Finance with caching."""
    try:
        # Use Ticker.history() API - returns adjusted prices by default
        # (auto_adjust=True means Close is already adjusted for dividends & splits)
        data_frames = {}
        failed_tickers = []

        for ticker in tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                # auto_adjust=True (default) returns dividend/split-adjusted prices
                hist = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True)
                if not hist.empty and 'Close' in hist.columns:
                    # Remove timezone info from index for consistency
                    hist.index = hist.index.tz_localize(None)
                    data_frames[ticker] = hist['Close']  # This is Adjusted Close
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)

        if failed_tickers:
            st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")

        if not data_frames:
            return None

        data = pd.DataFrame(data_frames)
        data = data.dropna()

        return data if not data.empty else None

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# ============================================
# Analysis Functions
# ============================================

def calculate_portfolio_metrics(data, weights, risk_free_rate=0.02):
    """Calculate portfolio performance metrics."""
    returns = data.pct_change().dropna()
    weighted_returns = returns.dot(weights)

    # Calculate metrics
    annual_return = weighted_returns.mean() * TRADING_DAYS
    annual_vol = weighted_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - risk_free_rate) / annual_vol

    # Sortino ratio
    downside_returns = weighted_returns[weighted_returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(TRADING_DAYS)
    sortino = (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0

    # Max drawdown
    cumulative = (1 + weighted_returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / peak) - 1
    max_dd = drawdown.min()

    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'cumulative_returns': cumulative,
        'weighted_returns': weighted_returns,
    }


def run_monte_carlo(data, weights, num_simulations=1000, time_horizon=252, initial_investment=10000):
    """Run Monte Carlo simulation."""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    results = np.zeros((num_simulations, time_horizon))

    for i in range(num_simulations):
        sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        portfolio_returns = sim_returns @ weights
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        results[i, :] = initial_investment * cumulative_returns

    final_values = results[:, -1]

    return {
        'results': results,
        'mean': np.mean(final_values),
        'median': np.median(final_values),
        'p5': np.percentile(final_values, 5),
        'p95': np.percentile(final_values, 95),
        'prob_loss': np.mean(final_values < initial_investment) * 100,
        'percentiles': {
            5: np.percentile(results, 5, axis=0),
            50: np.percentile(results, 50, axis=0),
            95: np.percentile(results, 95, axis=0),
        }
    }


def optimize_portfolio(data, strategy='max_sharpe', risk_free_rate=0.02):
    """Optimize portfolio weights."""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS
    n_assets = len(data.columns)

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -(ret - risk_free_rate) / vol

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)

    if strategy == 'max_sharpe':
        result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == 'min_volatility':
        result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        return None

    optimal_weights = result.x
    return {
        'weights': dict(zip(data.columns, optimal_weights)),
        'return': portfolio_return(optimal_weights),
        'volatility': portfolio_volatility(optimal_weights),
        'sharpe': (portfolio_return(optimal_weights) - risk_free_rate) / portfolio_volatility(optimal_weights)
    }


def calculate_benchmark_comparison(data, weights, benchmark_ticker, risk_free_rate=0.02):
    """Compare portfolio against a benchmark."""
    # Fetch benchmark data using Ticker API
    yf_ticker = yf.Ticker(benchmark_ticker)
    benchmark_hist = yf_ticker.history(start=data.index.min(), end=data.index.max())
    benchmark_hist.index = benchmark_hist.index.tz_localize(None)
    benchmark_data = benchmark_hist['Close']

    # Calculate returns
    portfolio_returns = data.pct_change().dropna().dot(weights)
    benchmark_returns = benchmark_data.pct_change().dropna()

    # Align dates
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]

    # Beta
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance

    # Alpha
    portfolio_mean = portfolio_returns.mean()
    benchmark_mean = benchmark_returns.mean()
    rf_daily = risk_free_rate / TRADING_DAYS
    alpha = (portfolio_mean - (rf_daily + beta * (benchmark_mean - rf_daily))) * TRADING_DAYS

    # Other metrics
    tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(TRADING_DAYS)
    info_ratio = (portfolio_returns.mean() - benchmark_returns.mean()) * TRADING_DAYS / tracking_error if tracking_error > 0 else 0

    correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]

    # Capture ratios
    up_mask = benchmark_returns > 0
    down_mask = benchmark_returns < 0

    up_capture = (portfolio_returns[up_mask].mean() / benchmark_returns[up_mask].mean() * 100) if up_mask.sum() > 0 else 0
    down_capture = (portfolio_returns[down_mask].mean() / benchmark_returns[down_mask].mean() * 100) if down_mask.sum() > 0 else 0

    return {
        'beta': beta,
        'alpha': alpha,
        'tracking_error': tracking_error,
        'information_ratio': info_ratio,
        'correlation': correlation,
        'up_capture': up_capture,
        'down_capture': down_capture,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'portfolio_annual_return': portfolio_returns.mean() * TRADING_DAYS,
        'benchmark_annual_return': benchmark_returns.mean() * TRADING_DAYS,
    }


# ============================================
# Sidebar
# ============================================

st.sidebar.title("📊 Engineer Investor")
st.sidebar.markdown("*Data-driven. No hype. Just math.*")
st.sidebar.markdown("---")

# Portfolio Selection
st.sidebar.header("Portfolio Settings")

preset = st.sidebar.selectbox(
    "Preset Portfolio",
    options=list(PRESET_PORTFOLIOS.keys()),
    index=2  # Default to Three-Fund
)

if preset != 'Custom':
    default_tickers = ', '.join(PRESET_PORTFOLIOS[preset].keys())
    default_weights = ', '.join([str(w) for w in PRESET_PORTFOLIOS[preset].values()])
else:
    default_tickers = 'VTI, VXUS, BND'
    default_weights = '0.4, 0.2, 0.4'

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", default_tickers)
weights_input = st.sidebar.text_input("Weights (comma-separated)", default_weights)

# Date range
st.sidebar.header("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start", datetime.now() - timedelta(days=5*365))
with col2:
    end_date = st.date_input("End", datetime.now())

# Risk-free rate
risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.10, 0.04, 0.005, format="%.1f%%")

st.sidebar.markdown("---")
st.sidebar.markdown("[@egr_investor](https://twitter.com/egr_investor)")
st.sidebar.markdown("[GitHub](https://github.com/engineerinvestor/Portfolio-Analysis)")
st.sidebar.caption("Not investment advice. Educational tools only.")


# ============================================
# Parse Portfolio
# ============================================

try:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    weights = np.array([float(w.strip()) for w in weights_input.split(',')])

    if len(tickers) != len(weights):
        st.error("Number of tickers must match number of weights")
        st.stop()

    if not np.isclose(weights.sum(), 1.0):
        st.error(f"Weights must sum to 1.0 (currently {weights.sum():.2f})")
        st.stop()

except Exception as e:
    st.error(f"Error parsing portfolio: {e}")
    st.stop()


# ============================================
# Main Content
# ============================================

st.title("Portfolio Analyzer")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "🎲 Monte Carlo",
    "⚖️ Optimization",
    "📊 Benchmark",
    "ℹ️ About"
])


# ============================================
# Tab 1: Performance Analysis
# ============================================

with tab1:
    st.header("Performance Analysis")

    # Fetch data
    with st.spinner("Fetching market data..."):
        data = fetch_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if data is None or data.empty:
        st.error("No data available. Check ticker symbols and date range.")
        st.stop()

    # Calculate metrics
    metrics = calculate_portfolio_metrics(data, weights, risk_free_rate)

    # ----------------------------------------------------
    # AI ANALYSIS STATE MANAGEMENT
    # ----------------------------------------------------
    # Invalidate AI analysis results if any portfolio input changes
    current_portfolio_key = f"{tickers_input}-{weights_input}-{start_date}-{end_date}-{risk_free_rate}"
    if "last_portfolio_key" not in st.session_state or st.session_state.last_portfolio_key != current_portfolio_key:
        st.session_state.last_portfolio_key = current_portfolio_key
        if "ai_analysis_results" in st.session_state:
            del st.session_state.ai_analysis_results

    # Helper function to render AI overlay commentary cards
    def render_ai_commentary(item_id: str):
        if "ai_analysis_results" in st.session_state:
            comments = st.session_state.ai_analysis_results
            comment_obj = next((c for c in comments if c.item_id == item_id), None)
            if comment_obj:
                if comment_obj.status == "ok":
                    st.info(f"🤖 **AI Analysis:** {comment_obj.comment}")
                elif comment_obj.status == "skipped_no_data":
                    st.warning(f"🤖 *Insufficient Data:* {comment_obj.comment}")
                elif comment_obj.status == "error":
                    st.error("⚠️ *AI error generating comment for this item.*")

    # ----------------------------------------------------
    # PHASE 2 NUMERIC CALCULATIONS
    # ----------------------------------------------------
    # 1. Rollup Asset Class Weights
    asset_alloc_raw = {}
    for t, w in zip(tickers, weights):
        meta = get_asset_class_and_region(t)
        ac = meta["asset_class"].capitalize()
        asset_alloc_raw[ac] = asset_alloc_raw.get(ac, 0.0) + (w * 100.0)
    asset_alloc = {ac: round(val, 2) for ac, val in asset_alloc_raw.items()}

    # 2. Rollup Region Weights
    region_alloc_raw = {}
    for t, w in zip(tickers, weights):
        meta = get_asset_class_and_region(t)
        reg = meta["region"].capitalize()
        region_alloc_raw[reg] = region_alloc_raw.get(reg, 0.0) + (w * 100.0)
    region_alloc = {reg: round(val, 2) for reg, val in region_alloc_raw.items()}

    # 3. Rollup Sector Weights
    port_weights_dict = dict(zip(tickers, weights))
    sector_alloc_raw = get_sector_weights_rollup(port_weights_dict)
    sector_alloc = {sec: round(val, 2) for sec, val in sector_alloc_raw.items()}

    # 4. Sector comparison vs benchmark VOO
    benchmark_ticker = "VOO"
    benchmark_sectors = get_benchmark_sector_weights(benchmark_ticker)
    
    sector_comparison = {}
    all_sectors = set(list(sector_alloc.keys()) + list(benchmark_sectors.keys()))
    for sec in all_sectors:
        p_wt = sector_alloc.get(sec, 0.0)
        b_wt = benchmark_sectors.get(sec, 0.0)
        if p_wt > 0 or b_wt > 0:
            sector_comparison[sec] = {
                "portfolio_pct": round(p_wt, 2),
                "benchmark_pct": round(b_wt, 2)
            }

    # 5. Holdings metadata
    holdings_metadata = []
    for t, w in zip(tickers, weights):
        classif = HoldingClassifier.get_classification(t)
        holdings_metadata.append({
            "ticker": t,
            "name": classif["name"],
            "weight_pct": round(w * 100.0, 2),
            "asset_class": classif["asset_class"].capitalize(),
            "region": classif["region"].capitalize(),
            "primary_sector": classif["sector"] or "N/A"
        })

    # ----------------------------------------------------
    # AI ANALYSIS PASS TRIGGER BOX
    # ----------------------------------------------------
    st.subheader("🤖 AI Analysis Pass")
    if not is_api_configured():
        st.warning(
            "API Key missing! Please set the 'GEMINI_API_KEY' or 'GOOGLE_API_KEY' "
            "environment variable to enable natural language commentary."
        )
    else:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            trigger_analysis = st.button("Generate AI Analysis Report", type="primary", use_container_width=True)
        with col_info:
            st.caption("Performs a fast, stateless batched commentary pass over every item, chart, and metric on this page.")
            
        if trigger_analysis:
            with st.spinner("Analyzing portfolio profile snapshot..."):
                # Construct dynamic portfolio summary for context
                summary_parts = [f"{ticker} {weight*100:.1f}%" for ticker, weight in zip(tickers, weights)]
                portfolio_summary = (
                    f"Portfolio: {', '.join(summary_parts)}. "
                    f"Annual return {metrics['Annual Return']*100:.2f}%, "
                    f"volatility {metrics['Annual Volatility']*100:.2f}%, "
                    f"max drawdown {metrics['Max Drawdown']*100:.2f}%."
                )

                # Prepare batched items (all numeric values rounded cleanly to 2 decimal places)
                profile_items = [
                    ProfileItem(item_id="annual_return", item_type="metric", title="Annual Return", data={"value_pct": round(float(metrics["Annual Return"] * 100.0), 2)}),
                    ProfileItem(item_id="annual_volatility", item_type="metric", title="Annual Volatility", data={"value_pct": round(float(metrics["Annual Volatility"] * 100.0), 2)}),
                    ProfileItem(item_id="sharpe_ratio", item_type="metric", title="Sharpe Ratio", data={"value": round(float(metrics["Sharpe Ratio"]), 2)}),
                    ProfileItem(item_id="sortino_ratio", item_type="metric", title="Sortino Ratio", data={"value": round(float(metrics["Sortino Ratio"]), 2)}),
                    ProfileItem(item_id="max_drawdown", item_type="metric", title="Max Drawdown", data={"value_pct": round(float(metrics["Max Drawdown"] * 100.0), 2)}),
                    
                    ProfileItem(item_id="allocation_ticker", item_type="chart", title="Allocation by Ticker", data={"by_ticker": {t: round(float(w * 100.0), 2) for t, w in zip(tickers, weights)}}),
                    ProfileItem(item_id="cumulative_returns", item_type="chart", title="Cumulative Returns", data={
                        "total_growth_pct": round(float((metrics["cumulative_returns"].iloc[-1] - 1) * 100.0), 2),
                        "period": f"{metrics['cumulative_returns'].index[0].strftime('%Y-%m')} to {metrics['cumulative_returns'].index[-1].strftime('%Y-%m')}",
                        "end_drawdown_pct": round(float(metrics["Max Drawdown"] * 100.0), 2)
                    }),
                    ProfileItem(item_id="individual_returns", item_type="chart", title="Individual Asset Returns", data={
                        t: round(float((1 + data[t].pct_change().dropna()).cumprod().iloc[-1] - 1) * 100.0, 2) for t in tickers
                    }),
                    
                    ProfileItem(item_id="allocation_asset_class", item_type="chart", title="Allocation by Asset Class", data={"by_asset_class": asset_alloc}, phase="p2"),
                    ProfileItem(item_id="allocation_region", item_type="chart", title="Allocation by Region", data={"by_region": region_alloc}, phase="p2"),
                    ProfileItem(item_id="sector_comparison", item_type="chart", title="Sector Weights vs Benchmark", data={"portfolio_vs_benchmark": sector_comparison}, phase="p2"),
                    ProfileItem(item_id="holdings_metadata", item_type="table", title="Holdings Metadata Table", data={"holdings": holdings_metadata}, phase="p2"),
                ]

                snapshot = ProfileSnapshot(
                    as_of_date=datetime.now().strftime("%Y-%m-%d"),
                    base_currency="USD",
                    items=profile_items,
                    portfolio_summary=portfolio_summary
                )
                
                try:
                    comments = analyze_profile(snapshot)
                    st.session_state.ai_analysis_results = comments
                    st.success("AI Commentary successfully generated! Insights have been attached below.")
                except Exception as e:
                    st.error(f"Error running AI Commentary Pass: {e}")

    # ----------------------------------------------------
    # RENDER V1 METRICS & OVERLAYS
    # ----------------------------------------------------
    st.subheader("Key Portfolio Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Annual Return", f"{metrics['Annual Return']*100:.2f}%")
        render_ai_commentary("annual_return")
    with col2:
        st.metric("Annual Volatility", f"{metrics['Annual Volatility']*100:.2f}%")
        render_ai_commentary("annual_volatility")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        render_ai_commentary("sharpe_ratio")
    with col4:
        st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
        render_ai_commentary("sortino_ratio")
    with col5:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")
        render_ai_commentary("max_drawdown")

    # Allocation pie chart & cumulative chart
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Allocation by Ticker")
        fig_pie = px.pie(
            values=weights,
            names=tickers,
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        render_ai_commentary("allocation_ticker")

    with col2:
        st.subheader("Cumulative Returns")
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=metrics['cumulative_returns'].index,
            y=metrics['cumulative_returns'].values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        fig_cumulative.update_layout(
            yaxis_title='Growth of $1',
            xaxis_title='Date',
            hovermode='x unified',
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)
        render_ai_commentary("cumulative_returns")

    # Individual asset performance
    st.subheader("Individual Asset Returns")
    asset_returns = data.pct_change().dropna()
    asset_cumulative = (1 + asset_returns).cumprod()

    fig_assets = go.Figure()
    for col in asset_cumulative.columns:
        fig_assets.add_trace(go.Scatter(
            x=asset_cumulative.index,
            y=asset_cumulative[col].values,
            mode='lines',
            name=col
        ))
    fig_assets.update_layout(
        yaxis_title='Growth of $1',
        xaxis_title='Date',
        hovermode='x unified',
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_assets, use_container_width=True)
    render_ai_commentary("individual_returns")

    # ----------------------------------------------------
    # RENDER PHASE 2 VISUAL PANELS & OVERLAYS
    # ----------------------------------------------------
    st.markdown("---")
    st.header("⚖️ Asset Class & Sector Allocation")
    st.write("Detailed metadata rollups derived dynamically from fund classifications and benchmarks.")

    col_ac, col_reg = st.columns(2)
    with col_ac:
        st.subheader("Allocation by Asset Class")
        fig_ac = px.pie(
            values=list(asset_alloc.values()),
            names=list(asset_alloc.keys()),
            hole=0.4
        )
        fig_ac.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_ac, use_container_width=True)
        render_ai_commentary("allocation_asset_class")

    with col_reg:
        st.subheader("Allocation by Region")
        fig_reg = px.pie(
            values=list(region_alloc.values()),
            names=list(region_alloc.keys()),
            hole=0.4
        )
        fig_reg.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_reg, use_container_width=True)
        render_ai_commentary("allocation_region")

    # GICS Sector Comparison Chart
    st.subheader("Sector Allocation vs S&P 500 (VOO)")
    sector_records = []
    for sec, sec_data in sector_comparison.items():
        sector_records.append({"Sector": sec, "Allocation": "Portfolio", "Weight (%)": sec_data["portfolio_pct"]})
        sector_records.append({"Sector": sec, "Allocation": "S&P 500 (VOO)", "Weight (%)": sec_data["benchmark_pct"]})
    
    if sector_records:
        sector_df = pd.DataFrame(sector_records)
        fig_sector = px.bar(
            sector_df,
            x="Sector",
            y="Weight (%)",
            color="Allocation",
            barmode="group",
            labels={"Weight (%)": "Weight (%)"},
            color_discrete_sequence=['#1F77B4', '#FF7F0E']
        )
        fig_sector.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.info("No sector exposure data available.")
    render_ai_commentary("sector_comparison")

    # Holdings details table
    st.subheader("Portfolio Holdings Details")
    metadata_df = pd.DataFrame(holdings_metadata)
    metadata_df.columns = ["Ticker", "Name", "Weight (%)", "Asset Class", "Region", "Primary Sector"]
    st.dataframe(
        metadata_df.style.format({"Weight (%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )
    render_ai_commentary("holdings_metadata")


# ============================================
# Tab 2: Monte Carlo Simulation
# ============================================

with tab2:
    st.header("Monte Carlo Simulation")

    col1, col2, col3 = st.columns(3)
    with col1:
        mc_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100)
    with col2:
        mc_horizon = st.slider("Time Horizon (days)", 21, 1260, 252, 21)
    with col3:
        mc_initial = st.number_input("Initial Investment ($)", 1000, 1000000, 10000, 1000)

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            mc_results = run_monte_carlo(data, weights, mc_simulations, mc_horizon, mc_initial)

        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median Final Value", f"${mc_results['median']:,.0f}")
        col2.metric("5th Percentile", f"${mc_results['p5']:,.0f}")
        col3.metric("95th Percentile", f"${mc_results['p95']:,.0f}")
        col4.metric("Probability of Loss", f"{mc_results['prob_loss']:.1f}%")

        # Plot simulation
        fig_mc = go.Figure()

        # Add percentile bands
        days = np.arange(mc_horizon)
        fig_mc.add_trace(go.Scatter(
            x=days, y=mc_results['percentiles'][95],
            mode='lines', line=dict(width=0),
            showlegend=False
        ))
        fig_mc.add_trace(go.Scatter(
            x=days, y=mc_results['percentiles'][5],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 100, 255, 0.2)',
            name='5th-95th Percentile'
        ))
        fig_mc.add_trace(go.Scatter(
            x=days, y=mc_results['percentiles'][50],
            mode='lines', line=dict(color='blue', width=2),
            name='Median'
        ))
        fig_mc.add_hline(y=mc_initial, line_dash="dash", line_color="red",
                        annotation_text=f"Initial: ${mc_initial:,}")

        fig_mc.update_layout(
            title=f'Monte Carlo Simulation ({mc_simulations:,} paths)',
            xaxis_title='Trading Days',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_mc, use_container_width=True)


# ============================================
# Tab 3: Portfolio Optimization
# ============================================

with tab3:
    st.header("Portfolio Optimization")

    strategy = st.selectbox(
        "Optimization Strategy",
        ["max_sharpe", "min_volatility"],
        format_func=lambda x: "Maximum Sharpe Ratio" if x == "max_sharpe" else "Minimum Volatility"
    )

    if st.button("Optimize Portfolio", type="primary"):
        with st.spinner("Optimizing portfolio..."):
            optimal = optimize_portfolio(data, strategy, risk_free_rate)

        if optimal:
            st.subheader("Optimal Weights")

            col1, col2 = st.columns([1, 2])

            with col1:
                # Display weights
                for ticker, weight in optimal['weights'].items():
                    if weight > 0.01:
                        st.write(f"**{ticker}:** {weight*100:.1f}%")

            with col2:
                # Pie chart
                fig_opt = px.pie(
                    values=list(optimal['weights'].values()),
                    names=list(optimal['weights'].keys()),
                    hole=0.4,
                    title="Optimized Allocation"
                )
                st.plotly_chart(fig_opt, use_container_width=True)

            # Metrics comparison
            st.subheader("Comparison: Current vs Optimized")

            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{optimal['return']*100:.2f}%",
                       f"{(optimal['return'] - metrics['Annual Return'])*100:+.2f}%")
            col2.metric("Volatility", f"{optimal['volatility']*100:.2f}%",
                       f"{(optimal['volatility'] - metrics['Annual Volatility'])*100:+.2f}%")
            col3.metric("Sharpe Ratio", f"{optimal['sharpe']:.2f}",
                       f"{optimal['sharpe'] - metrics['Sharpe Ratio']:+.2f}")


# ============================================
# Tab 4: Benchmark Comparison
# ============================================

with tab4:
    st.header("Benchmark Comparison")

    benchmark_ticker = st.selectbox(
        "Select Benchmark",
        options=list(BENCHMARKS.keys()),
        format_func=lambda x: f"{x} - {BENCHMARKS[x]}"
    )

    if st.button("Compare to Benchmark", type="primary"):
        with st.spinner("Calculating benchmark comparison..."):
            comparison = calculate_benchmark_comparison(data, weights, benchmark_ticker, risk_free_rate)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Beta", f"{comparison['beta']:.3f}")
        col2.metric("Alpha (annual)", f"{comparison['alpha']*100:.2f}%")
        col3.metric("Tracking Error", f"{comparison['tracking_error']*100:.2f}%")
        col4.metric("Information Ratio", f"{comparison['information_ratio']:.3f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Correlation", f"{comparison['correlation']:.3f}")
        col2.metric("Up Capture", f"{comparison['up_capture']:.1f}%")
        col3.metric("Down Capture", f"{comparison['down_capture']:.1f}%")
        col4.metric("Return Difference",
                   f"{(comparison['portfolio_annual_return'] - comparison['benchmark_annual_return'])*100:+.2f}%")

        # Cumulative returns comparison
        st.subheader("Cumulative Returns Comparison")

        port_cum = (1 + comparison['portfolio_returns']).cumprod()
        bench_cum = (1 + comparison['benchmark_returns']).cumprod()

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(
            x=port_cum.index, y=port_cum.values,
            mode='lines', name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        fig_bench.add_trace(go.Scatter(
            x=bench_cum.index, y=bench_cum.values,
            mode='lines', name=f'Benchmark ({benchmark_ticker})',
            line=dict(color='orange', width=2)
        ))
        fig_bench.update_layout(
            yaxis_title='Growth of $1',
            xaxis_title='Date',
            hovermode='x unified'
        )
        st.plotly_chart(fig_bench, use_container_width=True)


# ============================================
# Tab 5: About
# ============================================

with tab5:
    st.header("About Engineer Investor Portfolio Analyzer")

    st.markdown("""
    ### An electrical engineer's approach to portfolio analysis

    This tool was built with the philosophy that investors deserve **transparent, data-driven tools**
    without hidden agendas or gamification.

    **What This Tool Does:**
    - Calculates standard portfolio metrics (return, volatility, Sharpe ratio)
    - Runs Monte Carlo simulations to visualize uncertainty
    - Optimizes portfolios using mean-variance optimization
    - Compares your portfolio against common benchmarks

    **What This Tool Does NOT Do:**
    - Sell you anything
    - Gamify your investing experience
    - Encourage excessive trading
    - Provide "hot tips" or "signals"

    ---

    ### Disclaimer

    **This is not investment advice.**

    These tools are for educational purposes only. Past performance does not guarantee
    future results. Always do your own research and consider consulting a qualified
    financial advisor before making investment decisions.

    ---

    ### Open Source

    This project is open source and available on GitHub:

    [github.com/engineerinvestor/Portfolio-Analysis](https://github.com/engineerinvestor/Portfolio-Analysis)

    Contributions welcome!

    ---

    ### Contact

    - Twitter: [@egr_investor](https://twitter.com/egr_investor)
    - GitHub: [engineerinvestor](https://github.com/engineerinvestor)
    """)

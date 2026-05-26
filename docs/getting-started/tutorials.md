# Tutorials

Learn Portfolio Analysis through hands-on examples.

## Interactive Notebooks

The best way to learn is by doing. Open these notebooks in Google Colab:

### 1. Basic Portfolio Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engineerinvestor/Portfolio-Analysis/blob/main/Basic_Portfolio_Analysis.ipynb)

**What you'll learn:**

- Loading historical price data
- Calculating performance metrics
- Comparing portfolios to benchmarks
- Running Monte Carlo simulations
- Visualizing portfolio performance

**Time:** ~20 minutes

---

### 2. Factor Analysis Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engineerinvestor/Portfolio-Analysis/blob/main/Factor_Analysis_Demo.ipynb)

**What you'll learn:**

- Understanding Fama-French factors
- Running factor regressions (CAPM, FF3, FF5, Carhart)
- Interpreting alpha and factor betas
- Decomposing returns by factor
- Rolling factor analysis
- Factor-aware optimization

**Time:** ~30 minutes

---

### 3. Interactive Portfolio Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engineerinvestor/Portfolio-Analysis/blob/main/Interactive_Portfolio_Analysis.ipynb)

**What you'll learn:**

- Using interactive widgets
- Real-time portfolio adjustments
- Preset portfolio templates
- Dynamic visualization

**Time:** ~15 minutes

---

## Supplementary Notebooks

For more advanced workflows and specific analysis scenarios, explore the following local Jupyter notebooks in the repository:

### Advanced Analysis & Data
* **[Asset Correlations](https://github.com/engineerinvestor/Portfolio-Analysis/blob/main/Tutorials/Asset_Correlations.ipynb)**: Guide to calculating correlation matrices, rolling correlations, and covariance estimation for multi-asset portfolios.
* **[QuantStats & yfinance Integration](https://github.com/engineerinvestor/Portfolio-Analysis/blob/main/Tutorials/Portfolio_Analysis_Using_QuantStats_and_yfinance.ipynb)**: Leverage external packages like `QuantStats` alongside `yfinance` to perform deep performance profiling.
* **[Beginner's Guide to Contributing](https://github.com/engineerinvestor/Portfolio-Analysis/blob/main/Tutorials/Beginner's_Guide_to_Contributing_to_the_Portfolio_Analysis_Repository.ipynb)**: A walkthrough guide tailored for new developers looking to contribute.

### Advanced Visualizations
* **[Financial Independence & Net Worth Goals](https://github.com/engineerinvestor/Portfolio-Analysis/blob/main/Visualization/Financial_Independence_(FI)_Goals_and_Net_Worth_Visualization.ipynb)**: Modeling long-term wealth accumulation and mapping portfolios to retirement targets.
* **[Equity Style Box Visualization](https://github.com/engineerinvestor/Portfolio-Analysis/blob/main/Visualization/Visualize_the_Equity_Style_Box.ipynb)**: Plot standard Morningstar-style equity style boxes using constituent size and style characteristics.
* **[Portfolio Allocation Charts](https://github.com/engineerinvestor/Portfolio-Analysis/blob/main/Visualization/Portfolio_Allocation_Pie_Chart.ipynb)**: Advanced plotting options for clean asset allocation pie and donut charts.

---

## Video Tutorials

Coming soon! Subscribe to [Engineer Investor](https://twitter.com/egr_investor) for updates.

---

## Example Portfolios

### Classic 60/40
```python
tickers = ['VTI', 'BND']
weights = [0.60, 0.40]
```

### Three-Fund Portfolio
```python
tickers = ['VTI', 'VXUS', 'BND']
weights = [0.50, 0.20, 0.30]
```

### All-Weather Portfolio
```python
tickers = ['VTI', 'TLT', 'IEF', 'GLD', 'DBC']
weights = [0.30, 0.40, 0.15, 0.075, 0.075]
```

### Factor Tilted
```python
tickers = ['VTI', 'VBR', 'VTV', 'MTUM', 'BND']
weights = [0.25, 0.15, 0.15, 0.15, 0.30]
# Tilts: Small-cap value, Value, Momentum
```

---

## Common Workflows

### Workflow 1: Evaluate Your Portfolio

1. Load your holdings and weights
2. Calculate performance metrics
3. Compare to a benchmark
4. Run factor regression to understand exposures
5. Generate a tear sheet report

### Workflow 2: Optimize Allocation

1. Define your investment universe
2. Set constraints (max weight, long-only, etc.)
3. Run optimization (max Sharpe or min volatility)
4. Analyze the efficient frontier
5. Backtest the optimal portfolio

### Workflow 3: Factor Analysis

1. Load portfolio returns
2. Fetch Fama-French factors
3. Run regression to estimate betas
4. Decompose returns by factor
5. Identify factor tilts and alpha

---

## Streamlit Web App

For a no-code experience, try the [Streamlit App](https://engineer-investor-portfolio-analysis.streamlit.app/):

1. Select tickers from dropdown
2. Adjust weights with sliders
3. View instant performance metrics
4. Compare to benchmarks
5. Download results

---

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/engineerinvestor/Portfolio-Analysis/issues)
- **Discussions**: [Ask questions](https://github.com/engineerinvestor/Portfolio-Analysis/discussions)
- **Twitter**: [@egr_investor](https://twitter.com/egr_investor)

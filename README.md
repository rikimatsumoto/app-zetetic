
▗▄▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖ ▗▄▄▖
   ▗▞▘▐▌     █  ▐▌     █    █  ▐▌   
 ▗▞▘  ▐▛▀▀▘  █  ▐▛▀▀▘  █    █  ▐▌   
▐▙▄▄▄▖▐▙▄▄▖  █  ▐▙▄▄▖  █  ▗▄█▄▖▝▚▄▄▖
                                    
A Streamlit application that converts your investment thesis into executable strategies using **multiple AI models**, simulates those strategies with Yahoo Finance data, and provides comprehensive performance analysis with backtesting support and full AI cost transparency.

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Comparison** | Up to 3 models at a time (Anthropic, xAI, + any Ollama model) independently interpret the same thesis |
| **Backtesting** | Set a historical start date to see how strategies would have performed |
| **Internal Transaction Ledger** | Every trade is tracked in-memory with full audit trail |
| **Performance Reports** | Downloadable Markdown + Excel reports with Sharpe ratio, drawdown, model comparison, and thesis interpretation narratives |
| **AI Cost Tracker** | Per-model token counts and estimated USD cost, with net-of-cost return calculations |

## Architecture

```
                          ┌─ Tokens: 2,841 ─ Cost: $0.2340 ──┐
User Thesis ──┬──▶ Claude Opus 4  ──▶ Strategy A ──▶ Execute ─┤
              │   ┌─ Tokens: 2,103 ─ Cost: $0.0351 ──┐       │
              ├──▶ Claude Sonnet 4 ──▶ Strategy B ──▶ Execute ─┤──▶ Compare vs Benchmark
              │   ┌─ Tokens: 1,950 ─ Cost: $0 (local) ┐      │         ▼
              └──▶ Ollama (local)  ──▶ Strategy C ──▶ Execute ─┘   Performance Report
                                                                   (Markdown + Excel)
                                                                   Net returns after AI cost
```

## Quick Start

### 1. Install Dependencies

```bash
cd streamlit-zetetic

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Start Ollama for local models

For more information, go to: https://docs.ollama.com

```bash
# Install from https://ollama.com
ollama pull minimax-m2.5:cloud
```

### 3. Run the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter your Anthropic API key in the sidebar to use Claude models.

## Tab-by-Tab Walkthrough

### Tab 1 - Thesis

Enter your investment narrative. The same text goes to all selected models independently. After generation completes, a summary shows total tokens consumed and estimated AI cost across all models.

### Tab 2 - Strategies

Each model produces 3-4 ranked strategies (Conservative → Speculative). For each model you'll see:

- **Token & cost breakdown** — input/output tokens and estimated USD cost
- Allocation tables and pie charts per strategy
- One-click execution with live or historical prices (for backtesting)

### Tab 3 - Dashboard

Live (or historical) portfolio view showing:

- **KPI cards** with portfolio value, return %, AI cost, and net return (after AI cost)
- Multi-line performance chart: all portfolios vs benchmark
- Detailed holdings table with gain/loss per ticker
- Per-portfolio Excel download

### Tab 4 -Transactions

Full internal ledger with two views:

- **Combined** - all models' trades in one table
- **Per Model** - individual breakdown with transaction IDs, cash before/after each trade

### Tab 5 — Report

Standardized performance analysis report, downloadable as Markdown or Excel:

- **Thesis interpretation narratives** - comparative analysis of how each model translated the thesis into an actionable strategy, covering risk posture divergence, instrument mix, ticker overlap, and time horizon alignment
- **Per-model breakdowns** - translation summary, position-level reasoning, and rebalancing plans
- **Metrics table** - total return, AI cost, net return (after AI cost), annualized return, Sharpe ratio, max drawdown, volatility, best/worst holdings
- **Excel report** includes dedicated "Thesis Interpretation" sheet with per-model narratives and position reasoning tables

## Project Structure

```
investment_app/
├── app.py                  # Main Streamlit app (5-tab layout)
├── strategy_generator.py   # Multi-model LLM integration (Anthropic + Ollama)
│                            # Token tracking + cost estimation per provider
├── transaction_store.py    # Internal per-model transaction tracking
│                            # Stores AI usage data alongside portfolio state
├── portfolio_manager.py    # Excel portfolio file creation & export
├── market_data.py          # Yahoo Finance: live, historical & backtest prices
├── report_generator.py     # Markdown + Excel performance reports
│                            # Comparative narrative builder + cost metrics
├── requirements.txt
├── ARCHITECTURE.md         # Technical architecture
└── README.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `anthropic` | Claude API client |
| `yfinance` | Yahoo Finance market data |
| `openpyxl` | Excel file creation (`.xlsx`) |
| `pandas` | Data manipulation |
| `plotly` | Interactive charts |
| `requests` | HTTP client for Ollama API |
| `numpy` | Numerical computation (volatility, Sharpe) |

## Sidebar Configuration

| Setting | Description |
|---------|-------------|
| **Anthropic API Key** | Required for Claude models |
| **Model Selection** | Select number of models to compare; pick Anthropic/xAI/Ollama model (auto-detected) |
| **Starting Capital** | $1,000 – $1,000,000 (default $10,000) |
| **Benchmark** | S&P 500, Dow Jones, NASDAQ, Russell 2000 |
| **Backtest Date** | Toggle on to use a historical execution date (back to 2015) |
| **Reset** | 2-step confirmation: Click → Confirm → Execute |
| **AI Cost Tracker** | Always-visible running total of tokens and cost across all models |

## AI Cost Tracking

The app tracks token usage and estimates costs at every level:

| Where | What's Shown |
|-------|-------------|
| **After generation** (Tab 1) | Total tokens and cost across all models |
| **Strategy headers** (Tab 2) | Per-model input/output tokens and cost |
| **Dashboard KPIs** (Tab 3) | AI cost and net return (portfolio return minus AI cost) per portfolio |
| **Sidebar tracker** | Persistent running total, always visible |
| **Report metrics** (Tab 5) | AI cost, tokens, net return columns in comparison tables |
| **Excel report** | Dedicated rows for AI cost and net return in the Summary sheet |

### Pricing Used

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude Opus 4 | $15.00 | $75.00 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Ollama (any) | $0 (local) | $0 (local) |

Prices are defined in `strategy_generator.py` → `ANTHROPIC_PRICING`. Update them if Anthropic adjusts pricing.

## Reset safety

The reset function requires confirmation to prevent accidental data loss:

1. Click "Reset All Data"
2. Confirm "Yes" (or cancel)

All portfolios, transactions, strategies, reports, and temporary files are permanently deleted.

## Report Metrics Explained

| Metric | What It Measures |
|--------|-----------------|
| **Total Return** | (Current Value − Initial) / Initial |
| **AI Generation Cost** | Estimated USD cost of the LLM API call that produced the strategy |
| **Net Return** | Total Return minus AI cost — the "real" return after accounting for the cost of generating the strategy |
| **Annualized Return** | Return normalized to a yearly rate |
| **Max Drawdown** | Largest peak-to-trough decline observed in snapshots |
| **Volatility** | Annualized standard deviation of daily returns |
| **Sharpe Ratio** | (Annualized Return − Risk-Free Rate) / Volatility |

## Disclaimer

This tool is for learning and experimentation only. Always consult a qualified financial advisor before making real investment decisions. The strategy is automatically generated from large language models and may contain errors, omissions, or outdated information. It is provided “as is,” without warranties of any kind, express or implied. This application is not investment, legal, security, or policy advice and must not be relied upon for decision-making. You are responsible for independently verifying facts and conclusions with primary sources and qualified professionals. Neither the author nor providers of this service accept any liability for losses or harms arising from use of this content. No duty to update is assumed.




▗▄▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖ ▗▄▄▖
   ▗▞▘▐▌     █  ▐▌     █    █  ▐▌   
 ▗▞▘  ▐▛▀▀▘  █  ▐▛▀▀▘  █    █  ▐▌   
▐▙▄▄▄▖▐▙▄▄▖  █  ▐▙▄▄▖  █  ▗▄█▄▖▝▚▄▄▖
                                    
A Streamlit application that converts your investment thesis into executable strategies using **multiple AI models**, simulates those strategies with Yahoo Finance data, and provides comprehensive performance analysis with backtesting support and full AI cost transparency. This project was built with the help of Claude Code.

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Comparison** | Up to 3 models at a time (Anthropic, xAI, Google Gemini, + Ollama locally) independently interpret the same thesis |
| **Backtesting** | Set a historical start date to see how strategies would have performed |
| **Internal Transaction Ledger** | Every trade is tracked in-memory with full audit trail |
| **Risk-Adjusted Metrics** | Sharpe, Sortino, Calmar ratios + max drawdown, volatility, win rate, profit factor |
| **Performance Reports** | Downloadable Markdown + Excel reports with 7 Plotly charts, model comparison, and thesis interpretation narratives |
| **AI Cost Tracker** | Per-model token counts and estimated USD cost, with net-of-cost return calculations |
| **Streamlit Cloud Ready** | Ollama auto-disables on cloud; 3 cloud providers always available |

## Architecture

```
                          ┌─ Tokens: 2,841 ─ Cost: $0.2340 ──┐
User Thesis ──┬──▶ Claude Opus 4    ──▶ Strategy A ──▶ Execute ─┤
              │   ┌─ Tokens: 2,103 ─ Cost: $0.0351 ──┐        │
              ├──▶ Claude Sonnet 4  ──▶ Strategy B ──▶ Execute ─┤──▶ Compare vs Benchmark
              │   ┌─ Tokens: 1,800 ─ Cost: $0.0009 ──┐        │         ▼
              ├──▶ Gemini 2.5 Flash ──▶ Strategy C ──▶ Execute ─┤   Performance Report
              │   ┌─ Tokens: 1,950 ─ Cost: $0 (local) ┐       │   (Markdown + Excel)
              └──▶ Ollama (local)   ──▶ Strategy D ──▶ Execute ─┘   Net returns after AI cost
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

> **Note:** Ollama is auto-detected. On Streamlit Cloud (or any environment without a local Ollama server), the option is silently hidden — no configuration needed.

### 3. Run the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter your API key(s) in the sidebar for your chosen provider(s).

## Tab-by-Tab Walkthrough

### Guide Tab

Beginner-friendly tutorial covering the full workflow, a table of every metric with practical "how to read it" guidance (e.g. "Sharpe above 1.0 = good; above 2.0 = excellent"), FAQ, and tips.

### Tab 1 — Thesis

Enter your investment narrative. The same text goes to all selected models independently. After generation completes, a summary shows total tokens consumed and estimated AI cost across all models.

### Tab 2 — Strategies

Each model produces 3–4 ranked strategies (Conservative → Speculative). For each model you'll see:

- **Token & cost breakdown** — input/output tokens and estimated USD cost
- Allocation tables and pie charts per strategy
- One-click execution with live or historical prices (for backtesting)

### Tab 3 — Dashboard

Live (or historical) portfolio view showing:

- **KPI cards** with portfolio value, return %, AI cost, and net return (after AI cost)
- **Risk-adjusted metrics** — 7 metric cards across 2 rows: Sharpe, Sortino, Calmar / Max Drawdown, Volatility, Win Rate, Profit Factor
- **"ℹ️ What do these metrics mean?"** — collapsed expander with plain-English definitions and examples
- Multi-line performance chart with **$/% toggle** (absolute value vs percentage return)
- Detailed holdings table with gain/loss per ticker
- Per-portfolio Excel + JSON download
- **Transaction ledger** as collapsed diagnostic expander

### Tab 4 — Report

Standardized performance analysis report, downloadable as Markdown or Excel:

- **Key Metrics Comparison** table — Total Return, Net Return, Annualized Return, Max Drawdown, Volatility, Sharpe, Sortino, Calmar, Win Rate, Profit Factor, AI Cost, Best/Worst holdings
- **"ℹ️ What do these metrics mean?"** — collapsed expander (same definitions as Dashboard)
- **7 Plotly charts** (consistent model→color mapping across all):
  1. Return comparison bar (Total / Annualized / Net)
  2. Risk-adjusted ratios (Sharpe / Sortino / Calmar)
  3. Risk profile scatter (Volatility vs Max Drawdown)
  4. Portfolio composition donut pies
  5. Drawdown timeline
  6. Rolling Sharpe ratio (30-day window)
  7. Daily return distribution histograms
- **Thesis interpretation narratives** — comparative analysis of how each model translated the thesis into strategy, covering risk posture divergence, instrument mix, ticker overlap, and time horizon alignment
- **Excel report** includes dedicated "Thesis Interpretation" sheet with per-model narratives and position reasoning tables

## Project Structure

```
investment_app/
├── app.py                  # Main Streamlit app (Guide + 4-tab layout, ~1830 lines)
├── strategy_generator.py   # Multi-model LLM integration (4 providers, ~510 lines)
│                            # Anthropic, xAI, Google Gemini, Ollama
│                            # Token tracking + cost estimation per provider
├── transaction_store.py    # Internal per-model transaction tracking
│                            # Stores AI usage data alongside portfolio state
├── portfolio_manager.py    # Excel portfolio file creation & export
├── market_data.py          # Yahoo Finance: live, historical & backtest prices
├── report_generator.py     # Markdown + Excel performance reports (~920 lines)
│                            # Comparative narrative builder + 27+ metrics
├── requirements.txt
├── ARCHITECTURE.md         # Technical architecture
├── CLAUDE_CONTEXT.md       # Development context for AI assistants
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
| `requests` | HTTP client for xAI, Google Gemini, and Ollama APIs |
| `numpy` | Numerical computation (volatility, Sharpe, Sortino, etc.) |

## Sidebar Configuration

| Setting | Description |
|---------|-------------|
| **Provider per slot** | Pick **Anthropic** (Claude), **xAI** (Grok), **Google** (Gemini), or **Ollama** (local, if available) |
| **API Key** | Shown contextually per provider; Ollama needs none |
| **Starting Capital** | $1,000 – $1,000,000 (default $10,000) |
| **Benchmark** | S&P 500, Dow Jones, NASDAQ, Russell 2000 |
| **Backtest Date** | Toggle on to use a historical execution date (back to 2024) |
| **JSON Import** | Multi-file uploader with per-file add/remove |
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
| **Report metrics** (Tab 4) | AI cost, tokens, net return columns in comparison tables |
| **Excel report** | Dedicated rows for AI cost and net return in the Summary sheet |

### Pricing Used

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude Opus 4 | $15.00 | $75.00 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Grok 3 | $3.00 | $15.00 |
| Grok 3 Mini | $0.30 | $0.50 |
| Gemini 2.5 Flash | $0.10 | $0.40 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Ollama (any) | $0 (local) | $0 (local) |

Prices are defined in `strategy_generator.py` → `ANTHROPIC_PRICING` / `XAI_PRICING` / `GEMINI_PRICING`. Update them if providers adjust pricing.

## Reset Safety

The reset function requires confirmation to prevent accidental data loss:

1. Click "Reset All Data"
2. Confirm "Yes" (or cancel)

All portfolios, transactions, strategies, reports, and temporary files are permanently deleted.

## Report Metrics Explained

| Metric | What It Measures | How to Read It |
|--------|-----------------|----------------|
| **Total Return** | (Current Value − Initial) / Initial | +15% = grew 15%. Negative = lost money |
| **Net Return** | Total Return minus AI cost | Compare to Total Return to see if AI cost is material |
| **Annualized Return** | Return normalized to a yearly rate | 8–12% annually is a solid benchmark |
| **Max Drawdown** | Largest peak-to-trough decline | Lower is better. Above 30% = aggressive risk |
| **Volatility** | Annualized std dev of daily returns | Under 15% = stable; above 30% = very bumpy |
| **Sharpe Ratio** | (Ann. Return − Risk-Free Rate) / Volatility | Above 1.0 = good; above 2.0 = excellent |
| **Sortino Ratio** | Like Sharpe but only penalises downside | Higher = better. Sortino > Sharpe means upside volatility dominates |
| **Calmar Ratio** | Annualized Return / Max Drawdown | Above 1.0 = good; above 3.0 = excellent |
| **Win Rate** | % of trading days with positive return | 50–55% typical; above 55% = consistent |
| **Profit Factor** | Sum of gains / sum of losses | Above 1.0 = gains outweigh losses; above 1.5 = strong |

## Disclaimer

This tool is for learning and experimentation only. Always consult a qualified financial advisor before making real investment decisions. The strategy is automatically generated from large language models and may contain errors, omissions, or outdated information. It is provided "as is," without warranties of any kind, express or implied. This application is not investment, legal, security, or policy advice and must not be relied upon for decision-making. You are responsible for independently verifying facts and conclusions with primary sources and qualified professionals. Neither the author nor providers of this service accept any liability for losses or harms arising from use of this content. No duty to update is assumed.

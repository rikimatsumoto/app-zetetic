# Zetetic — Investment Strategy Engine

A Streamlit application that converts your investment thesis into executable strategies using **multiple AI models**, simulates those strategies with Yahoo Finance data, and provides comprehensive performance analysis with backtesting support and full AI cost transparency. This project was built with the help of Claude Code.

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Comparison** | Up to 3 model slots, each independently configurable across 5 providers (Anthropic, xAI, Google Gemini, Ollama Cloud, Ollama Local) |
| **Backtesting** | Set a historical start date to simulate how strategies would have performed |
| **Risk-Adjusted Metrics** | Sharpe, Sortino, Calmar ratios plus Max Drawdown, Volatility, Win Rate, Profit Factor |
| **7 Report Charts** | Heatmaps, scatter plots, composition pies, drawdown timelines, rolling Sharpe, return distributions — all scaling to 15 strategies |
| **AI Cost Tracker** | Per-model token counts and estimated USD cost, with net-of-cost return calculations |
| **JSON Persistence** | Export portfolios as JSON, re-import days/weeks/months later to track real performance |
| **Security-First Imports** | 10-layer validation on every uploaded JSON file |

## Architecture

```
                          ┌─ Tokens: 2,841 ─ Cost: $0.2340 ──┐
User Thesis ──┬──▶ Claude Opus 4   ──▶ Strategy A ──▶ Execute ─┤
              │   ┌─ Tokens: 1,950 ─ Cost: $0.0012 ──┐        │
              ├──▶ Gemini 2.5 Flash ──▶ Strategy B ──▶ Execute ─┤──▶ Compare vs Benchmark
              │   ┌─ Tokens: 2,103 ─ Cost: $0 (sub) ──┐       │         ▼
              └──▶ Ollama Cloud     ──▶ Strategy C ──▶ Execute ─┘   Performance Report
                                                                    (Markdown + Excel)
                                                                    Net returns after AI cost
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter your API key(s) in the sidebar for whichever provider(s) you want to use.

### 3. (Optional) Local Ollama Models

```bash
# Install from https://ollama.com
ollama pull llama3.1
ollama serve
```

When running locally, Ollama Local auto-detects available models. On Streamlit Cloud, Ollama Local is automatically hidden (no local GPU), but Ollama Cloud remains available with an API key from ollama.com.

## Supported Providers

| Provider | Models | API Key Required | Cost |
|----------|--------|-----------------|------|
| **Anthropic** | Claude Opus 4, Sonnet 4 | Yes | Per-token ($3–$75/1M tokens) |
| **xAI** | Grok 3, Grok 3 Mini | Yes | Per-token ($0.30–$15/1M tokens) |
| **Google** | Gemini 2.5 Flash, 1.5 Pro | Yes | Per-token ($0.15–$5/1M tokens) |
| **Ollama Cloud** | Dynamic (fetched from account) | Yes (ollama.com) | Subscription ($0–$100/month) |
| **Ollama Local** | Dynamic (auto-detected) | No | Free |

## Tab-by-Tab Walkthrough

### Guide Tab

Beginner-friendly tutorial covering the full workflow, FAQ, metric definitions with practical benchmarks (e.g. "Sharpe above 1.0 = good; above 2.0 = excellent"), and tips for getting the most out of the app.

### Tab 1 — Thesis

Enter your investment narrative. The same text goes to all selected models independently. After generation completes, a summary shows total tokens consumed and estimated AI cost across all models.

### Tab 2 — Strategies

Each model produces 3–4 ranked strategies (Conservative → Speculative). For each you'll see allocation tables and pie charts, plus the model's rebalancing notes and time horizon advice. Controls include:

- **Execution date**: "Today (live prices)" or "Historical date (backtest)" with date picker
- **Execute All**: one-click batch execution of all strategies across all models
- **Per-strategy execute**: individual control for selective execution
- **Execution reset**: clears executed portfolios while preserving generated strategies (avoids costly re-generation)
- **JSON export**: download each portfolio for later re-import

### Tab 3 — Dashboard

Portfolio view with:

- **KPI cards** — portfolio value, return %, AI cost, and net return (after AI cost) per strategy, with day-0 gating
- **Risk metrics** — 7 metrics per strategy in a full-width row: Sharpe, Sortino, Calmar, Max Drawdown, Volatility, Win Rate, Profit Factor
- **Performance chart** — all portfolios vs benchmark with $/% toggle; lines differentiated by both color (10 palette) and dash pattern (4 styles) for up to 40 unique combos
- **Holdings table** — per-ticker gain/loss with Excel + JSON export
- **Transaction ledger** — collapsed diagnostic expander showing the full internal audit trail
- **Metric glossary** — expandable reference explaining each metric with practical interpretation

### Tab 4 — Report

Auto-generated performance analysis, refreshed on each visit with latest prices:

- **Metrics comparison table** — all strategies side-by-side with return, risk, and cost columns
- **7 interactive charts**:
  1. Return Comparison (heatmap) — total, annualized, and net return
  2. Risk Ratios (heatmap) — Sharpe, Sortino, Calmar, Win Rate, Profit Factor
  3. Volatility vs Max Drawdown (scatter)
  4. Portfolio Composition (donut pies, 3-column grid)
  5. Drawdown Timeline (overlaid lines)
  6. Rolling Sharpe (30-day window)
  7. Daily Return Distribution (overlaid histograms)
- Charts 5–7 include an **overlay limiter** when >5 strategies: select up to 5 to display, defaulting to top performers
- **Downloads**: Markdown (.md) and Excel (.xlsx) reports

## Project Structure

```
investment_app/
├── app.py                  # Streamlit UI controller (~2180 lines)
├── strategy_generator.py   # Multi-provider LLM integration (~750 lines)
│                            # 5 providers, token tracking, cost estimation
├── models.toml             # Model names & pricing config
│                            # Edit this file to add/remove/update models
├── transaction_store.py    # Per-model transaction tracking (~240 lines)
│                            # Holdings, cash, snapshots, AI usage data
├── market_data.py          # Yahoo Finance wrapper (~160 lines)
│                            # Live, historical, and backtest prices
├── portfolio_manager.py    # Per-portfolio Excel export (~160 lines)
├── report_generator.py     # Metrics engine + report builder (~920 lines)
│                            # 27+ metrics, comparative narrative, Excel/Markdown
├── ARCHITECTURE.md         # Technical architecture reference
├── CLAUDE_CONTEXT.md       # Development history and context for AI assistants
├── README.md
└── requirements.txt
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
| `requests` | HTTP client for xAI, Gemini, Ollama APIs |
| `numpy` | Numerical computation (volatility, ratios) |

## Sidebar Configuration

| Setting | Description |
|---------|-------------|
| **Model Slots (1–3)** | Each slot selects a provider + model independently |
| **API Keys** | Shown contextually per selected provider |
| **Starting Capital** | $1,000 – $1,000,000 (default $10,000) |
| **Benchmark** | S&P 500, Dow Jones, NASDAQ, Russell 2000 |
| **Reset** | Multi-step confirmation: Click → Confirm → Type "DELETE" → Execute |
| **JSON Import** | Multi-file uploader with per-file add/remove |
| **AI Cost Tracker** | Always-visible running total of tokens and cost |

## AI Cost Tracking

The app tracks token usage and estimates costs at every level:

| Where | What's Shown |
|-------|-------------|
| **After generation** (Tab 1) | Total tokens and cost across all models |
| **Strategy headers** (Tab 2) | Per-model input/output tokens and cost |
| **Dashboard KPIs** (Tab 3) | AI cost and net return per portfolio |
| **Sidebar tracker** | Persistent running total, always visible |
| **Report metrics** (Tab 4) | AI cost and net return columns in comparison table |
| **Excel report** | Dedicated rows for AI cost and net return |

Pricing is defined per provider in `strategy_generator.py`. Ollama (Local and Cloud) shows $0 — local is free, cloud is subscription-based with no per-token charges.

## Report Metrics

| Metric | What It Measures | Good Benchmark |
|--------|-----------------|----------------|
| **Total Return** | (Current − Initial) / Initial | Positive, above benchmark |
| **Annualized Return** | Return normalized to yearly rate | Above risk-free rate (5%) |
| **Net Return** | Total return minus AI generation cost | Positive after costs |
| **Max Drawdown** | Largest peak-to-trough decline | Below 20% |
| **Volatility** | Annualized std dev of daily returns | Below 20% |
| **Sharpe Ratio** | Risk-adjusted return (vs risk-free rate) | Above 1.0 good, above 2.0 excellent |
| **Sortino Ratio** | Like Sharpe but penalizes only downside volatility | Above 1.5 good |
| **Calmar Ratio** | Annualized return / max drawdown | Above 1.0 good |
| **Win Rate** | % of days with positive returns | Above 52% |
| **Profit Factor** | Total gains / total losses | Above 1.5 good |

## JSON Persistence Workflow

The app is stateless between sessions. To track performance over time:

1. **Execute** strategies (live or backtest)
2. **Download** portfolio JSON from the Dashboard
3. **Close** the app — come back days, weeks, or months later
4. **Upload** the JSON(s) via the sidebar importer
5. The app replays price history from execution date to today and shows real returns

Multiple JSON files can be imported simultaneously. Each file is independently validated with 10 security checks before loading.

## Planned: Portfolio Rebalancing

The LLM already generates rebalancing advice (e.g. "rebalance quarterly when any position drifts more than 5%") — currently displayed but not acted upon. Planned implementation:

**Phase 1 — Mechanical Rebalancing**: Store original target weights alongside the portfolio. A simulation engine will step through dates and apply rebalancing trades when drift or schedule triggers fire, using historical prices. Works identically for backtested and forward-looking portfolios.

**Phase 2 — LLM-Assisted Rebalancing**: Send the original thesis + current portfolio state to the LLM, which returns updated target weights. Same execution pipeline as mechanical, but with AI-refreshed targets.

## Updating Models

All model names and pricing live in `models.toml`. To add a new model (e.g. Gemini 3 Flash), just add two entries:

```toml
# Under [google.models], add:
"Gemini 3 Flash (Preview)" = "gemini-3-flash-preview"

# Add a new pricing block:
[google.pricing."gemini-3-flash-preview"]
input  = 0.10
output = 0.40
```

No Python code changes needed. The app picks up changes on next restart.

## Disclaimer

This tool is for **educational and hypothetical purposes only**. It does not constitute financial advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making real investment decisions.

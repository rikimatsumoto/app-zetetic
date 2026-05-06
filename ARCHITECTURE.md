# Technical Architecture — Zetetic

## System Overview

Zetetic is a 6-module Streamlit application. A user's investment thesis flows through an LLM pipeline, gets converted into executable trades against Yahoo Finance data, and is analyzed in downloadable reports. All state lives in `st.session_state` — there is no database or filesystem persistence (though portfolios can be exported/imported as JSON for tracking over time).

```
┌─────────────────────────────────────────────────────────────────────┐
│  app.py  (Streamlit UI — 4 main tabs + Guide)                      │
│                                                                     │
│  Tab 1: Thesis ─────────────────────────────────────────────────┐   │
│  │  User writes investment narrative                            │   │
│  │  → strategy_generator.py (5 providers)                       │   │
│  │  → session_state["model_results"] + ["model_usages"]         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 2: Strategies ─────────────────────────────────────────────┐   │
│  │  Review allocations → Date controls → Execute All / each     │   │
│  │  → market_data → strategies_to_trades → transaction_store    │   │
│  │  → JSON export │ Execution reset (preserves strategies)      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 3: Dashboard ──────────────────────────────────────────────┐   │
│  │  KPI cards, 7 risk metrics (full-width rows per strategy),   │   │
│  │  perf chart ($/% toggle, dash-differentiated lines),         │   │
│  │  holdings detail, Excel/JSON export, txn ledger (expander)   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 4: Report ────────────────────────────────────────────────┐   │
│  │  Auto-generated metrics table, 7 charts (heatmaps, scatter,  │   │
│  │  pies, drawdown, rolling Sharpe, distribution), overlay       │   │
│  │  limiter (top 5), .md + .xlsx download                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Sidebar: Model config (5 providers), portfolio settings, reset,   │
│           multi-file JSON import, AI cost tracker                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `app.py` — UI Controller (~2180 lines)

Entry point. Owns all session state. Contains:
- **Security validation** (`_validate_portfolio_json`): 10 checks on every JSON import (size, UTF-8, structure, types, string sanitization, ticker format, numeric ranges, date validation, nested payload caps)
- **Day-0 detection** (`_is_day_zero`): gates meaningless metrics for same-day execution
- **Multi-file import**: `accept_multiple_files=True` with per-file identity tracking (`_import_file_ids` list + `_import_file_to_keys` dict) so each file can be added or removed independently
- **Chart styling constants**: `_PALETTE` (10 colors), `_DASH_PATTERNS` (4 styles), `_adaptive_legend()` helper — shared by Dashboard and Report tabs
- **Execution helpers**: `_execute_strategy()` for single or batch execution, execution reset that preserves generated strategies

**Session state keys:**

| Key | Type | Purpose |
|-----|------|---------|
| `portfolios` | `dict[str, dict]` | Executed portfolio state dicts |
| `model_results` | `dict[str, dict]` | Raw LLM strategy JSON per model |
| `model_usages` | `dict[str, dict]` | Token counts + cost per model |
| `model_providers` | `dict[str, str]` | Provider key per model label |
| `trades_executed` | `dict[str, bool]` | Dedup flag per exec_key |
| `strategies_generated` | `bool` | Gate for tabs 2–4 |
| `report_generated` | `bool` | Gate for report display |
| `execution_date` | `str` | YYYY-MM-DD — set in Tab 2 |
| `_import_file_ids` | `list[str]` | File identity strings for processed uploads |
| `_import_file_to_keys` | `dict[str, list]` | Maps file_id → [exec_key, ...] for removal |

**Tab 3 Dashboard features:**
- KPI cards with day-0 gating (shows "Day 0 — tracking started" instead of 0% return)
- Risk metrics: 7 `st.metric()` cards in a full-width `st.columns(7)` row per strategy (Sharpe, Sortino, Calmar, Max Drawdown, Volatility, Win Rate, Profit Factor)
- Performance chart with Absolute ($) / Percentage (%) toggle, dash-differentiated lines
- Per-portfolio holdings table with Excel + JSON export buttons
- Transaction ledger as collapsed diagnostic expander (not a main workflow step)

**Tab 4 Report features (auto-generated):**
- Metrics comparison table with all risk-adjusted ratios
- 7 charts using consistent color/dash maps:
  - **Charts 1–2**: Heatmaps (returns, risk ratios) with `RdYlGn` colorscale and numeric cell values — scales to 15+ strategies
  - **Chart 3**: Volatility vs Max Drawdown scatter
  - **Chart 4**: Portfolio composition donut pies (3-column grid)
  - **Charts 5–7**: Drawdown timeline, rolling Sharpe (30-day), daily return distribution — with overlay limiter (`st.multiselect`, top 5 by return, capped at 5)
- Adaptive legends: horizontal ≤7 strategies, vertical sidebar >7
- Downloadable Markdown + Excel reports

---

### `strategy_generator.py` — LLM Integration (~750 lines)

5 providers, unified interface. Model names and pricing loaded from `models.toml` at startup — no Python changes needed to add/remove/update models.

| Provider | Models | API Style | Cost Model |
|----------|--------|-----------|------------|
| Anthropic | Claude Opus 4, Sonnet 4 | `anthropic` SDK | Per-token ($3–$75/1M) |
| xAI | Grok 3, Grok 3 Mini | OpenAI-compatible REST | Per-token ($0.30–$15/1M) |
| Google | Gemini 2.5 Flash, 1.5 Pro | REST (`generativelanguage.googleapis.com`) | Per-token ($0.15–$5/1M) |
| Ollama Cloud | Dynamic (fetched from `ollama.com/api/tags`) | REST (`ollama.com/api/chat` + Bearer auth) | Subscription ($0–$100/mo) |
| Ollama Local | Dynamic (fetched from `localhost:11434/api/tags`) | REST (`localhost:11434/api/chat`) | Free |

Key functions:
- `generate_strategies(thesis, provider, model_name, api_key, ...)` — unified dispatcher
- `strategies_to_trades(strategy, total_capital, prices)` — allocation → concrete trades
- `_clean_json_text()` — 3-layer JSON sanitizer (comments, trailing commas, literal newlines in strings)

---

### `transaction_store.py` — Portfolio State (~240 lines)

In-memory ledger. `clear_all_portfolios()` resets all portfolio + import tracking keys (`_import_file_ids`, `_import_file_to_keys`).

---

### `market_data.py` — Yahoo Finance (~160 lines)

Wraps `yfinance` with MultiIndex compatibility. Live prices, historical series, backtest date resolution.

---

### `report_generator.py` — Report Engine (~920 lines)

**`compute_metrics()`** returns 27+ stats including:
- Return: total (% and $), annualized, net-of-AI-cost
- Risk: max drawdown, annualized volatility, downside deviation
- Risk-adjusted: **Sharpe ratio**, **Sortino ratio**, **Calmar ratio**
- Consistency: **win rate** (% positive days), **profit factor** (gains ÷ losses)
- Position analysis: best/worst holding returns, cash remaining, % invested
- AI cost: tokens used, estimated cost USD

**`generate_markdown_report()`** — comparative markdown with metrics table (includes Sortino/Calmar), per-model detail sections, AI-generated narrative

**`generate_excel_report()`** — multi-sheet .xlsx with styled metrics, per-model holding sheets

---

### `portfolio_manager.py` — Excel Export (~160 lines)

Per-portfolio .xlsx with holdings, gains/losses, and summary stats.

---

## Chart Scalability System

The app supports up to 15 simultaneous strategies (3 models × 5 strategies each). The chart system is designed to remain readable at this scale:

```
Strategy index:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
Color (10):      ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
Dash pattern:    ── ── ── ── ── ── ── ── ── ── -- -- -- -- --
                 solid (0-9)                    dash (10-14)
```

- **Color**: `_PALETTE[i % 10]` — 10 visually distinct colors
- **Dash**: `_DASH_PATTERNS[i // 10]` — cycles after each full color rotation
- **Result**: 40 unique (color, dash) combinations before any repetition
- **Stable mapping**: Built from full `portfolios` dict insertion order, not from filtered subsets. Both Dashboard and Report look up by label, ensuring identical colors across tabs even when day-0 portfolios create gaps.

For overlay charts (drawdown, rolling Sharpe, histogram), a `st.multiselect` limits display to 5 strategies when >5 exist, defaulting to top performers by total return.

---

## Export / Import Persistence

```
Execute today → 💾 Export JSON → ... days pass ... → 📂 Import JSON(s) → real returns
```

Multi-file uploader supports importing several portfolios simultaneously. Each file is independently validated (size, structure, types, ticker format, date range) before loading. Removing a file from the uploader removes exactly its associated portfolio(s) from session state.

---

## Planned: Rebalancing

### Architecture

The app is stateless between sessions. Every evaluation — whether "backtested" from a year ago or "live" from yesterday — follows the same path: replay price history from `start_date` to today. There is no live monitoring.

The rebalancing system will add a **simulation engine** that replaces the current read-only price replay loop. Instead of computing `cash + Σ(shares × price)` independently at each date, it steps through dates sequentially, checking triggers and mutating a working copy of holdings when rebalancing fires.

### New primitives (in `transaction_store.py`)

| Function | Purpose |
|----------|---------|
| `record_sell(portfolio, ticker, shares, price)` | Mirror of `record_buy()` — removes shares, adds cash |
| `compute_current_weights(portfolio, prices)` | Returns `{ticker: weight, "cash": weight}` |
| `compute_rebalance_trades(portfolio, target_weights, prices)` | Source-agnostic: any target weights → sell/buy orders |
| `execute_rebalance(portfolio, trades, prices)` | Calls sell/buy, logs to `rebalance_history` |

### New portfolio fields

| Field | Type | Purpose |
|-------|------|---------|
| `target_allocations` | `dict[str, float]` | Original `{ticker: weight}` from strategy — source of truth for mechanical rebalancing |
| `rebalance_config` | `dict` | `{frequency, drift_threshold, last_rebalance_date}` |
| `rebalance_history` | `list[dict]` | Timestamped events: trigger type, trades, pre/post weights |

### Phases

**Phase 1 (Mechanical)**: Store target weights at execution time. Simulation engine checks drift and schedule triggers at each date, applies sells/buys at historical prices. Same code path for backtest and forward-looking.

**Phase 2 (LLM-Assisted)**: New prompt sends thesis + current state + metrics to LLM, which returns updated target weights. Same `compute_rebalance_trades()` → `execute_rebalance()` pipeline.

---

## Key Design Constraints

- **No persistence** — `st.session_state` only; JSON export/import for manual persistence
- **Security-first imports** — 10 validation checks before any uploaded data enters session state
- **Transaction ledger** — diagnostic expander inside Dashboard (not a main workflow step)
- **Day-0 gating** — same-day executions show position snapshot; performance charts/metrics gated behind elapsed time
- **AI cost is estimated** — static pricing tables, not billed; Ollama Local = $0; Ollama Cloud = $0 (subscription)
- **No live/backtest distinction** — all portfolio evaluation is historical replay from `start_date` to today
- **Chart scalability** — heatmaps for comparisons, overlay limiter for time series, adaptive legends, 10 colors × 4 dashes

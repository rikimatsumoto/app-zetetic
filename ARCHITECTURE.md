# Technical Architecture — Zetetic

## System Overview

Zetetic is a 6-module Streamlit application. A user's investment thesis flows through an LLM pipeline, gets converted into executable trades against Yahoo Finance data, and is analyzed in downloadable reports. All state lives in `st.session_state` — there is no database or filesystem persistence (though portfolios can be exported/imported as JSON for tracking over time).

```
┌─────────────────────────────────────────────────────────────────────┐
│  app.py  (Streamlit UI — 4 main tabs + Guide)                      │
│                                                                     │
│  Tab 1: Thesis ─────────────────────────────────────────────────┐   │
│  │  User writes investment narrative                            │   │
│  │  → strategy_generator.py (Anthropic / xAI / Ollama)         │   │
│  │  → session_state["model_results"] + ["model_usages"]        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 2: Strategies ─────────────────────────────────────────────┐   │
│  │  Review allocations → Execute → JSON export                  │   │
│  │  → market_data → strategies_to_trades → transaction_store    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 3: Dashboard ──────────────────────────────────────────────┐   │
│  │  KPI cards, risk-adjusted metrics (Sharpe/Sortino/Calmar),   │   │
│  │  perf chart ($/% toggle), holdings detail, Excel/JSON        │   │
│  │  export, transaction ledger (diagnostic expander)            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 4: Report ────────────────────────────────────────────────┐   │
│  │  Metrics table, 5 Plotly charts, .md + .xlsx download        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Sidebar: Model config, portfolio settings, reset,                  │
│           multi-file JSON import, AI cost tracker                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `app.py` — UI Controller (~1650 lines)

Entry point. Owns all session state. Contains:
- **Security validation** (`_validate_portfolio_json`): 10 checks on every JSON import (size, UTF-8, structure, types, string sanitization, ticker format, numeric ranges, date validation, nested payload caps)
- **Day-0 detection** (`_is_day_zero`): gates meaningless metrics for same-day execution
- **Multi-file import**: `accept_multiple_files=True` with per-file identity tracking (`_import_file_ids` list + `_import_file_to_keys` dict) so each file can be added or removed independently

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
| `_import_file_ids` | `list[str]` | File identity strings for processed uploads |
| `_import_file_to_keys` | `dict[str, list]` | Maps file_id → [exec_key, ...] for removal |

**Tab 3 Dashboard features:**
- KPI cards with day-0 gating (shows "Day 0 — tracking started" instead of 0% return)
- Risk-adjusted metrics row: Sharpe, Sortino, Calmar ratios with volatility/drawdown captions
- Performance chart with Absolute ($) / Percentage (%) toggle
- Per-portfolio holdings table with Excel + JSON export buttons
- Transaction ledger as collapsed diagnostic expander (not a main workflow step)

**Tab 4 Report features:**
- Metrics comparison table with Sharpe, Sortino, Calmar, Win Rate, Profit Factor columns
- 7 Plotly charts (consistent model→color mapping across all): return comparison bar, risk ratios grouped bar, vol-vs-drawdown scatter, portfolio composition pies, drawdown timeline, rolling Sharpe (30-day), daily return distribution
- Downloadable Markdown + Excel reports

---

### `strategy_generator.py` — LLM Integration (~360 lines)

3 providers: Anthropic (Opus/Sonnet), xAI (Grok 3/Mini), Ollama (local, configurable timeout up to 900s).

---

### `transaction_store.py` — Portfolio State (~240 lines)

In-memory ledger. `clear_all_portfolios()` resets all portfolio + import tracking keys (`_import_file_ids`, `_import_file_to_keys`).

---

### `market_data.py` — Yahoo Finance (~160 lines)

Wraps `yfinance` with MultiIndex compatibility. Live prices, historical series, backtest date resolution.

---

### `report_generator.py` — Report Engine (~900 lines)

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

### `portfolio_manager.py` — Excel Export (~80 lines)

Per-portfolio .xlsx with holdings, gains/losses, and summary stats.

---

## Export / Import Persistence

```
Execute today → 💾 Export JSON → ... days pass ... → 📂 Import JSON(s) → real returns
```

Multi-file uploader supports importing several portfolios simultaneously. Each file is independently validated (size, structure, types, ticker format, date range) before loading. Removing a file from the uploader removes exactly its associated portfolio(s) from session state.

---

## Key Design Constraints

- **No persistence** — `st.session_state` only; JSON export/import for manual persistence
- **Security-first imports** — 10 validation checks before any uploaded data enters session state
- **Transaction ledger** — diagnostic expander inside Dashboard (not a main workflow step)
- **Day-0 gating** — same-day executions show position snapshot; performance charts/metrics gated behind elapsed time
- **AI cost is estimated** — static pricing tables, not billed; Ollama = $0

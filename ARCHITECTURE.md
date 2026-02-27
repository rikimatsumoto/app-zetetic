# Technical Architecture — Zetetic

## System Overview

Zetetic is a 6-module Streamlit application. A user's investment thesis flows through an LLM pipeline, gets converted into executable trades against Yahoo Finance data, and is analyzed in downloadable reports. All state lives in `st.session_state` — there is no database or filesystem persistence (though portfolios can be exported/imported as JSON for tracking over time).

```
┌─────────────────────────────────────────────────────────────────────┐
│  app.py  (Streamlit UI — 4 main tabs + Guide)                      │
│                                                                     │
│  Tab 1: Thesis ─────────────────────────────────────────────────┐   │
│  │  User writes investment narrative                            │   │
│  │  → strategy_generator.py (Anthropic / xAI / Google / Ollama) │   │
│  │  → session_state["model_results"] + ["model_usages"]        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 2: Strategies ─────────────────────────────────────────────┐   │
│  │  Review allocations → Execute → JSON export                  │   │
│  │  → market_data → strategies_to_trades → transaction_store    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 3: Dashboard ──────────────────────────────────────────────┐   │
│  │  KPI cards, risk metrics (Sharpe/Sortino/Calmar/Max DD/      │   │
│  │  Volatility/Win Rate/Profit Factor), perf chart ($/% toggle),│   │
│  │  holdings detail, metric glossary expander, transaction       │   │
│  │  ledger (diagnostic expander)                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Tab 4: Report ────────────────────────────────────────────────┐   │
│  │  Metrics table, 7 Plotly charts, metric glossary expander,   │   │
│  │  .md + .xlsx download                                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Sidebar: Model config (4 providers), portfolio settings, reset,   │
│           multi-file JSON import, AI cost tracker                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `app.py` — UI Controller (~1830 lines)

Entry point. Owns all session state. Contains:
- **Security validation** (`_validate_portfolio_json`): 10 checks on every JSON import (size, UTF-8, structure, types, string sanitization, ticker format, numeric ranges, date validation, nested payload caps)
- **Day-0 detection** (`_is_day_zero`): gates meaningless metrics for same-day execution
- **Multi-file import**: `accept_multiple_files=True` with per-file identity tracking (`_import_file_ids` list + `_import_file_to_keys` dict) so each file can be added or removed independently
- **`_METRIC_GLOSSARY`**: shared markdown constant used by Guide tab, Dashboard expander, and Report expander — single source of truth for all metric definitions with practical "how to read it" guidance
- **Ollama cloud auto-disable**: 3-layer detection (env vars → hostname → localhost probe); Ollama only appears in provider dropdown when locally reachable

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
- Risk-adjusted metrics: 7 cards in 2 rows — Sharpe, Sortino, Calmar / Max DD, Volatility, Win Rate, Profit Factor
- "ℹ️ What do these metrics mean?" expander (collapsed, uses `_METRIC_GLOSSARY`)
- Performance chart with Absolute ($) / Percentage (%) toggle
- Per-portfolio holdings table with Excel + JSON export buttons
- Transaction ledger as collapsed diagnostic expander (not a main workflow step)

**Tab 4 Report features:**
- Metrics comparison table with Volatility, Sharpe, Sortino, Calmar, Win Rate, Profit Factor columns
- "ℹ️ What do these metrics mean?" expander (collapsed, uses `_METRIC_GLOSSARY`)
- 7 Plotly charts (consistent model→color mapping via `color_map`/`legend_map` across all):
  1. Return comparison bar (Total / Annualized / Net)
  2. Risk-adjusted ratios grouped bar (Sharpe / Sortino / Calmar)
  3. Volatility vs Max Drawdown scatter
  4. Portfolio composition donut pies
  5. Drawdown timeline
  6. Rolling Sharpe (30-day window)
  7. Daily return distribution histograms
- Downloadable Markdown + Excel reports

---

### `strategy_generator.py` — LLM Integration (~510 lines)

4 providers: Anthropic (Opus/Sonnet), xAI (Grok 3/Mini), Google (Gemini 2.0 Flash/1.5 Pro), Ollama (local, configurable timeout up to 900s).

- Gemini uses REST API directly (`generativelanguage.googleapis.com`) — no SDK dependency
- `responseMimeType: "application/json"` ensures structured JSON output from Gemini
- Each provider has its own pricing table for cost estimation
- Unified `generate_strategies()` dispatcher routes to the correct provider function

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
- Pre-computes `clean_daily_returns` once (reused by volatility, Sortino, win rate, profit factor)

**`generate_markdown_report()`** — comparative markdown with metrics table (includes Sortino/Calmar/Win Rate/Profit Factor), per-model detail sections, AI-generated narrative

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
- **Gemini via REST** — no `google-generativeai` SDK; `requests` already in dependency list
- **Ollama graceful fallback** — auto-disabled on Streamlit Cloud; silently removed from dropdown, code remains for local use
- **Shared metric glossary** — `_METRIC_GLOSSARY` constant ensures Guide, Dashboard, and Report always show identical definitions

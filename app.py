"""
app.py — Zetetic (v2)
=========================================
Streamlit application with:
  1. Multi-model strategy generation (Anthropic / xAI / Google Gemini / Ollama)
  2. Backtesting with custom start dates using historical prices
  3. Internal transaction tracking system (per-model portfolios)
  4. Multi-step reset confirmation dialogs
  5. Downloadable performance analysis reports (Markdown + Excel)

Run with:  streamlit run app.py
"""

import os
import re
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Local modules ──────────────────────────────────────────────────────────────
from strategy_generator import (
    generate_strategies,
    strategies_to_trades,
    ANTHROPIC_MODELS,
    XAI_MODELS,
    GEMINI_MODELS,
    get_available_ollama_models,
    get_available_ollama_cloud_models,
    _empty_usage,
)
from transaction_store import (
    init_portfolio,
    record_buy,
    record_ai_usage,
    get_portfolio_value,
    get_all_tickers,
    clear_all_portfolios,
    record_snapshot,
)
from market_data import (
    get_current_prices,
    get_prices_at_date,
    get_historical_prices,
    get_benchmark_data,
)
from portfolio_manager import export_portfolio_to_excel
from report_generator import (
    generate_markdown_report,
    generate_excel_report,
    compute_metrics,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS: Day-0 detection & portfolio JSON export
# ══════════════════════════════════════════════════════════════════════════════

def _is_day_zero(portfolio: dict) -> bool:
    """Check if a portfolio was executed today (no time for performance data)."""
    return portfolio.get("start_date") == datetime.now().strftime("%Y-%m-%d")


def _portfolio_to_json(portfolio: dict) -> str:
    """Serialize a portfolio state dict to a downloadable JSON string."""
    # Deep-copy and ensure all values are JSON-serializable
    export = {
        "model_label": portfolio.get("model_label", "Unknown"),
        "provider": portfolio.get("provider", "unknown"),
        "initial_capital": portfolio.get("initial_capital", 0),
        "cash": portfolio.get("cash", 0),
        "start_date": portfolio.get("start_date", ""),
        "strategy_name": portfolio.get("strategy_name", ""),
        "strategy_data": portfolio.get("strategy_data", {}),
        "holdings": portfolio.get("holdings", {}),
        "transactions": portfolio.get("transactions", []),
        "performance_snapshots": portfolio.get("performance_snapshots", []),
        "ai_usage": portfolio.get("ai_usage"),
        "created_at": portfolio.get("created_at", ""),
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return json.dumps(export, indent=2, default=str)


# ── Security: Portfolio JSON validation ───────────────────────────────────────

# Maximum file size (5 MB) — prevents memory exhaustion from large uploads
_MAX_IMPORT_BYTES = 5 * 1024 * 1024

# Allowed top-level keys — reject unexpected fields that could carry payloads
_ALLOWED_KEYS = {
    "model_label", "provider", "initial_capital", "cash", "start_date",
    "strategy_name", "strategy_data", "holdings", "transactions",
    "performance_snapshots", "ai_usage", "created_at", "exported_at",
}

# Required top-level keys
_REQUIRED_KEYS = {
    "model_label", "initial_capital", "cash", "start_date",
    "strategy_name", "holdings", "transactions",
}

# Limits to prevent resource exhaustion
_MAX_HOLDINGS = 500
_MAX_TRANSACTIONS = 10_000
_MAX_SNAPSHOTS = 10_000
_MAX_STRING_LEN = 1_000      # per individual string field
_MAX_TICKER_LEN = 12         # longest real ticker is ~5-6 chars; 12 is generous


def _sanitize_str(value: str, max_len: int = _MAX_STRING_LEN, field: str = "") -> str:
    """
    Sanitize a string value:
    - Enforce type and length
    - Strip control characters (except newline/tab) that could be used for
      log injection, terminal escape attacks, or UI spoofing
    - Strip leading/trailing whitespace
    """
    if not isinstance(value, str):
        raise TypeError(f"Field '{field}' must be a string, got {type(value).__name__}")
    # Remove control chars (U+0000–U+001F) except \n and \t
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', value)
    cleaned = cleaned.strip()
    if len(cleaned) > max_len:
        raise ValueError(f"Field '{field}' exceeds max length ({max_len} chars)")
    return cleaned


def _validate_ticker(ticker: str) -> str:
    """Validate a stock ticker symbol — alphanumeric, dots, hyphens, carets only."""
    t = _sanitize_str(ticker, max_len=_MAX_TICKER_LEN, field="ticker")
    if not re.match(r'^[A-Za-z0-9.\-\^]{1,12}$', t):
        raise ValueError(f"Invalid ticker symbol: '{t}'")
    return t.upper()


def _validate_portfolio_json(raw_bytes: bytes) -> dict:
    """
    Validate and sanitize an uploaded portfolio JSON file.

    Security checks performed:
    1. File size limit (5 MB)
    2. Valid JSON parsing
    3. Top-level structure: only expected keys, all required keys present
    4. Type checking on every field
    5. String sanitization (control chars, length limits)
    6. Numeric range validation (no negative capital, no absurd values)
    7. Date format validation
    8. Ticker symbol format validation
    9. Holdings/transactions structure validation
    10. Array length limits (prevent resource exhaustion)

    Returns:
        Validated and sanitized portfolio dict.

    Raises:
        ValueError/TypeError with a user-friendly message on failure.
    """
    # ── 1. File size ───────────────────────────────────────────────────────
    if len(raw_bytes) > _MAX_IMPORT_BYTES:
        size_mb = len(raw_bytes) / (1024 * 1024)
        raise ValueError(f"File too large ({size_mb:.1f} MB). Maximum is 5 MB.")

    # ── 2. Parse JSON ──────────────────────────────────────────────────────
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        raise ValueError("File is not valid UTF-8 text.")

    if not isinstance(data, dict):
        raise TypeError("Portfolio file must contain a JSON object (not an array or scalar).")

    # ── 3. Key validation ──────────────────────────────────────────────────
    unexpected = set(data.keys()) - _ALLOWED_KEYS
    if unexpected:
        raise ValueError(f"Unexpected fields in file: {', '.join(sorted(unexpected))}")

    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(sorted(missing))}")

    # ── 4–6. Field type & range validation ─────────────────────────────────
    data["model_label"] = _sanitize_str(data["model_label"], field="model_label")
    if not data["model_label"]:
        raise ValueError("model_label cannot be empty.")

    data["strategy_name"] = _sanitize_str(data["strategy_name"], field="strategy_name")
    if not data["strategy_name"]:
        raise ValueError("strategy_name cannot be empty.")

    if "provider" in data:
        data["provider"] = _sanitize_str(data["provider"], max_len=50, field="provider")

    # Numeric fields
    for field in ("initial_capital", "cash"):
        val = data[field]
        if not isinstance(val, (int, float)):
            raise TypeError(f"'{field}' must be a number, got {type(val).__name__}")
        if val < 0:
            raise ValueError(f"'{field}' cannot be negative (got {val})")
        if val > 1e12:  # $1 trillion cap — well beyond any realistic use
            raise ValueError(f"'{field}' value unreasonably large (got {val})")

    # ── 7. Date validation ─────────────────────────────────────────────────
    date_str = _sanitize_str(str(data["start_date"]), max_len=10, field="start_date")
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"'start_date' must be YYYY-MM-DD format, got '{date_str}'")
    # Reject dates in the future or implausibly old
    if parsed_date > datetime.now() + timedelta(days=1):
        raise ValueError("start_date cannot be in the future.")
    if parsed_date.year < 2000:
        raise ValueError("start_date is unreasonably old (before 2000).")
    data["start_date"] = date_str

    # ── 8–9. Holdings validation ───────────────────────────────────────────
    holdings = data["holdings"]
    if not isinstance(holdings, dict):
        raise TypeError("'holdings' must be a JSON object.")
    if len(holdings) > _MAX_HOLDINGS:
        raise ValueError(f"Too many holdings ({len(holdings)}). Maximum is {_MAX_HOLDINGS}.")

    sanitized_holdings = {}
    for ticker, info in holdings.items():
        clean_ticker = _validate_ticker(ticker)
        if not isinstance(info, dict):
            raise TypeError(f"Holding '{clean_ticker}' must be an object.")
        shares = info.get("shares", 0)
        avg_cost = info.get("avg_cost", 0)
        if not isinstance(shares, (int, float)) or shares < 0:
            raise ValueError(f"Invalid shares for {clean_ticker}: {shares}")
        if not isinstance(avg_cost, (int, float)) or avg_cost < 0:
            raise ValueError(f"Invalid avg_cost for {clean_ticker}: {avg_cost}")
        sanitized_holdings[clean_ticker] = {
            "shares": shares,
            "avg_cost": round(avg_cost, 4),
        }
    data["holdings"] = sanitized_holdings

    # ── Transactions validation ────────────────────────────────────────────
    transactions = data["transactions"]
    if not isinstance(transactions, list):
        raise TypeError("'transactions' must be a JSON array.")
    if len(transactions) > _MAX_TRANSACTIONS:
        raise ValueError(f"Too many transactions ({len(transactions)}). Max is {_MAX_TRANSACTIONS}.")

    for idx, txn in enumerate(transactions):
        if not isinstance(txn, dict):
            raise TypeError(f"Transaction #{idx} must be an object.")
        # Sanitize string fields within each transaction
        for str_field in ("id", "action", "ticker", "notes", "timestamp"):
            if str_field in txn:
                txn[str_field] = _sanitize_str(str(txn[str_field]), field=f"txn.{str_field}")
        if "ticker" in txn:
            txn["ticker"] = _validate_ticker(txn["ticker"])
        # Validate numeric fields
        for num_field in ("shares", "price", "total_cost", "cash_before", "cash_after"):
            if num_field in txn:
                val = txn[num_field]
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Transaction #{idx} '{num_field}' must be a number.")

    # ── Performance snapshots ──────────────────────────────────────────────
    if "performance_snapshots" in data:
        snaps = data["performance_snapshots"]
        if not isinstance(snaps, list):
            raise TypeError("'performance_snapshots' must be an array.")
        if len(snaps) > _MAX_SNAPSHOTS:
            data["performance_snapshots"] = snaps[:_MAX_SNAPSHOTS]  # Truncate, don't reject
        for snap in data["performance_snapshots"]:
            if not isinstance(snap, dict):
                raise TypeError("Each snapshot must be an object.")

    # ── AI usage (optional, can be None) ───────────────────────────────────
    if "ai_usage" in data and data["ai_usage"] is not None:
        usage = data["ai_usage"]
        if not isinstance(usage, dict):
            raise TypeError("'ai_usage' must be an object or null.")
        # Sanitize string fields
        for str_field in ("provider", "model"):
            if str_field in usage:
                usage[str_field] = _sanitize_str(str(usage[str_field]), max_len=100, field=f"ai_usage.{str_field}")

    # ── Strategy data (optional nested dict) ─────────────────────────────
    if "strategy_data" in data:
        sd = data["strategy_data"]
        if not isinstance(sd, dict):
            raise TypeError("'strategy_data' must be an object.")
        # Cap serialized size to prevent oversized nested payloads
        sd_size = len(json.dumps(sd))
        if sd_size > 500_000:  # 500 KB
            raise ValueError(f"'strategy_data' too large ({sd_size} bytes). Max is 500 KB.")

    # ── Optional string fields ─────────────────────────────────────────────
    for opt_field in ("created_at", "exported_at"):
        if opt_field in data:
            data[opt_field] = _sanitize_str(str(data[opt_field]), field=opt_field)

    return data
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# METRIC GLOSSARY — reused in Dashboard and Report tabs
# ══════════════════════════════════════════════════════════════════════════════
_METRIC_GLOSSARY = """
| Metric | What it means | How to read it |
|--------|--------------|----------------|
| **Total Return** | Percentage gain/loss since execution. | +15% = grew 15%. Negative = lost money. |
| **Net Return** | Total return minus the AI generation cost — your *true* profit. | Compare to Total Return to see if AI cost is material. |
| **Annualized Return** | Return scaled to a per-year rate. | Makes strategies with different time periods comparable. 8–12% annually is a solid benchmark. |
| **Max Drawdown** | Largest peak-to-trough drop observed. | Lower is better. 10% means the portfolio once fell 10% from its high. Above 30% = aggressive risk. |
| **Volatility** | How much the portfolio swings day-to-day (annualized). | Lower = smoother ride. Under 15% is relatively stable; above 30% is very bumpy. |
| **Sharpe Ratio** | Return per unit of *total* risk (volatility). | Higher is better. Above **1.0** = good; above **2.0** = excellent; below **0** = losing vs risk-free rate. |
| **Sortino Ratio** | Like Sharpe, but only penalises *downside* swings. | Higher is better. Rewards strategies that swing up but not down. A Sortino > Sharpe means most volatility is on the upside. |
| **Calmar Ratio** | Annualized return ÷ max drawdown. | Higher is better. Shows how well you're compensated for the worst dip. Above **1.0** = good; above **3.0** = excellent. |
| **Win Rate** | Percentage of trading days with a positive return. | Higher is better. 50–55% is typical; above 55% shows consistent daily gains. |
| **Profit Factor** | Sum of all gains ÷ sum of all losses. | Above **1.0** = gaining more than losing. Above **1.5** = strong. Below 1.0 = losses outweigh gains. |
"""

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION HELPER — shared by per-strategy and "Execute All" buttons
# ══════════════════════════════════════════════════════════════════════════════

def _execute_strategy(model_label: str, strategy: dict, execution_date: str,
                      initial_capital: float) -> dict:
    """
    Execute a single strategy: fetch prices, create portfolio, record trades.

    Args:
        model_label:     Display name of the model that generated this strategy.
        strategy:        Strategy dict with 'name', 'allocations', etc.
        execution_date:  Date string (YYYY-MM-DD) — today for live, past for backtest.
        initial_capital: Dollar amount to invest.

    Returns:
        dict with keys: 'exec_key', 'ok_count', 'failed', 'missing', 'portfolio'
    """
    exec_key = f"{model_label}__{strategy['name']}"
    tickers = [a["ticker"].upper() for a in strategy["allocations"]]

    # Fetch prices: historical if backtest date is in the past, live if today
    today_str = datetime.now().strftime("%Y-%m-%d")
    if execution_date < today_str:
        prices = get_prices_at_date(tickers, execution_date)
    else:
        prices = get_current_prices(tickers)

    prices = {k: v for k, v in prices.items() if v is not None}
    missing = [t for t in tickers if t not in prices]

    trades = strategies_to_trades(strategy, initial_capital, prices)

    portfolio = init_portfolio(
        model_label=model_label,
        provider=st.session_state.get("model_providers", {}).get(model_label, "unknown"),
        initial_capital=initial_capital,
        start_date=execution_date,
        strategy_name=strategy["name"],
        strategy_data=strategy,
    )

    failed_buys = []
    for t in trades:
        if t.get("error") or t["shares"] == 0:
            continue
        result = record_buy(
            portfolio, t["ticker"], t["shares"], t["price"],
            notes=f"Initial: {t.get('rationale', '')}",
        )
        if isinstance(result, dict) and "error" in result:
            failed_buys.append(f"{t['ticker']}: {result['error']}")

    # Store in session state
    st.session_state["portfolios"][exec_key] = portfolio
    st.session_state["trades_executed"][exec_key] = True
    st.session_state["report_generated"] = False

    # Attach AI usage data to the portfolio
    usage = st.session_state.get("model_usages", {}).get(model_label, {})
    if usage:
        record_ai_usage(portfolio, usage)

    ok_count = sum(1 for t in trades if not t.get("error") and t["shares"] > 0)

    return {
        "exec_key": exec_key,
        "ok_count": ok_count,
        "failed": failed_buys,
        "missing": missing,
        "portfolio": portfolio,
    }


# ── Chart styling constants (shared by Dashboard + Report) ────────────────────

# 10-color palette from Plotly's qualitative set — enough for 3 models × 5 strategies
# Each strategy gets a unique color; no duplicates up to 10 portfolios.
_PALETTE = [
    "#636EFA",  # blue
    "#EF553B",  # red
    "#00CC96",  # green
    "#AB63FA",  # purple
    "#FFA15A",  # orange
    "#19D3F3",  # cyan
    "#FF6692",  # pink
    "#B6E880",  # lime
    "#FF97FF",  # magenta
    "#FECB52",  # yellow
]

# Dash patterns cycle alongside colors so even with >10 strategies,
# lines remain distinguishable (10 colors × 4 dashes = 40 unique combos).
_DASH_PATTERNS = ["solid", "dash", "dot", "dashdot"]


def _adaptive_legend(n_items: int) -> dict:
    """
    Return Plotly legend kwargs that switch from horizontal (compact, ≤7 items)
    to vertical sidebar layout (>7 items) to prevent legend overflow.
    """
    if n_items <= 7:
        return dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    else:
        return dict(
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
            font=dict(size=10),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ZETETIC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .model-card { border: 2px solid #ddd; border-radius: 10px; padding: 16px; margin: 8px 0; }
    .model-opus { border-left: 4px solid #8B5CF6; }
    .model-sonnet { border-left: 4px solid #3B82F6; }
    .model-ollama { border-left: 4px solid #10B981; }
    .model-ollama-cloud { border-left: 4px solid #1D8335; }
    .model-gemini { border-left: 4px solid #EA4335; }
    div[data-testid="stExpander"] { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 8px; }
    .reset-warning { background-color: #FEF2F2; border: 2px solid #EF4444;
                     border-radius: 8px; padding: 16px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "portfolios": {},           # {exec_key: portfolio_state_dict}
    "model_results": {},        # {model_label: raw strategy JSON from LLM}
    "model_usages": {},         # {model_label: usage_dict with tokens/cost}
    "model_providers": {},      # {model_label: provider_key} for execution routing
    "strategies_generated": False,
    "trades_executed": {},      # {exec_key: bool}
    "initial_capital": 10000.0,
    "thesis_text": "",
    "reset_step": 0,           # 0=none, 1=first confirm, 2=final confirm
    "report_generated": False,
    "execution_date": "",       # YYYY-MM-DD — set in Tab 2 (empty = today)
    "_import_file_ids": [],          # list of processed file identity strings
    "_import_file_to_keys": {},      # {file_id: [exec_key, ...]} for removal tracking
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Configuration + Model Selection + Reset
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    st.divider()

    # ── Model Selection (dynamic, up to 3) ─────────────────────────────────
    st.subheader("🤖 Model Selection")
    st.caption("Choose how many models to compare, then configure each one.")

    num_models = st.slider("Number of models to test", min_value=1, max_value=3, value=1)

    # ── Detect cloud deployment (Ollama unavailable) ──────────────────────
    # Streamlit Cloud sets HOSTNAME or runs from /mount; also no local Ollama
    _is_cloud = os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("HOSTNAME", "").startswith("streamlit")
    if not _is_cloud:
        # Final check: probe Ollama to see if it's actually reachable
        _ollama_reachable = len(get_available_ollama_models()) > 0
    else:
        _ollama_reachable = False

    # Provider options — Ollama Local only if locally reachable; Cloud always available
    PROVIDERS = ["Anthropic", "xAI (Grok)", "Google (Gemini)", "Ollama (Cloud)"]
    if _ollama_reachable:
        PROVIDERS.append("Ollama (Local)")

    # Per-provider display → internal key mapping
    _PROVIDER_KEY = {
        "Anthropic": "anthropic",
        "xAI (Grok)": "xai",
        "Google (Gemini)": "google",
        "Ollama (Cloud)": "ollama_cloud",
        "Ollama (Local)": "ollama",
    }

    # Shared state: Ollama URL + timeout + cached model lists (fetched once)
    ollama_url = "http://localhost:11434"
    ollama_timeout = 180
    _ollama_models_cache = None        # populated on first Ollama Local slot
    _ollama_cloud_models_cache = None  # populated on first Ollama Cloud slot

    # Store per-slot API keys alongside the active_models list
    active_models = []     # [(provider_key, display_label), ...]
    api_keys = {}          # {provider_key: api_key_str}

    for slot in range(num_models):
        st.markdown(f"---")
        st.markdown(f"**Model {slot + 1}**")

        # Default provider varies per slot for a nice out-of-box experience
        default_idx = min(slot, len(PROVIDERS) - 1)

        provider_display = st.selectbox(
            f"Provider",
            options=PROVIDERS,
            index=default_idx,
            key=f"provider_{slot}",
        )
        provider_key = _PROVIDER_KEY[provider_display]

        # ── Anthropic slot ─────────────────────────────────────────────
        if provider_key == "anthropic":
            # Only show the API key input once (shared across Anthropic slots)
            if "anthropic" not in api_keys:
                api_keys["anthropic"] = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    help="Get one at console.anthropic.com",
                    key="anthropic_api_key",
                )
            else:
                st.caption("_Using Anthropic key from above_")

            model_name = st.selectbox(
                "Model",
                options=list(ANTHROPIC_MODELS.keys()),
                key=f"anthropic_model_{slot}",
            )
            active_models.append(("anthropic", model_name))

        # ── xAI slot ───────────────────────────────────────────────────
        elif provider_key == "xai":
            if "xai" not in api_keys:
                api_keys["xai"] = st.text_input(
                    "xAI API Key",
                    type="password",
                    help="Get one at console.x.ai",
                    key="xai_api_key",
                )
            else:
                st.caption("_Using xAI key from above_")

            model_name = st.selectbox(
                "Model",
                options=list(XAI_MODELS.keys()),
                key=f"xai_model_{slot}",
            )
            active_models.append(("xai", model_name))

        # ── Google Gemini slot ─────────────────────────────────────────
        elif provider_key == "google":
            if "google" not in api_keys:
                api_keys["google"] = st.text_input(
                    "Google AI API Key",
                    type="password",
                    help="Get a free key at aistudio.google.com/apikey",
                    key="google_api_key",
                )
            else:
                st.caption("_Using Google key from above_")

            model_name = st.selectbox(
                "Model",
                options=list(GEMINI_MODELS.keys()),
                key=f"google_model_{slot}",
            )
            active_models.append(("google", model_name))

        # ── Ollama Cloud slot (always available) ──────────────────────
        elif provider_key == "ollama_cloud":
            if "ollama_cloud" not in api_keys:
                api_keys["ollama_cloud"] = st.text_input(
                    "Ollama Cloud API Key",
                    type="password",
                    help="Get one at ollama.com → Account → API Keys",
                    key="ollama_cloud_api_key",
                )
            else:
                st.caption("_Using Ollama Cloud key from above_")

            # Fetch cloud models once (requires API key)
            cloud_key = api_keys.get("ollama_cloud", "")
            if cloud_key:
                if _ollama_cloud_models_cache is None:
                    with st.spinner("Fetching Ollama Cloud models..."):
                        _ollama_cloud_models_cache = get_available_ollama_cloud_models(cloud_key)

                if _ollama_cloud_models_cache:
                    model_name = st.selectbox(
                        "Model",
                        options=_ollama_cloud_models_cache,
                        key=f"ollama_cloud_model_{slot}",
                        help=f"Found {len(_ollama_cloud_models_cache)} model(s) on ollama.com",
                    )
                    active_models.append(("ollama_cloud", model_name))
                else:
                    st.warning(
                        "No models returned from Ollama Cloud. "
                        "Check your API key and ollama.com account."
                    )
            else:
                st.caption("Enter your API key to see available cloud models.")

        # ── Ollama Local slot (only shown when locally available) ─────
        elif provider_key == "ollama":
            # Show URL + timeout inputs once, share across Ollama slots
            if _ollama_models_cache is None:
                ollama_url = st.text_input(
                    "Ollama URL",
                    value="http://localhost:11434",
                    help="URL of your local Ollama server",
                    key="ollama_url",
                )
                ollama_timeout = st.slider(
                    "Timeout (seconds)",
                    min_value=60, max_value=900, value=180, step=30,
                    help="Increase for large models like DeepSeek-R1. "
                         "Small models (8B) need ~60s, large ones (70B+) may need 5–10 min on a personal device.",
                    key="ollama_timeout",
                )
                _ollama_models_cache = get_available_ollama_models(ollama_url)
            else:
                st.caption(f"_Using Ollama at {ollama_url} · {ollama_timeout}s timeout_")

            if _ollama_models_cache:
                model_name = st.selectbox(
                    "Model",
                    options=_ollama_models_cache,
                    key=f"ollama_model_{slot}",
                    help=f"Found {len(_ollama_models_cache)} model(s) on {ollama_url}",
                )
                active_models.append(("ollama", model_name))
            else:
                st.warning(
                    "No Ollama models detected. Make sure Ollama is running "
                    "(`ollama serve`) and you've pulled at least one model."
                )

    st.markdown("---")
    st.info(f"**{len(active_models)}** model(s) configured")

    st.divider()

    # ── Portfolio Settings ─────────────────────────────────────────────────
    st.subheader("💰 Portfolio Settings")

    initial_capital = st.number_input(
        "Starting Capital ($)",
        min_value=1000.0, max_value=1_000_000.0,
        value=10000.0, step=1000.0,
    )
    st.session_state["initial_capital"] = initial_capital

    benchmark_ticker = st.selectbox(
        "Benchmark Index",
        options=["^GSPC", "^DJI", "^IXIC", "^RUT"],
        format_func=lambda x: {
            "^GSPC": "S&P 500", "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ Composite", "^RUT": "Russell 2000",
        }.get(x, x),
    )

    st.divider()

    # ── Reset (multi-step confirmation) ────────────────────────────────────
    st.subheader("🗑️ Reset")

    reset_step = st.session_state["reset_step"]

    if reset_step == 0:
        if st.button("🔄 Reset All Data", use_container_width=True):
            st.session_state["reset_step"] = 1
            st.rerun()

    elif reset_step == 1:
        st.markdown(
            '<div class="reset-warning">'
            '⚠️ <strong>Are you sure?</strong> This will delete ALL portfolios, '
            'transactions, strategies, and reports.'
            '</div>',
            unsafe_allow_html=True,
        )
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, continue", type="primary", use_container_width=True):
                st.session_state["reset_step"] = 2
                st.rerun()
        with col_no:
            if st.button("Cancel", use_container_width=True):
                st.session_state["reset_step"] = 0
                st.rerun()

    elif reset_step == 2:
        # Final step: actually perform the reset
        clear_all_portfolios(st.session_state)
        st.session_state["reset_step"] = 0
        st.session_state["execution_date"] = ""
        st.toast("All data has been reset.", icon="🗑️")
        st.rerun()

    st.divider()

    # ── Import Saved Portfolio ─────────────────────────────────────────────
    st.subheader("📂 Import Portfolio")
    st.caption("Load one or more previously exported portfolio `.json` files.")

    uploaded_files = st.file_uploader(
        "Upload portfolio JSON(s)",
        type=["json"],
        key="portfolio_import",
        accept_multiple_files=True,
        help="Export portfolios from the Strategies or Dashboard tab, then re-import here.",
    )

    # Build set of current file IDs to detect additions / removals
    current_file_ids = {f"{f.name}_{f.size}" for f in uploaded_files} if uploaded_files else set()
    previous_file_ids = set(st.session_state.get("_import_file_ids", []))

    # ── Handle files removed (user clicked ✕ on a file) ───────────────────
    removed_ids = previous_file_ids - current_file_ids
    if removed_ids:
        file_to_keys = st.session_state.get("_import_file_to_keys", {})
        for fid in removed_ids:
            for ikey in file_to_keys.get(fid, []):
                st.session_state.get("portfolios", {}).pop(ikey, None)
                st.session_state.get("trades_executed", {}).pop(ikey, None)
            file_to_keys.pop(fid, None)
        st.session_state["_import_file_ids"] = list(current_file_ids)
        st.session_state["_import_file_to_keys"] = file_to_keys
        if not st.session_state.get("portfolios"):
            st.session_state["strategies_generated"] = False
        st.session_state["report_generated"] = False
        st.rerun()

    # ── Process newly added files ──────────────────────────────────────────
    added_ids = current_file_ids - previous_file_ids
    if added_ids and uploaded_files:
        new_imports = 0
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in added_ids:
                continue  # Already processed on a previous run

            try:
                raw_bytes = uploaded_file.read()
                import_data = _validate_portfolio_json(raw_bytes)

                exec_key = f"{import_data['model_label']}__{import_data['strategy_name']}"
                base_key = exec_key
                suffix = 2
                while exec_key in st.session_state.get("portfolios", {}):
                    exec_key = f"{base_key} ({suffix})"
                    suffix += 1

                import_data.setdefault("strategy_data", {})
                import_data.setdefault("performance_snapshots", [])
                import_data.setdefault("ai_usage", None)
                import_data.setdefault("provider", "unknown")
                import_data.setdefault("created_at", "imported")

                st.session_state.setdefault("portfolios", {})[exec_key] = import_data
                st.session_state.setdefault("trades_executed", {})[exec_key] = True
                st.session_state["strategies_generated"] = True
                st.session_state["report_generated"] = False

                # Track file→key mapping for per-file removal
                file_to_keys = st.session_state.setdefault("_import_file_to_keys", {})
                file_to_keys.setdefault(file_id, []).append(exec_key)

                provider = import_data.get("provider", "unknown")
                st.session_state.setdefault("model_providers", {})[import_data["model_label"]] = provider
                if import_data.get("ai_usage"):
                    st.session_state.setdefault("model_usages", {})[import_data["model_label"]] = import_data["ai_usage"]

                new_imports += 1
                st.success(
                    f"✅ Imported **{import_data['model_label']} — {import_data['strategy_name']}** "
                    f"(executed {import_data['start_date']}, {len(import_data['holdings'])} holdings)"
                )

            except (ValueError, TypeError) as e:
                st.error(f"❌ **{uploaded_file.name}**: {e}")
            except json.JSONDecodeError:
                st.error(f"❌ **{uploaded_file.name}**: Not valid JSON.")
            except Exception as e:
                st.error(f"❌ **{uploaded_file.name}**: {e}")

        st.session_state["_import_file_ids"] = list(current_file_ids)
        if new_imports > 0:
            st.rerun()

    elif current_file_ids:
        # All files already processed — show count
        n = sum(len(v) for v in st.session_state.get("_import_file_to_keys", {}).values())
        if n:
            st.caption(f"📁 {n} imported portfolio(s) loaded. Remove files above to unload.")

    st.divider()

    # ── AI Cost Tracker (always visible) ───────────────────────────────────
    model_usages = st.session_state.get("model_usages", {})
    if model_usages:
        st.subheader("💰 AI Cost Tracker")
        total_tokens = sum(u.get("total_tokens", 0) for u in model_usages.values())
        total_cost = sum(u.get("estimated_cost_usd", 0) for u in model_usages.values())

        st.metric("Total Tokens", f"{total_tokens:,}")
        st.metric("Total AI Cost", f"${total_cost:.4f}" if total_cost > 0 else "Free")

        for name, u in model_usages.items():
            cost = u.get("estimated_cost_usd", 0)
            tokens = u.get("total_tokens", 0)
            if cost > 0:
                cost_label = f"${cost:.4f}"
            elif u.get("provider") == "ollama_cloud":
                cost_label = "Free (subscription)"
            else:
                cost_label = "Free (local)"
            st.caption(f"**{name}:** {tokens:,} tokens · {cost_label}")

        st.divider()

    st.caption("Built with Streamlit • Anthropic • xAI • Google • Ollama (Local + Cloud) • Yahoo Finance")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title("ZETETIC")
st.markdown("*Multi-model strategy comparison with backtesting & performance analytics*")

tab_guide, tab_thesis, tab_strategies, tab_dashboard, tab_report = st.tabs([
    "📖 Guide", "1️⃣ Thesis", "2️⃣ Strategies", "3️⃣ Dashboard", "4️⃣ Report"
])

# Global execution_date — set by Tab 2 controls, read by Dashboard + Report.
# Falls back to today on first load (before user visits Tab 2).
execution_date = (
    st.session_state["execution_date"]
    if st.session_state["execution_date"]
    else datetime.now().strftime("%Y-%m-%d")
)


# ══════════════════════════════════════════════════════════════════════════════
# GUIDE TAB: BEGINNER TUTORIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.header("Welcome to the investment strategy engine")
    st.markdown(
        "This app turns your investment thesis/ideas into testable portfolios using AI. "
        "Here's how to get started in 5 minutes."
    )

    # ── Quick Start ────────────────────────────────────────────────────────
    st.subheader("Quick start (3 steps)")

    st.markdown("""
**Step 1 — Write a thesis** (Tab 1️⃣)

Describe what you believe about the market in plain English. For example:

> *"I think AI and cloud computing will keep growing for the next 2 years.
> I want exposure to semiconductor and infrastructure companies, but I also
> want some diversification to manage risk."*

Don't worry about being precise - the selected AI will interpret your intent and turn
it into specific stock picks and allocations.

**Step 2 — Review & execute** (Tab 2️⃣)

Each AI model independently generates 3–4 strategies ranked by risk level.
Browse the allocations, compare how different models interpreted your thesis,
and click **"Execute"** on any strategy you'd like to track.

**Step 3 — Monitor & analyze** (Tabs 3️⃣ – 4️⃣)

Once executed, the Dashboard shows live portfolio performance vs. a benchmark,
risk-adjusted metrics (Sharpe, Sortino, Calmar), and a diagnostic transaction ledger.
The Report tab generates a downloadable analysis with comparative charts.
""")

    st.divider()

    # ── Sidebar Setup ─────────────────────────────────────────────────────
    st.subheader("⚙️ Sidebar Setup")
    st.markdown("""
Before generating strategies, configure these settings in the **sidebar** (left panel):

| Setting | What to do |
|---------|-----------|
| **Number of models** | Choose 1–3 models to compare side by side |
| **Provider per slot** | Pick **Anthropic** (Claude), **xAI** (Grok), **Google** (Gemini), **Ollama Cloud** (hosted at ollama.com), or **Ollama Local** (your machine, if available) for each slot |
| **API Keys** | Enter your key for each cloud provider — Anthropic ([console.anthropic.com](https://console.anthropic.com)), xAI ([console.x.ai](https://console.x.ai)), Google ([aistudio.google.com](https://aistudio.google.com/apikey)), or Ollama Cloud ([ollama.com](https://ollama.com)) — only shown when you select that provider |
| **Starting Capital** | Set how much hypothetical money to invest ($1K–$1M) |
| **Benchmark** | Pick an index to compare against (S&P 500 is the default) |
| **Backtest Date** | Set in the **Strategies** tab — choose "Historical date" to simulate investing on a past date and see how it would have performed |
""")

    st.divider()

    # ── Key Concepts ──────────────────────────────────────────────────────
    st.subheader("FAQ")

    with st.expander("What is a thesis?", expanded=False):
        st.markdown(
            "An investment thesis is your belief about where the market is heading. "
            "It doesn't need to be formal, just describe what sectors, trends, or "
            "companies you think will do well (or poorly) and roughly how long you're "
            "planning to hold. The AI translates your narrative into concrete positions."
        )

    with st.expander("What do the risk levels mean?", expanded=False):
        st.markdown("""
Each strategy is tagged with a risk level:

- 🟢 **Conservative** — Mostly large-cap ETFs and blue chips. Lower potential return, lower volatility.
- 🟡 **Moderate** — Mix of ETFs and growth stocks. Balanced risk/reward.
- 🟠 **Aggressive** — Individual growth stocks, concentrated positions. Higher potential but more volatile.
- 🔴 **Speculative** — High-conviction bets on emerging trends. Highest risk.
""")

    with st.expander("What does 'Execute' actually do?", expanded=False):
        st.markdown(
            "Clicking Execute creates a simulated portfolio. It looks up the real stock prices"
            "(live or historical if backtesting) from Yahoo Finance, calculates how many whole shares "
            "you could buy with your starting capital, and records every trade in an internal ledger. "
            "No real money is involved! This is a paper trading simulation."
        )

    with st.expander("How is AI cost tracked?", expanded=False):
        st.markdown(
            "Every time the app calls an AI model, it records how many tokens were used and "
            "estimates the API cost in USD. This appears throughout the app - in the sidebar, "
            "on strategy headers, on dashboard KPI cards, and in reports. The **Net Return** "
            "metric subtracts AI cost from your portfolio profit, so you can see whether the "
            "AI-generated strategy earned more than it cost to create."
        )

    with st.expander("What is backtesting?", expanded=False):
        st.markdown(
            "Backtesting lets you ask \"what if I had invested on a past date?\" In the "
            "**Strategies** tab, switch to \"Historical date\", pick a start date, and execute. "
            "The app will use that date's real closing prices for your initial trades, then "
            "show how the portfolio would have performed from then until today."
        )

    with st.expander("How do I track a 'today' portfolio over time?", expanded=False):
        st.markdown(
            "When you execute a strategy using today's date, there's no price history "
            "to chart yet - it's Day 0. To track it over time:\n\n"
            "1. **Export** the portfolio as a `.json` file (button appears after execution "
            "in both the Strategies and Dashboard tabs)\n"
            "2. Come back days or weeks later\n"
            "3. **Import** the `.json` from the sidebar — the app re-fetches live prices "
            "and calculates real returns, charts, and metrics from your original entry point\n\n"
            "This lets you test whether an AI-generated strategy actually worked."
        )

    st.divider()

    # ── Report Metrics ────────────────────────────────────────────────────
    st.subheader("📊 Understanding Report Metrics")
    st.markdown(_METRIC_GLOSSARY)

    st.divider()

    # ── Tips ──────────────────────────────────────────────────────────────
    st.subheader("💡 Tips")
    st.markdown("""
- **Compare models.** The same thesis often produces very different strategies from Claude vs. Grok vs. a local model. That's the point — you get multiple perspectives.
- **Try backtesting first.** Before trusting a strategy with real decisions, test it on historical data to see if the AI's picks would have actually worked.
- **Check the Report tab.** The comparative narrative explains *why* the models diverged — it's not just numbers, it's analysis.
- **Inspect transactions.** Expand the Transaction Ledger at the bottom of the Dashboard to audit exactly what was bought, at what price, and how cash was allocated.
- **Reset safely.** The reset button requires confirmation, so you won't accidentally delete your work.
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: THESIS INPUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_thesis:
    st.header("Enter your investment thesis")
    st.caption(
        "Describe your market view. The same thesis will be sent to each selected "
        "model independently, so you can compare how different LLMs interpret it."
    )

    thesis = st.text_area(
        "Investment thesis",
        height=200,
        value=st.session_state.get("thesis_text", ""),
        placeholder=(
            "Example: I believe AI infrastructure will see massive growth over the "
            "next 2 years. Companies providing cloud compute, semiconductors, and "
            "AI tooling will outperform. I want exposure to the AI theme while "
            "managing downside risk through diversification."
        ),
    )

    if not active_models:
        st.warning("Please select at least one model in the sidebar.")

    col1, col2 = st.columns([1, 3])
    with col1:
        generate_btn = st.button(
            "Generate strategies",
            type="primary",
            use_container_width=True,
            disabled=(not active_models),
        )

    if generate_btn:
        if not thesis.strip():
            st.error("Please enter an investment thesis.")
        else:
            # Validate API keys for each provider that's in use
            missing_keys = []
            for provider, _ in active_models:
                if provider in ("anthropic", "xai", "google", "ollama_cloud") and not api_keys.get(provider, ""):
                    label = {
                        "anthropic": "Anthropic", "xai": "xAI",
                        "google": "Google", "ollama_cloud": "Ollama Cloud",
                    }.get(provider, provider)
                    if label not in missing_keys:
                        missing_keys.append(label)
            if missing_keys:
                st.error(f"API key required for: {', '.join(missing_keys)}. Enter in the sidebar.")
            else:
                st.session_state["thesis_text"] = thesis
                model_results = {}
                model_usages = {}
                # Map from result key → (provider, model_name) for execution
                model_providers = {}
                progress_bar = st.progress(0, text="Generating strategies...")

                for i, (provider, model_name) in enumerate(active_models):
                    progress_bar.progress(
                        i / len(active_models),
                        text=f"Generating with {model_name}...",
                    )
                    # Build a unique display key (handles duplicate model picks)
                    display_key = model_name
                    if display_key in model_results:
                        display_key = f"{model_name} ({i + 1})"

                    try:
                        result, usage = generate_strategies(
                            thesis=thesis,
                            provider=provider,
                            model_name=model_name,
                            api_key=api_keys.get(provider, ""),
                            ollama_url=ollama_url,
                            ollama_timeout=ollama_timeout,
                        )
                        model_results[display_key] = result
                        model_usages[display_key] = usage
                        model_providers[display_key] = provider
                        st.toast(f"{display_key} complete!", icon="✅")
                    except Exception as e:
                        st.error(f"**{display_key}** failed: {e}")
                        model_results[display_key] = {"error": str(e)}
                        model_usages[display_key] = _empty_usage(provider, model_name)
                        model_providers[display_key] = provider

                progress_bar.progress(1.0, text="Done!")
                st.session_state["model_results"] = model_results
                st.session_state["model_usages"] = model_usages
                st.session_state["model_providers"] = model_providers
                st.session_state["strategies_generated"] = True
                st.session_state["trades_executed"] = {}
                st.session_state["portfolios"] = {}
                st.session_state["report_generated"] = False

                # Show generation cost summary
                total_cost = sum(u.get("estimated_cost_usd", 0) for u in model_usages.values())
                total_tokens = sum(u.get("total_tokens", 0) for u in model_usages.values())
                num_ok = sum(1 for v in model_results.values() if "error" not in v)
                st.success(
                    f"Generated strategies from {num_ok} of {len(active_models)} models. "
                    f"**Tokens used:** {total_tokens:,} | "
                    f"**Estimated AI cost:** ${total_cost:.4f}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: REVIEW & EXECUTE STRATEGIES
# FIX: No st.stop() — use if/else so subsequent tabs still render.
# ══════════════════════════════════════════════════════════════════════════════
with tab_strategies:
    st.header("Review & execute strategies")

    if not st.session_state["strategies_generated"]:
        st.info("Generate strategies in the Thesis tab first.")
    else:
        model_results = st.session_state["model_results"]
        risk_icons = {
            "Conservative": "🟢", "Moderate": "🟡",
            "Aggressive": "🟠", "Speculative": "🔴",
        }

        valid_models = {k: v for k, v in model_results.items() if "error" not in v}

        if not valid_models:
            st.error("No models returned valid strategies. Try regenerating.")
        else:
            # ── Execution date picker (top of tab) ────────────────────────
            st.subheader("📅 Execution Date")
            date_col1, date_col2 = st.columns([1, 2])
            with date_col1:
                date_mode = st.radio(
                    "When to invest?",
                    options=["Today (live prices)", "Historical date (backtest)"],
                    key="date_mode",
                    horizontal=True,
                )
            use_backtest = date_mode.startswith("Historical")

            with date_col2:
                if use_backtest:
                    backtest_date = st.date_input(
                        "Backtest start date",
                        value=datetime.now() - timedelta(days=180),
                        min_value=datetime(2024, 1, 1),
                        max_value=datetime.now() - timedelta(days=1),
                        help="Strategy will be executed using closing prices on this date.",
                        key="backtest_date_input",
                    )
                    execution_date = backtest_date.strftime("%Y-%m-%d")
                else:
                    execution_date = datetime.now().strftime("%Y-%m-%d")
                    st.caption(f"Using today's prices — **{execution_date}**")

            # Persist for downstream tabs (Dashboard, Report)
            st.session_state["execution_date"] = execution_date

            # If user changed the date after executing, offer to clear executions only
            # (preserves LLM-generated strategies — no re-generation / token cost)
            # existing_dates = {
            #     p["start_date"]
            #     for p in st.session_state.get("portfolios", {}).values()
            # }
            # if existing_dates and execution_date not in existing_dates:
            #     st.warning(
            #         f"⚠️ You have portfolios executed on **{', '.join(sorted(existing_dates))}** "
            #         f"but the current date is **{execution_date}**."
            #     )
            #     if st.button(
            #         "🔄 Clear executions & re-execute on new date",
            #         help="Clears all executed portfolios but keeps your AI-generated strategies. "
            #              "No new tokens will be spent — you can re-execute immediately.",
            #         key="reset_executions_btn",
            #     ):
            #         # Clear only execution state — preserve strategies + thesis
            #         st.session_state["portfolios"] = {}
            #         st.session_state["trades_executed"] = {}
            #         st.session_state["report_generated"] = False
            #         # Also clear import tracking (imports were executed on old date)
            #         st.session_state["_import_file_ids"] = []
            #         st.session_state["_import_file_to_keys"] = {}
            #         st.toast("Executions cleared — strategies preserved. Ready to re-execute!", icon="✅")
            #         st.rerun()
            
            if st.button(
                "🔄 Clear executions & re-execute on new date",
                help="Clears all executed portfolios but keeps your AI-generated strategies. "
                "No new tokens will be spent — you can re-execute immediately.",
                key="reset_executions_btn",
                ):
                # Clear only execution state — preserve strategies + thesis
                st.session_state["portfolios"] = {}
                st.session_state["trades_executed"] = {}
                st.session_state["report_generated"] = False
                # Also clear import tracking (imports were executed on old date)
                st.session_state["_import_file_ids"] = []
                st.session_state["_import_file_to_keys"] = {}
                st.toast("Executions cleared — strategies preserved. Ready to re-execute!", icon="✅")
                st.rerun()

            st.divider()

            # ── Collect all unexecuted strategies for "Execute All" ────────
            all_pending = []    # [(model_label, strategy, idx), ...]
            for model_label, result in valid_models.items():
                for idx, strategy in enumerate(result.get("strategies", [])):
                    exec_key = f"{model_label}__{strategy['name']}"
                    if exec_key not in st.session_state.get("trades_executed", {}):
                        all_pending.append((model_label, strategy, idx))

            # ── "Execute All" button ──────────────────────────────────────
            if all_pending:
                total_strats = sum(
                    len(r.get("strategies", []))
                    for r in valid_models.values()
                )
                executed_count = total_strats - len(all_pending)
                st.caption(
                    f"**{executed_count}** of **{total_strats}** strategies executed · "
                    f"**{len(all_pending)}** remaining"
                )

                if st.button(
                    f"🚀 Execute All Remaining ({len(all_pending)} strategies)",
                    type="primary",
                    use_container_width=True,
                    key="exec_all_btn",
                ):
                    progress = st.progress(0, text="Executing strategies...")
                    results_summary = []

                    for i, (ml, strat, _idx) in enumerate(all_pending):
                        progress.progress(
                            i / len(all_pending),
                            text=f"Executing {ml} — {strat['name']}...",
                        )
                        try:
                            res = _execute_strategy(
                                ml, strat, execution_date, initial_capital,
                            )
                            results_summary.append(
                                f"✅ **{ml}** — {strat['name']}: "
                                f"{res['ok_count']} trades"
                            )
                            if res["missing"]:
                                results_summary.append(
                                    f"   ⚠️ No price data: {', '.join(res['missing'])}"
                                )
                            if res["failed"]:
                                results_summary.append(
                                    f"   ⚠️ Failed: {'; '.join(res['failed'])}"
                                )
                        except Exception as e:
                            results_summary.append(
                                f"❌ **{ml}** — {strat['name']}: {e}"
                            )

                    progress.progress(1.0, text="Done!")
                    for line in results_summary:
                        st.markdown(line)
                    st.rerun()
            else:
                st.success("✅ All strategies have been executed.")

            st.divider()

            # ── Per-model strategy cards ──────────────────────────────────
            for model_label, result in valid_models.items():
                provider = st.session_state.get("model_providers", {}).get(model_label, "")
                color = {
                    "anthropic": "#8B5CF6",    # Purple for Claude
                    "xai": "#E44332",          # Red for Grok
                    "google": "#EA4335",       # Google red for Gemini
                    "ollama_cloud": "#1D8335", # Dark green for Ollama Cloud
                    "ollama": "#10B981",       # Green for local
                }.get(provider, "#6B7280")

                st.markdown(
                    f"### <span style='color:{color}'>🤖 {model_label}</span>",
                    unsafe_allow_html=True,
                )

                # Show token usage and cost for this model
                usage = st.session_state.get("model_usages", {}).get(model_label, {})
                if usage.get("total_tokens"):
                    if usage.get("estimated_cost_usd", 0) == 0:
                        cost_str = (
                            "Free (subscription)" if provider == "ollama_cloud"
                            else "Free (local)"
                        )
                    else:
                        cost_str = f"${usage['estimated_cost_usd']:.4f}"
                    st.caption(
                        f"🔢 **Tokens:** {usage['input_tokens']:,} in / "
                        f"{usage['output_tokens']:,} out "
                        f"({usage['total_tokens']:,} total) · "
                        f"💰 **Cost:** {cost_str}"
                    )
                st.caption(f"**Thesis Summary:** {result.get('thesis_summary', 'N/A')}")

                for idx, strategy in enumerate(result.get("strategies", [])):
                    risk = strategy.get("risk_level", "Unknown")
                    icon = risk_icons.get(risk, "⚪")

                    with st.expander(
                        f"{icon} Strategy {idx+1}: {strategy['name']} — {risk} Risk",
                        expanded=(idx == 0),
                    ):
                        st.markdown(f"**Rationale:** {strategy['rationale']}")
                        st.markdown(f"**Time Horizon:** {strategy.get('time_horizon', 'N/A')}")
                        st.markdown(f"**Rebalancing:** {strategy.get('rebalancing_notes', 'N/A')}")

                        # Allocation table
                        alloc_df = pd.DataFrame(strategy["allocations"])
                        alloc_df["weight_display"] = alloc_df["weight"].apply(
                            lambda x: f"{x*100:.1f}%"
                        )
                        display_cols = [c for c in ["ticker", "name", "weight_display", "rationale"]
                                        if c in alloc_df.columns]
                        st.dataframe(alloc_df[display_cols], use_container_width=True, hide_index=True)

                        # Pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=[a["ticker"] for a in strategy["allocations"]],
                            values=[a["weight"] * 100 for a in strategy["allocations"]],
                            hole=0.4, textinfo="label+percent",
                        )])
                        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                                          showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Execute button (individual)
                        exec_key = f"{model_label}__{strategy['name']}"
                        already_executed = exec_key in st.session_state.get("trades_executed", {})

                        if already_executed:
                            st.success("✅ Executed")
                            # Offer JSON export for saving / re-importing later
                            port = st.session_state.get("portfolios", {}).get(exec_key)
                            if port:
                                safe_fn = re.sub(r'[\\/*?:\[\]\s]', '_', exec_key)
                                st.download_button(
                                    "💾 Export Portfolio (.json)",
                                    data=_portfolio_to_json(port),
                                    file_name=f"portfolio_{safe_fn}_{port['start_date']}.json",
                                    mime="application/json",
                                    key=f"export_tab2_{exec_key}",
                                )
                        else:
                            if st.button(
                                f"Execute: {strategy['name']}",
                                key=f"exec_{model_label}_{idx}",
                                use_container_width=True,
                            ):
                                with st.spinner(f"Fetching prices and executing for {model_label}..."):
                                    res = _execute_strategy(
                                        model_label, strategy,
                                        execution_date, initial_capital,
                                    )

                                if res["missing"]:
                                    st.warning(f"Skipping (no price data): {', '.join(res['missing'])}")
                                st.success(
                                    f"Executed {res['ok_count']} trades "
                                    f"for {model_label} — {strategy['name']}"
                                )
                                if res["failed"]:
                                    st.warning(
                                        f"⚠️ {len(res['failed'])} trade(s) failed: "
                                        + "; ".join(res["failed"])
                                    )
                                st.rerun()

                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PORTFOLIO DASHBOARD
# FIX: No st.stop() — conditional rendering only.
# FIX: Snapshot dedup set built once per portfolio (O(n) not O(n²)).
# FIX: Safe benchmark lookup using .asof() for date alignment.
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.header("Portfolio Dashboard")

    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.info("Execute at least one strategy in the Strategies tab to see the dashboard.")
    else:
        # ── Stable color/dash mapping — keyed by portfolio label ─────────
        # Built from the FULL portfolios dict (insertion order), so every
        # label always gets the same color/dash regardless of which subset
        # a particular chart renders.  Report tab builds an identical map
        # from the same dict.
        port_color_map = {}
        port_dash_map = {}
        for i, label in enumerate(portfolios.keys()):
            port_color_map[label] = _PALETTE[i % len(_PALETTE)]
            port_dash_map[label] = _DASH_PATTERNS[i // len(_PALETTE) % len(_DASH_PATTERNS)]

        # ── Batch-fetch all tickers across all portfolios ─────────────────
        all_tickers = set()
        for port in portfolios.values():
            all_tickers.update(get_all_tickers(port))
        all_tickers = list(all_tickers)

        with st.spinner("Refreshing market data..."):
            current_prices = get_current_prices(all_tickers)

        # ── KPI Cards ─────────────────────────────────────────────────────
        st.subheader("Portfolio Overview")
        cols = st.columns(min(len(portfolios), 4))

        # Determine which portfolios have enough history for performance data
        has_history = {}  # {label: bool}
        for label, port in portfolios.items():
            has_history[label] = not _is_day_zero(port)

        any_with_history = any(has_history.values())

        for i, (label, port) in enumerate(portfolios.items()):
            col = cols[i % len(cols)]
            val = get_portfolio_value(port, current_prices)

            with col:
                if has_history[label]:
                    # Has elapsed time — show real return
                    ret = (val - port["initial_capital"]) / port["initial_capital"]
                    ai_cost = (port.get("ai_usage") or {}).get("estimated_cost_usd", 0.0)
                    net_profit = val - port["initial_capital"] - ai_cost
                    net_return = net_profit / port["initial_capital"]

                    st.metric(
                        label=f"**{port['model_label']}**\n{port['strategy_name']}",
                        value=f"${val:,.2f}",
                        delta=f"{ret:+.2%}",
                    )
                    cost_label = "Free" if ai_cost == 0 else f"${ai_cost:.4f}"
                    st.caption(
                        f"Cash: ${port['cash']:,.2f} | "
                        f"Holdings: {len(port['holdings'])} | "
                        f"AI Cost: {cost_label}"
                    )
                    if ai_cost > 0:
                        st.caption(f"Net return (after AI cost): **{net_return:+.2%}**")
                else:
                    # Day 0 — show position snapshot only
                    st.metric(
                        label=f"**{port['model_label']}**\n{port['strategy_name']}",
                        value=f"${val:,.2f}",
                        delta="Day 0 — tracking started",
                        delta_color="off",
                    )
                    st.caption(
                        f"Cash: ${port['cash']:,.2f} | "
                        f"Holdings: {len(port['holdings'])} | "
                        f"Executed today"
                    )
                    st.info(
                        "📅 Performance data will appear once the market "
                        "moves from your entry prices. Export this portfolio "
                        "and re-import it later to track returns.",
                        icon="💡",
                    )

        # ── Performance Chart ─────────────────────────────────────────────
        if any_with_history:
            st.subheader("Performance vs Benchmark")

            # Toggle: absolute dollar value vs percentage return
            chart_mode = st.radio(
                "Chart display",
                options=["Absolute ($)", "Percentage (%)"],
                horizontal=True,
                key="chart_mode",
            )

            start_dates = [p["start_date"] for p in portfolios.values()]
            earliest_start = min(start_dates) if start_dates else execution_date

            with st.spinner("Loading historical data for chart..."):
                bench_data = get_benchmark_data(benchmark_ticker, earliest_start)
                hist_prices = (
                    get_historical_prices(all_tickers, earliest_start)
                    if all_tickers else pd.DataFrame()
                )

            fig = go.Figure()

            # Prepare benchmark series (used for both chart and snapshots)
            # Use portfolios' stored initial_capital for normalization (not the
            # sidebar widget, which the user may have changed after execution)
            port_capitals = [p["initial_capital"] for p in portfolios.values()]
            chart_capital = port_capitals[0] if port_capitals else initial_capital

            bench_close = pd.Series(dtype=float)
            if not bench_data.empty:
                bench_close = bench_data["Close"]
                if isinstance(bench_close, pd.DataFrame):
                    bench_close = bench_close.iloc[:, 0]

                bench_start_val = float(bench_close.iloc[0])
                if bench_start_val > 0:
                    if chart_mode == "Percentage (%)":
                        bench_plot = ((bench_close / bench_start_val) - 1) * 100
                    else:
                        bench_plot = (bench_close / bench_start_val) * chart_capital

                    fig.add_trace(go.Scatter(
                        x=bench_plot.index, y=bench_plot.values,
                        mode="lines", name=f"Benchmark ({benchmark_ticker})",
                        line=dict(color="#9CA3AF", width=2, dash="dash"),
                    ))

            # Plot each portfolio's historical value
            if not hist_prices.empty:
                port_count = sum(1 for _, p in portfolios.items() if not _is_day_zero(p))
                for i, (label, port) in enumerate(portfolios.items()):
                    if _is_day_zero(port):
                        continue  # Skip day-0 portfolios in chart
                    color = port_color_map[label]
                    dash = port_dash_map[label]
                    port_series = pd.Series(index=hist_prices.index, dtype=float)

                    for date in hist_prices.index:
                        daily_val = port["cash"]
                        for ticker, info in port["holdings"].items():
                            if ticker in hist_prices.columns:
                                try:
                                    p = hist_prices.loc[date, ticker]
                                    # FIX: Handle potential Series from MultiIndex
                                    if hasattr(p, 'iloc'):
                                        p = p.iloc[0]
                                    if pd.notna(p):
                                        daily_val += info["shares"] * float(p)
                                except (KeyError, TypeError, IndexError):
                                    pass
                        port_series[date] = daily_val

                    # FIX: Record snapshots — build dedup set ONCE per portfolio (O(n))
                    if not bench_close.empty:
                        existing_dates = {
                            s["date"] for s in port.get("performance_snapshots", [])
                        }
                        for date in port_series.index:
                            date_str = date.strftime("%Y-%m-%d")
                            if date_str not in existing_dates:
                                # FIX: Safe benchmark lookup — use .asof() for nearest-date
                                # .asof() returns NaN if date is before all benchmark data
                                try:
                                    bench_val = bench_close.asof(date)
                                    if pd.isna(bench_val):
                                        bench_val = 0.0
                                    else:
                                        bench_val = float(bench_val)
                                except Exception:
                                    bench_val = 0.0
                                record_snapshot(
                                    port, float(port_series[date]), bench_val, date_str
                                )
                                existing_dates.add(date_str)  # Keep set in sync

                    fig.add_trace(go.Scatter(
                        x=port_series.index,
                        y=(((port_series / port["initial_capital"]) - 1) * 100).values
                            if chart_mode == "Percentage (%)" else port_series.values,
                        mode="lines",
                        name=f"{port['model_label']}: {port['strategy_name']}",
                        line=dict(color=color, width=2.5, dash=dash),
                    ))

            if chart_mode == "Percentage (%)":
                fig.add_hline(y=0, line_dash="dot", line_color="gray",
                              annotation_text="Breakeven")
                y_title = "Return (%)"
            else:
                fig.add_hline(y=chart_capital, line_dash="dot", line_color="gray",
                              annotation_text=f"Initial: ${chart_capital:,.0f}")
                y_title = "Value ($)"

            fig.update_layout(
                title="Portfolio Value Over Time",
                yaxis_title=y_title, xaxis_title="Date",
                height=500, template="plotly_white",
                legend=_adaptive_legend(port_count + 1),  # +1 for benchmark
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # All portfolios are day-0 — explain what happens next
            st.subheader("📅 Performance Tracking")
            st.info(
                "All portfolios were executed today. Performance charts and return metrics "
                "will appear once prices move from your entry point.\n\n"
                "**What to do now:**\n"
                "- Review your holdings below\n"
                "- **Export each portfolio as JSON** (button below each holding)\n"
                "- Come back later and **import the JSON** from the sidebar to see real returns",
                icon="⏳",
            )

        # ── Risk-Adjusted Metrics ──────────────────────────────────────────
        if any_with_history:
            st.subheader("Risk-Adjusted Metrics")
            history_ports = [(l, p) for l, p in portfolios.items() if has_history[l]]

            # One full-width row per strategy — avoids nested columns truncation
            for label, port in history_ports:
                m = compute_metrics(port, current_prices)
                st.markdown(f"**{port['model_label']}** · {port['strategy_name']}")

                # All 7 metrics in a single row across the full page width
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                c1.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
                c2.metric("Sortino", f"{m['sortino_ratio']:.2f}")
                c3.metric("Calmar", f"{m['calmar_ratio']:.2f}")
                c4.metric("Max DD", f"{m['max_drawdown']:.1%}")
                c5.metric("Volatility", f"{m['volatility']:.1%}")
                c6.metric("Win Rate", f"{m['win_rate']:.0%}")
                c7.metric("Profit Factor", f"{m['profit_factor']:.2f}")

            # Collapsed glossary for metric explanations
            with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                st.markdown(_METRIC_GLOSSARY)

        # ── Per-portfolio Holdings ────────────────────────────────────────
        st.subheader("Holdings Detail")

        for label, port in portfolios.items():
            with st.expander(
                f"📋 {port['model_label']} — {port['strategy_name']}", expanded=True
            ):
                rows = []
                for ticker, info in port["holdings"].items():
                    price = current_prices.get(ticker, 0) or 0
                    mkt_val = info["shares"] * price
                    gain = mkt_val - (info["shares"] * info["avg_cost"])
                    ret = (
                        (price - info["avg_cost"]) / info["avg_cost"]
                        if info["avg_cost"] > 0 else 0
                    )
                    rows.append({
                        "Ticker": ticker,
                        "Shares": int(info["shares"]),
                        "Avg Cost": f"${info['avg_cost']:.2f}",
                        "Current": f"${price:.2f}" if current_prices.get(ticker) is not None else "N/A",
                        "Value": f"${mkt_val:,.2f}",
                        "G/L": f"${gain:+,.2f}",
                        "Return": f"{ret:+.2%}",
                    })
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows), use_container_width=True, hide_index=True
                    )

                # Download buttons: Excel + JSON export
                safe_label = re.sub(r'[\\/*?:\[\]\s]', '_', label)
                excel_path = f"/tmp/portfolio_{safe_label}.xlsx"
                export_portfolio_to_excel(port, current_prices, excel_path)

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            f"📥 Excel",
                            data=f.read(),
                            file_name=(
                                f"portfolio_{port['model_label'].replace(' ', '_')}_"
                                f"{port['strategy_name'].replace(' ', '_')}.xlsx"
                            ),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_{label}",
                            use_container_width=True,
                        )
                with dl_col2:
                    st.download_button(
                        "💾 Export JSON",
                        data=_portfolio_to_json(port),
                        file_name=f"portfolio_{safe_label}_{port['start_date']}.json",
                        mime="application/json",
                        key=f"json_{label}",
                        use_container_width=True,
                    )

        # ── Transaction Ledger (diagnostic — collapsed by default) ────────
        st.divider()
        with st.expander("🔍 Transaction Ledger (diagnostic)", expanded=False):
            all_txns = []
            for label, port in portfolios.items():
                for txn in port.get("transactions", []):
                    all_txns.append({
                        "Model": port["model_label"],
                        "Strategy": port["strategy_name"],
                        "Date": txn["timestamp"],
                        "Action": txn["action"],
                        "Ticker": txn["ticker"],
                        "Shares": txn["shares"],
                        "Price": f"${txn['price']:.2f}",
                        "Total": f"${txn['total_cost']:,.2f}",
                        "Cash After": f"${txn['cash_after']:,.2f}",
                        "Notes": txn.get("notes", ""),
                    })
            if all_txns:
                st.dataframe(
                    pd.DataFrame(all_txns), use_container_width=True, hide_index=True
                )
                st.caption(f"**Total transactions:** {len(all_txns)}")
            else:
                st.caption("No transactions recorded yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PERFORMANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.header("Performance Analysis Report")

    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.info("Execute strategies first to generate a performance report.")
    else:
        # Separate portfolios with history from day-0 portfolios
        ports_with_history = {k: v for k, v in portfolios.items() if not _is_day_zero(v)}
        ports_day_zero = {k: v for k, v in portfolios.items() if _is_day_zero(v)}

        if ports_day_zero:
            day0_names = ", ".join(
                f"**{v['model_label']}**" for v in ports_day_zero.values()
            )
            if ports_with_history:
                st.warning(
                    f"📅 {day0_names} executed today — no performance data yet. "
                    f"The report below covers only portfolios with elapsed history."
                )
            else:
                st.info(
                    f"📅 All portfolios were executed today. Performance metrics require "
                    f"at least one day of price movement.\n\n"
                    f"**Export your portfolios as JSON** from the Dashboard tab, then "
                    f"re-import them later to generate a full performance report.",
                    icon="⏳",
                )

        # Use only portfolios with history for the report (or all if backtested)
        report_portfolios = ports_with_history if ports_with_history else {}

        if report_portfolios:
            # Collect prices
            all_tickers_rpt = set()
            for port in report_portfolios.values():
                all_tickers_rpt.update(get_all_tickers(port))
            rpt_prices = get_current_prices(list(all_tickers_rpt)) if all_tickers_rpt else {}

            thesis_text = st.session_state.get("thesis_text", "No thesis provided.")

            # Auto-generate report — regenerates on each visit with latest prices
            with st.spinner("Computing metrics and building report..."):
                md_report = generate_markdown_report(
                    thesis=thesis_text,
                    portfolios=report_portfolios,
                    all_prices=rpt_prices,
                    benchmark_ticker=benchmark_ticker,
                    initial_capital=initial_capital,
                    start_date=execution_date,
                )
                with open("/tmp/report.md", "w") as f:
                    f.write(md_report)

                generate_excel_report(
                    thesis=thesis_text,
                    portfolios=report_portfolios,
                    all_prices=rpt_prices,
                    benchmark_ticker=benchmark_ticker,
                    initial_capital=initial_capital,
                    start_date=execution_date,
                    filepath="/tmp/report.xlsx",
                )
                st.session_state["report_generated"] = True

            if st.session_state.get("report_generated"):
                # ── Inline metrics ────────────────────────────────────────
                st.subheader("Key Metrics Comparison")

                all_rpt_metrics = {}
                metrics_table = []
                for label, port in report_portfolios.items():
                    m = compute_metrics(port, rpt_prices)
                    all_rpt_metrics[label] = m
                    cost_str = (
                        "Free" if m["ai_cost_usd"] == 0
                        else f"${m['ai_cost_usd']:.4f}"
                    )
                    metrics_table.append({
                        "Model": port["model_label"],
                        "Strategy": port["strategy_name"],
                        "Current Value": f"${m['current_value']:,.2f}",
                        "Total Return": f"{m['total_return_pct']:+.2%}",
                        "Net Return": f"{m['net_return_pct']:+.2%}",
                        "Ann. Return": f"{m['annualized_return']:+.2%}",
                        "Max Drawdown": f"{m['max_drawdown']:.2%}",
                        "Volatility": f"{m['volatility']:.1%}",
                        "Sharpe": f"{m['sharpe_ratio']:.2f}",
                        "Sortino": f"{m['sortino_ratio']:.2f}",
                        "Calmar": f"{m['calmar_ratio']:.2f}",
                        "Win Rate": f"{m['win_rate']:.0%}",
                        "Profit Factor": f"{m['profit_factor']:.2f}",
                        "AI Cost": cost_str,
                        "Best": f"{m['best_holding']} ({m['best_holding_return']:+.1%})",
                        "Worst": f"{m['worst_holding']} ({m['worst_holding_return']:+.1%})",
                    })
                st.dataframe(
                    pd.DataFrame(metrics_table), use_container_width=True, hide_index=True
                )

                # Collapsed glossary for metric explanations
                with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                    st.markdown(_METRIC_GLOSSARY)

                # ── Visual Charts ─────────────────────────────────────────
                if len(all_rpt_metrics) > 0:
                    st.subheader("Performance Charts")

                    # ── Build color/dash/legend maps from shared constants ──
                    # Color/dash assigned from FULL portfolios dict (same
                    # insertion order as Dashboard tab) so every label gets
                    # the same color across both tabs.
                    all_portfolios = st.session_state.get("portfolios", {})
                    rpt_labels = list(all_rpt_metrics.keys())
                    n_strats = len(rpt_labels)
                    color_map = {}   # {exec_key: hex}
                    dash_map = {}    # {exec_key: dash pattern}
                    legend_map = {}  # {exec_key: "Model: Strategy"}
                    for i, label in enumerate(all_portfolios.keys()):
                        color_map[label] = _PALETTE[i % len(_PALETTE)]
                        dash_map[label] = _DASH_PATTERNS[i // len(_PALETTE) % len(_DASH_PATTERNS)]
                    for label in rpt_labels:
                        port = report_portfolios[label]
                        legend_map[label] = f"{port['model_label']}: {port['strategy_name']}"

                    # ── Chart 1: Return Comparison (heatmap) ──────────────────
                    return_metrics = [
                        ("Total Return %", "total_return_pct"),
                        ("Ann. Return %", "annualized_return"),
                        ("Net Return %", "net_return_pct"),
                    ]
                    # Build matrix: rows = strategies, cols = metrics
                    ret_z = []
                    ret_text = []
                    ret_y_labels = []
                    for label in rpt_labels:
                        m = all_rpt_metrics[label]
                        row_vals = [m[rm[1]] * 100 for rm in return_metrics]
                        ret_z.append(row_vals)
                        ret_text.append([f"{v:+.2f}%" for v in row_vals])
                        ret_y_labels.append(legend_map[label])

                    fig_returns = go.Figure(data=go.Heatmap(
                        z=ret_z,
                        x=[rm[0] for rm in return_metrics],
                        y=ret_y_labels,
                        text=ret_text,
                        texttemplate="%{text}",
                        textfont=dict(size=13),
                        colorscale="RdYlGn",
                        zmid=0,
                        colorbar=dict(title="%"),
                        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
                    ))
                    fig_returns.update_layout(
                        title="Return Comparison",
                        height=max(250, 50 * n_strats + 100),
                        template="plotly_white",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=250),
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)

                    # ── Chart 2: Risk-Adjusted Ratios (heatmap) ───────────────
                    ratio_keys = [
                        ("Sharpe", "sharpe_ratio"),
                        ("Sortino", "sortino_ratio"),
                        ("Calmar", "calmar_ratio"),
                        ("Win Rate %", "win_rate"),
                        ("Profit Factor", "profit_factor"),
                    ]
                    risk_z = []
                    risk_text = []
                    risk_y_labels = []
                    for label in rpt_labels:
                        m = all_rpt_metrics[label]
                        row_vals = []
                        row_text = []
                        for rk_name, rk_key in ratio_keys:
                            val = m[rk_key]
                            if rk_key == "win_rate":
                                val_display = val * 100  # Normalize to % for heatmap
                                row_text.append(f"{val:.0%}")
                            else:
                                val_display = val
                                row_text.append(f"{val:.2f}")
                            row_vals.append(val_display)
                        risk_z.append(row_vals)
                        risk_text.append(row_text)
                        risk_y_labels.append(legend_map[label])

                    fig_risk = go.Figure(data=go.Heatmap(
                        z=risk_z,
                        x=[rk[0] for rk in ratio_keys],
                        y=risk_y_labels,
                        text=risk_text,
                        texttemplate="%{text}",
                        textfont=dict(size=13),
                        colorscale="RdYlGn",
                        colorbar=dict(title="Value"),
                        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
                    ))
                    fig_risk.update_layout(
                        title="Risk-Adjusted Ratios",
                        height=max(250, 50 * n_strats + 100),
                        template="plotly_white",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=250),
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                    # ── Chart 3: Volatility vs Max Drawdown (scatter) ─────────
                    fig_vol = go.Figure()
                    for label in rpt_labels:
                        m = all_rpt_metrics[label]
                        fig_vol.add_trace(go.Scatter(
                            x=[m["volatility"] * 100],
                            y=[m["max_drawdown"] * 100],
                            mode="markers+text",
                            name=legend_map[label],
                            text=[legend_map[label].split(":")[0]],
                            textposition="top center",
                            marker=dict(size=16, color=color_map[label]),
                        ))
                    fig_vol.update_layout(
                        title="Risk Profile: Volatility vs Max Drawdown",
                        xaxis_title="Annualized Volatility (%)",
                        yaxis_title="Max Drawdown (%)",
                        height=400, template="plotly_white",
                        legend=_adaptive_legend(n_strats),
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                    # ── Chart 4: Portfolio Composition (pies) ─────────────────
                    pie_cols = st.columns(min(len(rpt_labels), 3))
                    for i, label in enumerate(rpt_labels):
                        port = report_portfolios[label]
                        tickers = list(port["holdings"].keys())
                        values = []
                        for t in tickers:
                            price = rpt_prices.get(t, 0) or 0
                            values.append(port["holdings"][t]["shares"] * price)
                        if port["cash"] > 0:
                            tickers.append("Cash")
                            values.append(port["cash"])
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=tickers, values=values,
                            hole=0.4, textinfo="label+percent",
                        )])
                        fig_pie.update_layout(
                            title=legend_map[label],
                            height=350, showlegend=False,
                            margin=dict(t=40, b=10, l=10, r=10),
                        )
                        with pie_cols[i % len(pie_cols)]:
                            st.plotly_chart(fig_pie, use_container_width=True)

                    # ── Overlay chart strategy selector ────────────────────────
                    # For charts 5-7 (overlay/line charts), let user pick which
                    # strategies to display — default top 5 by total return.
                    has_snapshots = any(
                        len(report_portfolios[l].get("performance_snapshots", [])) > 2
                        for l in rpt_labels
                    )
                    if has_snapshots:
                        # Rank by total return, default to top 5
                        ranked_labels = sorted(
                            rpt_labels,
                            key=lambda l: all_rpt_metrics[l]["total_return_pct"],
                            reverse=True,
                        )
                        default_top = ranked_labels[:5]

                        if n_strats > 5:
                            st.markdown("---")
                            st.markdown(
                                "**Overlay charts** — select up to 5 strategies to display "
                                "(defaulting to top 5 by total return):"
                            )
                            overlay_selection = st.multiselect(
                                "Strategies to overlay",
                                options=[legend_map[l] for l in ranked_labels],
                                default=[legend_map[l] for l in default_top],
                                max_selections=5,
                                key="overlay_chart_selection",
                            )
                            # Map display names back to keys
                            reverse_legend = {v: k for k, v in legend_map.items()}
                            overlay_labels = [
                                reverse_legend[name]
                                for name in overlay_selection
                                if name in reverse_legend
                            ]
                        else:
                            # ≤5 strategies: show all, no selector needed
                            overlay_labels = rpt_labels

                        n_overlay = len(overlay_labels)

                        # ── Chart 5: Drawdown Timeline ────────────────────────
                        fig_dd = go.Figure()
                        for label in overlay_labels:
                            port = report_portfolios[label]
                            snaps = port.get("performance_snapshots", [])
                            if len(snaps) < 2:
                                continue
                            vals = pd.Series(
                                [s["portfolio_value"] for s in snaps],
                                index=pd.to_datetime([s["date"] for s in snaps]),
                            )
                            peak = vals.cummax()
                            drawdown = ((vals - peak) / peak) * 100
                            fig_dd.add_trace(go.Scatter(
                                x=drawdown.index, y=drawdown.values,
                                mode="lines", fill="tozeroy",
                                name=legend_map[label],
                                line=dict(
                                    color=color_map[label],
                                    dash=dash_map[label],
                                ),
                            ))
                        fig_dd.update_layout(
                            title="Drawdown Over Time",
                            yaxis_title="Drawdown (%)", xaxis_title="Date",
                            height=400, template="plotly_white",
                            legend=_adaptive_legend(n_overlay),
                        )
                        st.plotly_chart(fig_dd, use_container_width=True)

                        # ── Chart 6: Rolling Sharpe (30-day window) ───────────
                        fig_rs = go.Figure()
                        has_rolling_data = False
                        for label in overlay_labels:
                            port = report_portfolios[label]
                            snaps = port.get("performance_snapshots", [])
                            if len(snaps) < 30:
                                continue
                            vals = pd.Series(
                                [float(s["portfolio_value"]) for s in snaps],
                                index=pd.to_datetime([s["date"] for s in snaps]),
                            )
                            rets = vals.pct_change().dropna()
                            rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
                            if len(rets) < 30:
                                continue
                            daily_rf = (1 + 0.05) ** (1/252) - 1
                            excess = rets - daily_rf
                            roll_mean = excess.rolling(30).mean()
                            roll_std = excess.rolling(30).std()
                            rolling_sharpe = (roll_mean / roll_std) * np.sqrt(252)
                            rolling_sharpe = rolling_sharpe.dropna()
                            if len(rolling_sharpe) > 0:
                                has_rolling_data = True
                                fig_rs.add_trace(go.Scatter(
                                    x=rolling_sharpe.index,
                                    y=rolling_sharpe.values,
                                    mode="lines",
                                    name=legend_map[label],
                                    line=dict(
                                        color=color_map[label], width=2,
                                        dash=dash_map[label],
                                    ),
                                ))
                        if has_rolling_data:
                            fig_rs.add_hline(y=0, line_dash="dot",
                                             line_color="gray")
                            fig_rs.update_layout(
                                title="Rolling Sharpe Ratio (30-day window)",
                                yaxis_title="Sharpe Ratio",
                                xaxis_title="Date",
                                height=400, template="plotly_white",
                                legend=_adaptive_legend(n_overlay),
                            )
                            st.plotly_chart(fig_rs, use_container_width=True)

                        # ── Chart 7: Daily Return Distribution ────────────────
                        fig_dist = go.Figure()
                        has_dist_data = False
                        for label in overlay_labels:
                            port = report_portfolios[label]
                            snaps = port.get("performance_snapshots", [])
                            if len(snaps) < 5:
                                continue
                            vals = pd.Series(
                                [float(s["portfolio_value"]) for s in snaps],
                            )
                            rets = vals.pct_change().dropna()
                            rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
                            if len(rets) > 0:
                                has_dist_data = True
                                fig_dist.add_trace(go.Histogram(
                                    x=rets.values * 100,
                                    name=legend_map[label],
                                    marker_color=color_map[label],
                                    opacity=0.65,
                                    nbinsx=40,
                                ))
                        if has_dist_data:
                            fig_dist.add_vline(x=0, line_dash="dot",
                                               line_color="gray")
                            fig_dist.update_layout(
                                title="Daily Return Distribution",
                                xaxis_title="Daily Return (%)",
                                yaxis_title="Frequency",
                                barmode="overlay",
                                height=400, template="plotly_white",
                                legend=_adaptive_legend(n_overlay),
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                # ── Downloads ─────────────────────────────────────────────
                st.subheader("Download Reports")
                col_md, col_xl = st.columns(2)

                if os.path.exists("/tmp/report.md"):
                    with col_md:
                        with open("/tmp/report.md", "r") as f:
                            st.download_button(
                                "📄 Download Markdown Report",
                                data=f.read(),
                                file_name=f"strategy_report_{datetime.now().strftime('%Y%m%d')}.md",
                                mime="text/markdown",
                                use_container_width=True,
                            )

                if os.path.exists("/tmp/report.xlsx"):
                    with col_xl:
                        with open("/tmp/report.xlsx", "rb") as f:
                            st.download_button(
                                "📊 Download Excel Report",
                                data=f.read(),
                                file_name=f"strategy_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                            )

                # ── Preview ───────────────────────────────────────────────
                if os.path.exists("/tmp/report.md"):
                    with st.expander("📖 Preview Report", expanded=False):
                        with open("/tmp/report.md", "r") as f:
                            st.markdown(f.read())


# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for learning and experimentation only. "
    "Always consult a qualified financial advisor before making real investment "
    "decisions. The strategy is automatically generated from large language models "
    "and may contain errors, omissions, or outdated information. It is provided “as is,” "
    "without warranties of any kind, express or implied. This application is not investment, "
    "legal, security, or policy advice and must not be relied upon for decision-making. "
    "You are responsible for independently verifying facts and conclusions with primary sources "
    "and qualified professionals. Neither the author nor providers of this service accept any "
    "liability for losses or harms arising from use of this content. No duty to update is assumed."
    )

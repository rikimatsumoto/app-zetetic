"""
transaction_store.py
====================
Internal transaction tracking system.

Maintains a per-model portfolio state in memory (mirrored to session state)
with full transaction history. Each "model run" is an independent portfolio
that tracks its own holdings, cash, and transaction log.

Data structure (stored in st.session_state["portfolios"]):
{
    "Model Name": {
        "model_label":    "Claude Opus 4",
        "provider":       "anthropic",
        "initial_capital": 10000.0,
        "cash":           2345.67,
        "start_date":     "2024-06-01",
        "strategy_name":  "AI Infrastructure Growth",
        "strategy_data":  { ... full strategy dict ... },
        "holdings": {
            "AAPL": {"shares": 10, "avg_cost": 185.50},
            ...
        },
        "transactions": [
            {
                "id":         "txn_001",
                "timestamp":  "2024-06-01 09:30:00",
                "action":     "BUY",
                "ticker":     "AAPL",
                "shares":     10,
                "price":      185.50,
                "total_cost": 1855.00,
                "cash_before": 10000.00,
                "cash_after":  8145.00,
                "notes":      "Initial allocation — conservative AI play",
            },
            ...
        ],
        "performance_snapshots": [
            {"date": "2024-06-01", "portfolio_value": 10000.0, "benchmark_value": 5200.0},
            ...
        ],
    }
}
"""

import uuid
from datetime import datetime
from typing import Optional


def _gen_txn_id() -> str:
    """Generate a short unique transaction ID."""
    return f"txn_{uuid.uuid4().hex[:8]}"


def init_portfolio(model_label: str, provider: str, initial_capital: float,
                   start_date: str, strategy_name: str, strategy_data: dict) -> dict:
    """
    Create a fresh portfolio state dict for a model run.

    Args:
        model_label:     Human-friendly name, e.g. "Claude Opus 4"
        provider:        "anthropic" or "ollama"
        initial_capital: Starting cash ($)
        start_date:      Execution / backtest date "YYYY-MM-DD"
        strategy_name:   Name of the selected strategy
        strategy_data:   The full strategy dict from the LLM

    Returns:
        Portfolio state dict ready to be stored in session state.
    """
    return {
        "model_label": model_label,
        "provider": provider,
        "initial_capital": initial_capital,
        "cash": initial_capital,
        "start_date": start_date,
        "strategy_name": strategy_name,
        "strategy_data": strategy_data,
        "holdings": {},
        "transactions": [],
        "performance_snapshots": [],
        "ai_usage": None,  # Populated by record_ai_usage() after generation
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def record_ai_usage(portfolio: dict, usage: dict):
    """
    Store the AI token usage and cost data in the portfolio.

    Args:
        portfolio: Portfolio state dict (mutated in place).
        usage:     Dict from strategy_generator with keys:
                   provider, model, input_tokens, output_tokens,
                   total_tokens, estimated_cost_usd.
    """
    portfolio["ai_usage"] = usage


def record_buy(portfolio: dict, ticker: str, shares: float, price: float,
               notes: str = "") -> dict:
    """
    Record a BUY transaction in the portfolio's internal ledger.

    Updates:
      - portfolio["cash"] (decreased)
      - portfolio["holdings"][ticker] (created or averaged up)
      - portfolio["transactions"] (appended)

    Args:
        portfolio: The portfolio state dict (mutated in place).
        ticker:    Stock symbol.
        shares:    Number of shares to buy.
        price:     Price per share at execution.
        notes:     Optional description.

    Returns:
        Transaction record dict, or {"error": "..."} on failure.
    """
    ticker = ticker.upper()
    total_cost = round(shares * price, 2)
    cash_before = portfolio["cash"]

    if total_cost > cash_before:
        return {"error": f"Insufficient cash: need ${total_cost:.2f}, have ${cash_before:.2f}"}

    cash_after = round(cash_before - total_cost, 2)
    portfolio["cash"] = cash_after

    # Update holdings with average cost basis
    if ticker in portfolio["holdings"]:
        existing = portfolio["holdings"][ticker]
        old_shares = existing["shares"]
        old_cost = existing["avg_cost"]
        new_shares = old_shares + shares
        new_avg = round((old_shares * old_cost + shares * price) / new_shares, 4)
        portfolio["holdings"][ticker] = {"shares": new_shares, "avg_cost": new_avg}
    else:
        portfolio["holdings"][ticker] = {"shares": shares, "avg_cost": price}

    txn = {
        "id": _gen_txn_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "BUY",
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "total_cost": total_cost,
        "cash_before": cash_before,
        "cash_after": cash_after,
        "notes": notes,
    }
    portfolio["transactions"].append(txn)
    return txn


def record_sell(portfolio: dict, ticker: str, shares: float, price: float,
                notes: str = "") -> dict:
    """
    Record a SELL transaction. Mirrors record_buy but increases cash.
    """
    ticker = ticker.upper()
    if ticker not in portfolio["holdings"]:
        return {"error": f"No holding found for {ticker}"}

    existing = portfolio["holdings"][ticker]
    if shares > existing["shares"]:
        return {"error": f"Only {existing['shares']} shares of {ticker} held, cannot sell {shares}"}

    total_proceeds = round(shares * price, 2)
    cash_before = portfolio["cash"]
    cash_after = round(cash_before + total_proceeds, 2)
    portfolio["cash"] = cash_after

    new_shares = existing["shares"] - shares
    if new_shares == 0:
        del portfolio["holdings"][ticker]
    else:
        portfolio["holdings"][ticker]["shares"] = new_shares

    txn = {
        "id": _gen_txn_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "SELL",
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "total_cost": total_proceeds,
        "cash_before": cash_before,
        "cash_after": cash_after,
        "notes": notes,
    }
    portfolio["transactions"].append(txn)
    return txn


def record_snapshot(portfolio: dict, portfolio_value: float,
                    benchmark_value: float, date_str: Optional[str] = None):
    """Append a performance snapshot to the portfolio's history."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    portfolio["performance_snapshots"].append({
        "date": date_str,
        "portfolio_value": round(portfolio_value, 2),
        "benchmark_value": round(benchmark_value, 2),
    })


def get_portfolio_value(portfolio: dict, prices: dict) -> float:
    """Calculate total portfolio value from holdings + cash."""
    total = portfolio["cash"]
    for ticker, info in portfolio["holdings"].items():
        p = prices.get(ticker.upper())
        if p is not None:  # Treat $0 as valid (delisted); only skip truly missing
            total += info["shares"] * p
    return round(total, 2)


def get_all_tickers(portfolio: dict) -> list:
    """Return list of all tickers currently held."""
    return list(portfolio["holdings"].keys())


def clear_all_portfolios(session_state: dict):
    """
    Nuclear reset — wipes ALL portfolio data from session state.
    Called after the user confirms the multi-step reset dialog.
    """
    session_state["portfolios"] = {}
    session_state["model_results"] = {}
    session_state["model_usages"] = {}
    session_state["model_providers"] = {}
    session_state["strategies_generated"] = False
    session_state["trades_executed"] = {}
    session_state["report_generated"] = False
    session_state["_import_file_ids"] = []
    session_state["_import_file_to_keys"] = {}

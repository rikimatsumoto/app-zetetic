"""
market_data.py
==============
Fetches live and historical market data from Yahoo Finance.
Key addition: get_prices_at_date() for backtesting — fetches the closing price
on (or nearest to) a specific historical date.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_current_prices(tickers: list) -> dict:
    """
    Fetch the latest available price for each ticker.
    Returns {ticker: price} dict. Missing tickers map to None.
    """
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            try:
                price = stock.fast_info.get("lastPrice")
                if price and price > 0:
                    prices[ticker.upper()] = round(float(price), 2)
                    continue
            except Exception:
                pass
            # Fallback to recent history
            hist = stock.history(period="5d")
            if not hist.empty:
                prices[ticker.upper()] = round(float(hist["Close"].iloc[-1]), 2)
            else:
                prices[ticker.upper()] = None
        except Exception:
            prices[ticker.upper()] = None
    return prices


def get_prices_at_date(tickers: list, target_date: str) -> dict:
    """
    Fetch the closing price for each ticker on a specific historical date.
    Used for backtesting — gets the price at which trades would have executed.

    If the exact date is a weekend/holiday, returns the nearest prior trading day's close.

    Args:
        tickers:     List of ticker symbols.
        target_date: "YYYY-MM-DD" string of the desired execution date.

    Returns:
        {ticker: price} dict. Missing tickers map to None.
    """
    prices = {}
    # Fetch a small window around the target to handle weekends/holidays
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start = (target_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in tickers:
        try:
            hist = yf.download(ticker, start=start, end=end, progress=False)
            if hist.empty:
                prices[ticker.upper()] = None
                continue

            # Flatten MultiIndex if present (yfinance 0.2.31+ returns MultiIndex)
            close = hist["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            # Get the last available close on or before the target date
            mask = close.index <= pd.Timestamp(target_date)
            valid = close[mask]
            if not valid.empty:
                prices[ticker.upper()] = round(float(valid.iloc[-1]), 2)
            elif not close.empty:
                # Fall back to earliest available if all dates are after target
                prices[ticker.upper()] = round(float(close.iloc[0]), 2)
            else:
                prices[ticker.upper()] = None
        except Exception:
            prices[ticker.upper()] = None
    return prices


def get_historical_prices(tickers: list, start_date: str,
                          end_date: str = None) -> pd.DataFrame:
    """
    Fetch historical closing prices for multiple tickers.
    Returns DataFrame with DatetimeIndex and one column per ticker.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        if len(tickers) == 1:
            data = yf.download(tickers[0], start=start_date, end=end_date, progress=False)
            if not data.empty:
                close = data["Close"]
                # Newer yfinance may return DataFrame even for single ticker
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return close.to_frame(name=tickers[0])
            return pd.DataFrame()

        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(0):
            closes = data["Close"]
            # Flatten any remaining MultiIndex level
            if isinstance(closes.columns, pd.MultiIndex):
                closes.columns = closes.columns.get_level_values(-1)
            return closes
        elif "Close" in data.columns:
            closes = data[["Close"]]
            closes.columns = tickers
            return closes
        return data
    except Exception:
        return pd.DataFrame()


def get_benchmark_data(benchmark: str = "^GSPC", start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
    """
    Fetch benchmark (e.g. S&P 500) historical data.
    Returns DataFrame with Date index and 'Close' column.
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        data = yf.download(benchmark, start=start_date, end=end_date, progress=False)
        return data
    except Exception:
        return pd.DataFrame()


def get_ticker_info(ticker: str) -> dict:
    """Fetch basic info about a ticker (name, sector, market cap, etc.)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A"}

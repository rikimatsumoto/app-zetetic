"""
strategy_generator.py
=====================
Multi-model strategy generation engine.
Supports Anthropic, xAI (Grok), Google Gemini, and local Ollama models
via a unified interface.

The key idea: the SAME thesis goes to up to 3 different models independently,
so the user can compare how model selection affects strategy implementation.

Supported providers:
  - Anthropic: Claude Opus 4, Claude Sonnet 4
  - xAI:       Grok 3, Grok 3 Mini
  - Google:    Gemini 2.5 Flash, Gemini 1.5 Pro
  - Ollama:    any model running locally (llama3, qwen, mistral, etc.)
"""

import json
import requests
import anthropic


# ── System prompt shared by ALL models ─────────────────────────────────────────
STRATEGY_SYSTEM_PROMPT = """You are an expert portfolio strategist and financial analyst. 
The user will provide an investment thesis or narrative. Your job is to translate it into 
3-4 concrete, executable investment strategies ranked from least to most risky.

CRITICAL RULES:
1. Each strategy MUST use real, currently-tradeable ticker symbols (US exchanges preferred).
2. Allocations must sum to 100% of the investable capital for each strategy.
3. Include a mix of ETFs and individual stocks as appropriate.
4. Be specific about allocation percentages — no vague ranges.
5. Rank strategies from Conservative (1) to most Aggressive/Speculative (3 or 4).

Respond ONLY with valid JSON matching this exact schema (no markdown, no backticks, no explanation):

{
  "thesis_summary": "One-sentence restatement of the user's thesis",
  "strategies": [
    {
      "name": "Strategy name",
      "risk_level": "Conservative | Moderate | Aggressive | Speculative",
      "risk_score": 1,
      "rationale": "2-3 sentence explanation of why this strategy fits the thesis",
      "allocations": [
        {
          "ticker": "AAPL",
          "name": "Apple Inc.",
          "weight": 0.25,
          "rationale": "Brief reason for this pick"
        }
      ],
      "rebalancing_notes": "When/how to rebalance",
      "time_horizon": "Short-term (< 6 mo) | Medium-term (6-18 mo) | Long-term (18+ mo)"
    }
  ]
}

The risk_score should be 1 (lowest risk) through 4 (highest risk), matching the rank order.
Ensure every ticker is a real, currently tradeable symbol.
RESPOND ONLY WITH THE JSON. NO OTHER TEXT."""


# ── Model registry ─────────────────────────────────────────────────────────────
# Maps human-friendly display names → API model identifiers

ANTHROPIC_MODELS = {
    "Claude Opus 4": "claude-opus-4-20250514",
    "Claude Sonnet 4": "claude-sonnet-4-20250514",
}

XAI_MODELS = {
    "Grok 3": "grok-3-latest",
    "Grok 3 Mini": "grok-3-mini-latest",
}

GEMINI_MODELS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 1.5 Pro": "gemini-1.5-pro",
}

# ── Pricing ($ per 1M tokens) — used for cost estimation ──────────────────────
# Sources: Anthropic, xAI, & Google pricing pages. Update these if pricing changes.
ANTHROPIC_PRICING = {
    "claude-opus-4-20250514":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input":  3.00, "output": 15.00},
}

XAI_PRICING = {
    "grok-3-latest":      {"input": 3.00, "output": 15.00},
    "grok-3-mini-latest": {"input": 0.30, "output":  0.50},
}

GEMINI_PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-1.5-pro":   {"input": 1.25, "output": 5.00},
}
# Ollama models run locally — $0 API cost, but we still track tokens for reference


def _empty_usage(provider: str, model: str) -> dict:
    """Return a zeroed-out usage dict for error cases."""
    return {
        "provider": provider,
        "model": model,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def get_available_ollama_models(ollama_url: str = "http://localhost:11434") -> list:
    """
    Query the local Ollama server for installed models.
    Returns list of model name strings, or empty list if Ollama isn't running.
    """
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except (requests.ConnectionError, requests.Timeout):
        pass
    return []


def _clean_json_text(text: str) -> str:
    """
    Fix common LLM JSON quirks that break json.loads():
      - Trailing commas before } or ]  (e.g.  "a": 1,} )
      - Single-line // comments
      - Unescaped control characters outside strings
      - Literal newlines / tabs INSIDE JSON string values (common in Gemini 2.5)
    """
    import re

    # 1. Remove single-line // comments (rough heuristic — not inside strings)
    text = re.sub(r'(?m)^\s*//.*$', '', text)          # full-line comments
    text = re.sub(r',\s*//[^\n]*', ',', text)           # trailing comments after values

    # 2. Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # 3. Strip control characters outside of strings (except \n, \r, \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # 4. Fix literal newlines / tabs INSIDE JSON string values.
    #    Walk character-by-character tracking in-string state.
    #    When we hit a raw newline/tab inside a quoted string, escape it.
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        # Toggle in_string on unescaped quote
        if ch == '"' and (i == 0 or text[i - 1] != '\\'):
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
        i += 1

    return ''.join(result)


def _parse_strategy_json(raw_text: str) -> dict:
    """
    Robustly parse LLM output into strategy JSON.

    Parse pipeline (tries each step, stops on first success):
      1. Strict json.loads
      2. json.loads with strict=False  (allows control chars in strings)
      3. Full text cleanup  (_clean_json_text) + strict=False

    Handles markdown fencing, preamble text, trailing commas, comments,
    unescaped newlines inside strings, and other common LLM quirks —
    especially from Gemini 2.5 Flash (a thinking model).
    """
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    if text.startswith("json"):
        text = text[4:]
    text = text.strip()

    # Extract JSON object if surrounded by extra text
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace + 1]

    # Attempt 1: strict parse (fast path for well-formed JSON)
    try:
        strategies = json.loads(text)
    except json.JSONDecodeError:
        # Attempt 2: allow control characters inside strings
        try:
            strategies = json.loads(text, strict=False)
        except json.JSONDecodeError:
            # Attempt 3: full cleanup (trailing commas, comments, newline escaping)
            cleaned = _clean_json_text(text)
            try:
                strategies = json.loads(cleaned, strict=False)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse strategy JSON.\nError: {e}\n"
                    f"Cleaned text (first 500 chars): {cleaned[:500]}"
                )

    if "strategies" not in strategies:
        raise ValueError("Response missing 'strategies' key.")

    # Auto-normalize allocation weights to sum to 1.0
    for s in strategies["strategies"]:
        total_weight = sum(a["weight"] for a in s.get("allocations", []))
        if total_weight > 0 and abs(total_weight - 1.0) > 0.02:
            for a in s["allocations"]:
                a["weight"] = round(a["weight"] / total_weight, 4)

    return strategies


# ── Provider: Anthropic ────────────────────────────────────────────────────────

def generate_strategies_anthropic(thesis: str, api_key: str,
                                  model: str = "claude-sonnet-4-20250514") -> tuple:
    """
    Generate strategies via Anthropic Claude API.

    Returns:
        (strategies_dict, usage_dict) tuple.
        usage_dict contains input/output tokens and estimated cost in USD.
    """
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=STRATEGY_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Here is my investment thesis:\n\n{thesis}\n\n"
                "Please generate 3-4 actionable investment strategies ranked by risk level."
            ),
        }],
    )

    # Extract token usage from the response
    input_tokens = getattr(message.usage, "input_tokens", 0)
    output_tokens = getattr(message.usage, "output_tokens", 0)

    # Calculate estimated cost using pricing table
    pricing = ANTHROPIC_PRICING.get(model, {"input": 0, "output": 0})
    cost = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )

    usage = {
        "provider": "anthropic",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost, 6),
    }

    strategies = _parse_strategy_json(message.content[0].text)
    return strategies, usage


# ── Provider: Ollama (local models) ───────────────────────────────────────────

def generate_strategies_ollama(thesis: str, model: str = "llama3",
                               ollama_url: str = "http://localhost:11434",
                               timeout: int = 180) -> tuple:
    """
    Generate strategies via a local Ollama model.
    Returns (strategies_dict, usage_dict). Cost is $0 for local models.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Here is my investment thesis:\n\n{thesis}\n\n"
                    "Please generate 3-4 actionable investment strategies ranked by risk level. "
                    "Respond ONLY with valid JSON, no other text."
                ),
            },
        ],
        "stream": False,
        "format": "json",  # Request structured JSON output
    }

    try:
        resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {ollama_url}. "
            "Make sure Ollama is running (ollama serve)."
        )
    except requests.HTTPError as e:
        raise ValueError(f"Ollama API error: {e}")

    data = resp.json()
    raw_text = data.get("message", {}).get("content", "")
    if not raw_text:
        raise ValueError(f"Empty response from Ollama model '{model}'.")

    # Ollama returns token counts in response metadata
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    usage = {
        "provider": "ollama",
        "model": model,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": 0.0,  # Local models = free
    }

    return _parse_strategy_json(raw_text), usage


# ── Provider: xAI (Grok models) ──────────────────────────────────────────────

def generate_strategies_xai(thesis: str, api_key: str,
                            model: str = "grok-3-latest") -> tuple:
    """
    Generate strategies via the xAI Grok API (OpenAI-compatible endpoint).

    Returns:
        (strategies_dict, usage_dict) tuple.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Here is my investment thesis:\n\n{thesis}\n\n"
                    "Please generate 3-4 actionable investment strategies ranked by risk level. "
                    "Respond ONLY with valid JSON, no other text."
                ),
            },
        ],
        "temperature": 0.7,
    }

    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers, json=payload, timeout=120,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        raise ConnectionError("Cannot reach the xAI API. Check your internet connection.")
    except requests.HTTPError as e:
        # Extract error message from response body if available
        try:
            err_detail = resp.json().get("error", {}).get("message", str(e))
        except Exception:
            err_detail = str(e)
        raise ValueError(f"xAI API error: {err_detail}")

    data = resp.json()
    raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not raw_text:
        raise ValueError(f"Empty response from xAI model '{model}'.")

    # Extract token usage from OpenAI-compatible response
    usage_data = data.get("usage", {})
    input_tokens = usage_data.get("prompt_tokens", 0)
    output_tokens = usage_data.get("completion_tokens", 0)

    # Calculate estimated cost
    pricing = XAI_PRICING.get(model, {"input": 0, "output": 0})
    cost = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )

    usage = {
        "provider": "xai",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost, 6),
    }

    return _parse_strategy_json(raw_text), usage


# ── Provider: Google Gemini ──────────────────────────────────────────────────

def generate_strategies_gemini(thesis: str, api_key: str,
                               model: str = "gemini-2.5-flash") -> tuple:
    """
    Generate strategies via the Google Gemini REST API.
    Uses the generateContent endpoint directly (no SDK dependency).

    Returns:
        (strategies_dict, usage_dict) tuple.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    # Define the exact schema we expect — Gemini validates output against this
    # before returning, preventing malformed JSON at the source.
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "thesis_summary": {"type": "STRING"},
            "strategies": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "risk_level": {"type": "STRING"},
                        "risk_score": {"type": "INTEGER"},
                        "rationale": {"type": "STRING"},
                        "allocations": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "ticker": {"type": "STRING"},
                                    "name": {"type": "STRING"},
                                    "weight": {"type": "NUMBER"},
                                    "rationale": {"type": "STRING"},
                                },
                                "required": ["ticker", "name", "weight", "rationale"],
                            },
                        },
                        "rebalancing_notes": {"type": "STRING"},
                        "time_horizon": {"type": "STRING"},
                    },
                    "required": ["name", "risk_level", "risk_score", "rationale",
                                 "allocations", "rebalancing_notes", "time_horizon"],
                },
            },
        },
        "required": ["thesis_summary", "strategies"],
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"{STRATEGY_SYSTEM_PROMPT}\n\n"
                            f"Here is my investment thesis:\n\n{thesis}\n\n"
                            "Please generate 3-4 actionable investment strategies ranked by risk level. "
                            "Respond ONLY with valid JSON, no other text."
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.ConnectionError:
        raise ConnectionError("Cannot reach the Google Gemini API. Check your internet connection.")
    except requests.HTTPError as e:
        try:
            err_detail = resp.json().get("error", {}).get("message", str(e))
        except Exception:
            err_detail = str(e)
        raise ValueError(f"Gemini API error: {err_detail}")

    data = resp.json()

    # Extract text from Gemini response structure
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"Empty response from Gemini model '{model}'.")
    raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    if not raw_text:
        raise ValueError(f"No text content in Gemini response for model '{model}'.")

    # Extract token usage from Gemini metadata
    usage_meta = data.get("usageMetadata", {})
    input_tokens = usage_meta.get("promptTokenCount", 0)
    output_tokens = usage_meta.get("candidatesTokenCount", 0)

    # Calculate estimated cost
    pricing = GEMINI_PRICING.get(model, {"input": 0, "output": 0})
    cost = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )

    usage = {
        "provider": "google",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost, 6),
    }

    return _parse_strategy_json(raw_text), usage


# ── Unified interface ─────────────────────────────────────────────────────────

def generate_strategies(thesis: str, provider: str, model_name: str,
                        api_key: str = "",
                        ollama_url: str = "http://localhost:11434",
                        ollama_timeout: int = 180) -> tuple:
    """
    Unified entry point — routes to the correct provider.

    Args:
        thesis:         User's investment narrative.
        provider:       "anthropic", "xai", "google", or "ollama".
        model_name:     Display name (Anthropic/xAI/Google) or model string (Ollama).
        api_key:        API key for Anthropic, xAI, or Google (ignored for Ollama).
        ollama_url:     Ollama server URL (ignored for cloud providers).
        ollama_timeout: Request timeout in seconds for Ollama calls.

    Returns:
        Tuple of (strategies_dict, usage_dict).
        strategies_dict has 'thesis_summary' and 'strategies'.
        usage_dict has 'input_tokens', 'output_tokens', 'estimated_cost_usd', etc.
    """
    if provider == "anthropic":
        model_id = ANTHROPIC_MODELS.get(model_name, model_name)
        return generate_strategies_anthropic(thesis, api_key, model_id)
    elif provider == "xai":
        model_id = XAI_MODELS.get(model_name, model_name)
        return generate_strategies_xai(thesis, api_key, model_id)
    elif provider == "google":
        model_id = GEMINI_MODELS.get(model_name, model_name)
        return generate_strategies_gemini(thesis, api_key, model_id)
    elif provider == "ollama":
        return generate_strategies_ollama(thesis, model_name, ollama_url, timeout=ollama_timeout)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def strategies_to_trades(strategy: dict, total_capital: float, prices: dict) -> list:
    """
    Convert a strategy's allocations into concrete trade orders using market prices.
    Shares are rounded down to whole numbers; leftover stays as cash.
    """
    trades = []
    remaining_cash = total_capital

    for alloc in strategy.get("allocations", []):
        ticker = alloc["ticker"].upper()
        weight = alloc["weight"]
        target_dollars = total_capital * weight

        if ticker not in prices or prices[ticker] is None:
            trades.append({
                "ticker": ticker, "shares": 0, "price": None,
                "dollar_amount": 0, "weight": weight,
                "error": f"Price not found for {ticker}",
            })
            continue

        price = prices[ticker]
        if price <= 0:
            continue

        shares = int(target_dollars / price)
        actual_cost = round(shares * price, 2)

        if actual_cost > remaining_cash:
            shares = int(remaining_cash / price)
            actual_cost = round(shares * price, 2)

        remaining_cash -= actual_cost
        trades.append({
            "ticker": ticker,
            "name": alloc.get("name", ticker),
            "shares": shares,
            "price": price,
            "dollar_amount": actual_cost,
            "weight": weight,
            "rationale": alloc.get("rationale", ""),
        })

    return trades

#!/usr/bin/env python3
"""
Agent Cost Tracker MCP Server

Tracks AI agent token usage and API costs with budget alerts.
Provides self-awareness for AI agents about their spending.

Tools:
  - cost_track_call    Track a single API call
  - cost_get_usage     Get usage summary
  - cost_set_budget    Set a budget limit
  - cost_list_models   List supported models with pricing
  - cost_check_budget  Check budget status
  - cost_estimate      Estimate cost before making a call
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import tiktoken

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Annotations,
    CallToolRequest,
    ListToolsRequest,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARACTER_LIMIT = 25000
DATA_DIR = Path.home() / ".agent-cost-tracker"
CALLS_FILE = DATA_DIR / "calls.json"
BUDGETS_FILE = DATA_DIR / "budgets.json"
MODELS_FILE = DATA_DIR / "models.json"
VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Default model pricing — pre-seeded
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "gpt-4o": {
        "provider": "OpenAI",
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
        "context_window": 128000,
    },
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "context_window": 128000,
    },
    "claude-sonnet-4": {
        "provider": "Anthropic",
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "context_window": 200000,
    },
    "claude-haiku-3.5": {
        "provider": "Anthropic",
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
        "context_window": 200000,
    },
    "deepseek-chat": {
        "provider": "DeepSeek",
        "input_cost_per_1m": 0.27,
        "output_cost_per_1m": 1.10,
        "context_window": 128000,
    },
    "deepseek-reasoner": {
        "provider": "DeepSeek",
        "input_cost_per_1m": 0.55,
        "output_cost_per_1m": 2.19,
        "context_window": 128000,
    },
    "gemini-2.5-pro": {
        "provider": "Google",
        "input_cost_per_1m": 1.25,
        "output_cost_per_1m": 10.00,
        "context_window": 1048576,
    },
    "gemini-2.5-flash": {
        "provider": "Google",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "context_window": 1048576,
    },
    "llama-4-maverick": {
        "provider": "Meta",
        "input_cost_per_1m": 0.20,
        "output_cost_per_1m": 0.80,
        "context_window": 128000,
    },
    "command-r-plus": {
        "provider": "Cohere",
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
        "context_window": 128000,
    },
}

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def ensure_data_dir() -> None:
    """Create the data directory and seed files if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MODELS_FILE.exists():
        write_json(MODELS_FILE, DEFAULT_MODELS)
    if not CALLS_FILE.exists():
        write_json(CALLS_FILE, [])
    if not BUDGETS_FILE.exists():
        write_json(BUDGETS_FILE, {})


def _maybe_seed_models() -> None:
    """If models.json exists, merge any missing default models into it."""
    try:
        current = read_json(MODELS_FILE)
        changed = False
        for key, val in DEFAULT_MODELS.items():
            if key not in current:
                current[key] = val
                changed = True
        if changed:
            write_json(MODELS_FILE, current)
    except Exception:
        write_json(MODELS_FILE, DEFAULT_MODELS)


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write a JSON file atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


def _append_call(record: dict) -> None:
    """Append a call record to calls.json."""
    calls = read_json(CALLS_FILE)
    calls.append(record)
    write_json(CALLS_FILE, calls)


def _truncate(text: str, limit: int = CHARACTER_LIMIT) -> str:
    """Truncate text to the character limit."""
    if len(text) <= limit:
        return text
    return text[: limit - 50] + "\n\n... [truncated, response too long]\n"


def _make_error(message: str) -> dict:
    """Return an error-as-result dict."""
    return {"status": "error", "error": message, "isError": True}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a model given token counts."""
    models = read_json(MODELS_FILE)
    if model not in models:
        raise ValueError(f"Unknown model: {model}")
    m = models[model]
    input_cost = (input_tokens / 1_000_000) * m["input_cost_per_1m"]
    output_cost = (output_tokens / 1_000_000) * m["output_cost_per_1m"]
    return round(input_cost + output_cost, 6)


def _estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count using tiktoken."""
    # Map model names to tiktoken encoding names
    encoding_map = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "claude-sonnet-4": "cl100k_base",
        "claude-haiku-3.5": "cl100k_base",
        "deepseek-chat": "cl100k_base",
        "deepseek-reasoner": "cl100k_base",
        "gemini-2.5-pro": "cl100k_base",
        "gemini-2.5-flash": "cl100k_base",
        "llama-4-maverick": "cl100k_base",
        "command-r-plus": "cl100k_base",
    }
    encoding_name = encoding_map.get(model, "cl100k_base")
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    """ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def _start_of_day() -> datetime:
    """Start of today in UTC."""
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_week() -> datetime:
    """Start of current week (Monday) in UTC."""
    now = datetime.now(timezone.utc)
    monday = now - timedelta(days=now.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_month() -> datetime:
    """Start of current month in UTC."""
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _in_period(ts_str: str, period: str) -> bool:
    """Check if a timestamp is within the given period."""
    try:
        ts = datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return False

    if period == "today":
        return ts >= _start_of_day()
    elif period == "week":
        return ts >= _start_of_week()
    elif period == "month":
        return ts >= _start_of_month()
    elif period == "all":
        return True
    # Default to month
    return ts >= _start_of_month()


# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(server: Server):
    """Async lifespan — ensures data dir and seeds models on startup."""
    ensure_data_dir()
    _maybe_seed_models()
    yield


server = Server("agent-cost-tracker", version=VERSION, lifespan=lifespan)


# ===========================================================================
# TOOLS
# ===========================================================================


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools with proper annotations."""
    return [
        Tool(
            name="cost_track_call",
            description="Track a single API call: record model, tokens, and cost. Auto-calculates cost if omitted. Returns running totals.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g., gpt-4o, claude-sonnet-4)",
                    },
                    "input_tokens": {
                        "type": "integer",
                        "description": "Number of input/prompt tokens consumed",
                    },
                    "output_tokens": {
                        "type": "integer",
                        "description": "Number of output/completion tokens generated",
                    },
                    "cost": {
                        "type": "number",
                        "description": "Optional: provide cost manually (auto-calculated if omitted)",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Optional: identifier for the calling agent",
                    },
                },
                "required": ["model", "input_tokens", "output_tokens"],
            },
            annotations=Annotations(
                title="Track API Call",
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=True,
            ),
        ),
        Tool(
            name="cost_get_usage",
            description="Get usage summary: total cost, total tokens, calls today/this month, breakdown by model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["today", "week", "month", "all"],
                        "description": "Time period to summarize (default: month)",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown)",
                    },
                },
            },
            annotations=Annotations(
                title="Get Usage Summary",
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        Tool(
            name="cost_set_budget",
            description="Set a monthly budget with alert threshold. Returns current budget status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "monthly_limit": {
                        "type": "number",
                        "description": "Maximum monthly spend in USD",
                    },
                    "alert_threshold_percent": {
                        "type": "number",
                        "description": "Alert when spending reaches this percentage of budget (0-100, default: 80)",
                    },
                },
                "required": ["monthly_limit"],
            },
            annotations=Annotations(
                title="Set Budget",
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        Tool(
            name="cost_list_models",
            description="List all supported models with their per-token pricing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown)",
                    },
                },
            },
            annotations=Annotations(
                title="List Models",
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        Tool(
            name="cost_check_budget",
            description="Check current budget status: spending, remaining, projected monthly spend.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown)",
                    },
                },
            },
            annotations=Annotations(
                title="Check Budget",
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        Tool(
            name="cost_estimate",
            description="Estimate cost *before* making an API call. Provide model and estimated token counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g., gpt-4o)",
                    },
                    "estimated_input_tokens": {
                        "type": "integer",
                        "description": "Estimated input/prompt tokens",
                    },
                    "estimated_output_tokens": {
                        "type": "integer",
                        "description": "Estimated output tokens",
                    },
                    "input_text": {
                        "type": "string",
                        "description": "Input text to auto-estimate tokens from (alternative to estimated_input_tokens)",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown)",
                    },
                },
                "required": ["model"],
            },
            annotations=Annotations(
                title="Estimate Cost",
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
    ]


@server.call_tool()
async def call_tool(request: CallToolRequest) -> list[TextContent]:
    """Route tool calls to the appropriate handler."""
    name = request.params.name
    args = request.params.arguments or {}

    handlers = {
        "cost_track_call": handle_track_call,
        "cost_get_usage": handle_get_usage,
        "cost_set_budget": handle_set_budget,
        "cost_list_models": handle_list_models,
        "cost_check_budget": handle_check_budget,
        "cost_estimate": handle_estimate,
    }

    handler = handlers.get(name)
    if handler is None:
        result = _make_error(f"Unknown tool: {name}")
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    try:
        result = handler(args)
    except Exception as e:
        result = _make_error(str(e))

    if isinstance(result, dict):
        fmt = args.get("format", "markdown")
        if fmt == "json" or result.get("status") == "error":
            text = json.dumps(result, indent=2)
        else:
            text = _format_markdown(result)
    elif isinstance(result, str):
        text = result
    else:
        text = json.dumps(result, indent=2)

    text = _truncate(text)
    return [TextContent(type="text", text=text)]


# ===========================================================================
# Tool handlers
# ===========================================================================


def handle_track_call(args: dict) -> dict:
    """Track a single API call."""
    model = args.get("model", "")
    input_tokens = args.get("input_tokens", 0)
    output_tokens = args.get("output_tokens", 0)
    cost = args.get("cost")
    agent_id = args.get("agent_id", "unknown")

    # Validate
    models = read_json(MODELS_FILE)
    if model not in models:
        return _make_error(
            f"Unknown model '{model}'. Use cost_list_models to see supported models."
        )

    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return _make_error("input_tokens and output_tokens must be integers")

    if input_tokens < 0 or output_tokens < 0:
        return _make_error("Token counts cannot be negative")

    # Calculate or validate cost
    if cost is None:
        try:
            cost = _calculate_cost(model, input_tokens, output_tokens)
        except ValueError as e:
            return _make_error(str(e))
    else:
        cost = round(float(cost), 6)

    # Record call
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": _now(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": cost,
        "agent_id": agent_id,
    }
    _append_call(record)

    # Compute running totals
    calls = read_json(CALLS_FILE)
    total_cost = sum(c["cost"] for c in calls)
    total_tokens = sum(c["total_tokens"] for c in calls)
    today_calls = sum(1 for c in calls if _in_period(c["timestamp"], "today"))
    month_calls = sum(1 for c in calls if _in_period(c["timestamp"], "month"))

    # Budget check
    budget_status = _budget_alert(calls)

    return {
        "status": "recorded",
        "call_id": record["id"],
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
        "running_total_cost": round(total_cost, 4),
        "running_total_tokens": total_tokens,
        "calls_today": today_calls,
        "calls_this_month": month_calls,
        "budget_warning": budget_status.get("warning", False),
        "budget_message": budget_status.get("message", ""),
    }


def handle_get_usage(args: dict) -> dict:
    """Get usage summary for a period."""
    period = args.get("period", "month")
    fmt = args.get("format", "markdown")

    if period not in ("today", "week", "month", "all"):
        period = "month"

    calls = read_json(CALLS_FILE)
    filtered = [c for c in calls if _in_period(c["timestamp"], period)]

    total_cost = round(sum(c["cost"] for c in filtered), 4)
    total_input = sum(c["input_tokens"] for c in filtered)
    total_output = sum(c["output_tokens"] for c in filtered)
    total_tokens = total_input + total_output
    call_count = len(filtered)

    # Breakdown by model
    by_model = {}
    for c in filtered:
        m = c["model"]
        if m not in by_model:
            by_model[m] = {"calls": 0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0}
        by_model[m]["calls"] += 1
        by_model[m]["cost"] += c["cost"]
        by_model[m]["input_tokens"] += c["input_tokens"]
        by_model[m]["output_tokens"] += c["output_tokens"]

    # Round model costs
    for m in by_model:
        by_model[m]["cost"] = round(by_model[m]["cost"], 4)

    # Budget info
    budget = _get_budget()
    budget_info = None
    if budget:
        budget_info = _budget_alert(calls)

    return {
        "status": "ok",
        "period": period,
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "call_count": call_count,
        "by_model": by_model,
        "budget": budget_info,
        "format": fmt,
    }


def handle_set_budget(args: dict) -> dict:
    """Set a monthly budget."""
    monthly_limit = args.get("monthly_limit")
    alert_threshold_percent = args.get("alert_threshold_percent", 80)

    if monthly_limit is None:
        return _make_error("monthly_limit is required")

    monthly_limit = float(monthly_limit)
    alert_threshold_percent = float(alert_threshold_percent)

    if monthly_limit <= 0:
        return _make_error("monthly_limit must be positive")
    if not 1 <= alert_threshold_percent <= 100:
        return _make_error("alert_threshold_percent must be between 1 and 100")

    budget = {
        "monthly_limit": monthly_limit,
        "alert_threshold_percent": alert_threshold_percent,
        "created_at": _now(),
        "updated_at": _now(),
    }
    write_json(BUDGETS_FILE, budget)

    # Return current status
    calls = read_json(CALLS_FILE)
    alert = _budget_alert(calls)

    return {
        "status": "ok",
        "message": f"Monthly budget set to ${monthly_limit:.2f} with {alert_threshold_percent}% alert threshold",
        "monthly_limit": monthly_limit,
        "alert_threshold_percent": alert_threshold_percent,
        "current_spending": alert.get("current_spending", 0),
        "remaining": alert.get("remaining", monthly_limit),
        "percent_used": alert.get("percent_used", 0),
        "warning": alert.get("warning", False),
        "warning_message": alert.get("message", ""),
    }


def handle_list_models(args: dict) -> dict:
    """List all supported models with pricing."""
    fmt = args.get("format", "markdown")
    models = read_json(MODELS_FILE)

    model_list = []
    for name, info in models.items():
        model_list.append(
            {
                "name": name,
                "provider": info["provider"],
                "input_cost_per_1m": info["input_cost_per_1m"],
                "output_cost_per_1m": info["output_cost_per_1m"],
                "context_window": info["context_window"],
            }
        )

    return {
        "status": "ok",
        "models": model_list,
        "count": len(model_list),
        "format": fmt,
    }


def handle_check_budget(args: dict) -> dict:
    """Check budget status."""
    fmt = args.get("format", "markdown")
    calls = read_json(CALLS_FILE)
    budget = _get_budget()

    if not budget:
        return {
            "status": "ok",
            "budget_set": False,
            "message": "No budget has been set. Use cost_set_budget to configure one.",
            "format": fmt,
        }

    alert = _budget_alert(calls)

    # Projected monthly spend
    month_calls = [c for c in calls if _in_period(c["timestamp"], "month")]
    month_cost = sum(c["cost"] for c in month_calls)

    now = datetime.now(timezone.utc)
    days_in_month = 30  # approximate
    day_of_month = now.day
    remaining_days = days_in_month - day_of_month

    projected = month_cost
    if day_of_month > 0 and month_cost > 0:
        daily_rate = month_cost / day_of_month
        projected = month_cost + (daily_rate * remaining_days)
    projected = round(projected, 2)

    return {
        "status": "ok",
        "budget_set": True,
        "monthly_limit": budget["monthly_limit"],
        "alert_threshold_percent": budget["alert_threshold_percent"],
        "current_spending": round(month_cost, 4),
        "remaining": round(budget["monthly_limit"] - month_cost, 4),
        "percent_used": round((month_cost / budget["monthly_limit"]) * 100, 1),
        "projected_monthly_spend": projected,
        "warning": alert.get("warning", False),
        "warning_message": alert.get("message", ""),
        "format": fmt,
    }


def handle_estimate(args: dict) -> dict:
    """Estimate cost before making a call."""
    model = args.get("model", "")
    estimated_input_tokens = args.get("estimated_input_tokens")
    estimated_output_tokens = args.get("estimated_output_tokens", 0)
    input_text = args.get("input_text")
    fmt = args.get("format", "markdown")

    models = read_json(MODELS_FILE)
    if model not in models:
        return _make_error(
            f"Unknown model '{model}'. Use cost_list_models to see supported models."
        )

    # Auto-estimate tokens from input_text if provided
    if input_text and estimated_input_tokens is None:
        estimated_input_tokens = _estimate_tokens(input_text, model)

    if estimated_input_tokens is None:
        estimated_input_tokens = 0

    estimated_input_tokens = int(estimated_input_tokens)
    estimated_output_tokens = int(estimated_output_tokens)

    if estimated_input_tokens < 0 or estimated_output_tokens < 0:
        return _make_error("Token estimates cannot be negative")

    cost = _calculate_cost(model, estimated_input_tokens, estimated_output_tokens)

    return {
        "status": "ok",
        "model": model,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
        "estimated_cost_usd": cost,
        "format": fmt,
    }


# ===========================================================================
# Budget helpers
# ===========================================================================


def _get_budget() -> dict | None:
    """Read the budget config."""
    if not BUDGETS_FILE.exists():
        return None
    try:
        budget = read_json(BUDGETS_FILE)
        if not budget or "monthly_limit" not in budget:
            return None
        return budget
    except Exception:
        return None


def _budget_alert(calls: list) -> dict:
    """Check budget and return alert info."""
    budget = _get_budget()
    if not budget:
        return {"warning": False, "message": ""}

    month_calls = [c for c in calls if _in_period(c["timestamp"], "month")]
    month_cost = sum(c["cost"] for c in month_calls)
    limit = budget["monthly_limit"]
    threshold = budget["alert_threshold_percent"] / 100.0
    percent_used = (month_cost / limit) * 100 if limit > 0 else 0

    warning = percent_used >= budget["alert_threshold_percent"]
    if percent_used >= 100:
        message = (
            f"🚨 **BUDGET EXCEEDED!** Spent ${month_cost:.2f} of ${limit:.2f} monthly budget "
            f"({percent_used:.1f}%). Consider pausing non-essential API calls."
        )
    elif warning:
        message = (
            f"⚠️ **Budget alert:** Spending at {percent_used:.1f}% of ${limit:.2f} monthly budget "
            f"(${month_cost:.2f} spent, ${limit - month_cost:.2f} remaining)."
        )
    else:
        message = ""

    return {
        "warning": warning,
        "message": message,
        "current_spending": round(month_cost, 4),
        "remaining": round(limit - month_cost, 4),
        "percent_used": round(percent_used, 1),
        "monthly_limit": limit,
    }


# ===========================================================================
# Markdown formatting
# ===========================================================================


def _format_markdown(data: dict) -> str:
    """Format a result dict as readable markdown."""
    status = data.get("status", "")

    if status == "error":
        return f"❌ **Error:** {data.get('error', 'Unknown error')}"

    lines = []
    fmt = data.pop("format", "markdown")

    if "models" in data:
        # Model list
        lines.append("## 📊 Supported Models\n")
        lines.append("| Model | Provider | Input $/1M | Output $/1M | Context Window |")
        lines.append("|---|---|---|---|---|")
        for m in data["models"]:
            lines.append(
                f"| {m['name']} | {m['provider']} | ${m['input_cost_per_1m']:.2f} | "
                f"${m['output_cost_per_1m']:.2f} | {m['context_window']:,} |"
            )
        return "\n".join(lines)

    if status == "recorded":
        # Track call
        lines.append("## ✅ Call Tracked\n")
        lines.append(f"- **Call ID:** `{data.get('call_id', '')}`")
        lines.append(f"- **Model:** {data.get('model', '')}")
        lines.append(f"- **Input Tokens:** {data.get('input_tokens', 0):,}")
        lines.append(f"- **Output Tokens:** {data.get('output_tokens', 0):,}")
        lines.append(f"- **Total Tokens:** {data.get('total_tokens', 0):,}")
        lines.append(f"- **Cost:** ${data.get('cost_usd', 0):.6f}\n")
        lines.append("### Running Totals")
        lines.append(f"- **Total Cost:** ${data.get('running_total_cost', 0):.4f}")
        lines.append(f"- **Total Tokens:** {data.get('running_total_tokens', 0):,}")
        lines.append(f"- **Calls Today:** {data.get('calls_today', 0)}")
        lines.append(f"- **Calls This Month:** {data.get('calls_this_month', 0)}")
        if data.get("budget_warning"):
            lines.append(f"\n{data['budget_message']}")
        return "\n".join(lines)

    if status == "ok":
        # Generic OK — determine type from fields
        if "period" in data and "by_model" in data:
            # Usage summary
            return _format_usage_markdown(data)
        if "budget_set" in data:
            # Budget check
            return _format_budget_markdown(data)
        if "monthly_limit" in data and "alert_threshold_percent" in data:
            # Budget set
            return _format_budget_set_markdown(data)
        if "estimated_cost_usd" in data:
            # Estimate
            lines.append("## 💰 Cost Estimate\n")
            lines.append(f"- **Model:** {data.get('model', '')}")
            lines.append(f"- **Est. Input Tokens:** {data.get('estimated_input_tokens', 0):,}")
            lines.append(f"- **Est. Output Tokens:** {data.get('estimated_output_tokens', 0):,}")
            lines.append(f"- **Est. Total Tokens:** {data.get('estimated_total_tokens', 0):,}")
            lines.append(f"- **Est. Cost:** ${data.get('estimated_cost_usd', 0):.6f}")
            return "\n".join(lines)

    # Fallback
    return json.dumps(data, indent=2)


def _format_usage_markdown(data: dict) -> str:
    """Format usage summary."""
    lines = [f"## 📈 Usage Summary ({data.get('period', 'month')})\n"]
    lines.append(f"- **Total Cost:** ${data.get('total_cost_usd', 0):.4f}")
    lines.append(f"- **Total Tokens:** {data.get('total_tokens', 0):,}")
    lines.append(f"- **Input Tokens:** {data.get('total_input_tokens', 0):,}")
    lines.append(f"- **Output Tokens:** {data.get('total_output_tokens', 0):,}")
    lines.append(f"- **API Calls:** {data.get('call_count', 0)}\n")

    by_model = data.get("by_model", {})
    if by_model:
        lines.append("### By Model\n")
        lines.append("| Model | Calls | Cost | Input Tokens | Output Tokens |")
        lines.append("|---|---|---|---|---|")
        for model, info in sorted(by_model.items()):
            lines.append(
                f"| {model} | {info['calls']} | ${info['cost']:.4f} | "
                f"{info['input_tokens']:,} | {info['output_tokens']:,} |"
            )

    budget = data.get("budget")
    if budget and budget.get("warning"):
        lines.append(f"\n{budget['message']}")

    return "\n".join(lines)


def _format_budget_markdown(data: dict) -> str:
    """Format budget check."""
    if not data.get("budget_set"):
        return "## 💰 Budget Status\n\nNo budget has been set. Use `cost_set_budget` to configure one."

    lines = ["## 💰 Budget Status\n"]
    lines.append(f"- **Monthly Limit:** ${data.get('monthly_limit', 0):.2f}")
    lines.append(f"- **Current Spending:** ${data.get('current_spending', 0):.4f}")
    lines.append(f"- **Remaining:** ${data.get('remaining', 0):.4f}")
    lines.append(f"- **Percent Used:** {data.get('percent_used', 0)}%")
    lines.append(f"- **Projected Monthly Spend:** ${data.get('projected_monthly_spend', 0):.2f}")
    lines.append(f"- **Alert Threshold:** {data.get('alert_threshold_percent', 80)}%\n")

    if data.get("warning"):
        lines.append(data["warning_message"])

    return "\n".join(lines)


def _format_budget_set_markdown(data: dict) -> str:
    """Format budget set confirmation."""
    lines = ["## ✅ Budget Set\n"]
    lines.append(f"- **Monthly Limit:** ${data.get('monthly_limit', 0):.2f}")
    lines.append(f"- **Alert Threshold:** {data.get('alert_threshold_percent', 80)}%")
    lines.append(f"- **Current Spending:** ${data.get('current_spending', 0):.4f}")
    lines.append(f"- **Remaining:** ${data.get('remaining', 0):.4f}")
    lines.append(f"- **Percent Used:** {data.get('percent_used', 0)}%")
    if data.get("warning"):
        lines.append(f"\n{data['warning_message']}")
    return "\n".join(lines)


# ===========================================================================
# Entry point
# ===========================================================================


def main():
    """Run the MCP server."""
    import asyncio

    asyncio.run(_run_server())


async def _run_server():
    """Async entry point for the stdio server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()

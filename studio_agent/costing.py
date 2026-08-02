"""Pure helpers for auditable provider-cost estimates.

Provider adapters normalize their usage through this module, then place the returned
tracking payload on ``Generation.meta["cost_tracking"]``.  The project ledger can
therefore persist one consistent record without knowing vendor-specific pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostEstimate:
    cost_usd: float
    tracking: dict[str, Any]


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def normalize_token_usage(usage: dict[str, Any]) -> dict[str, int]:
    """Normalize OpenAI/Ark token fields and separate cached input tokens."""

    input_tokens = _non_negative_int(
        usage.get("input_tokens", usage.get("prompt_tokens", 0))
    )
    output_tokens = _non_negative_int(
        usage.get("output_tokens", usage.get("completion_tokens", 0))
    )
    details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {}
    )
    cached = min(
        input_tokens,
        _non_negative_int(
            usage.get("cached_input_tokens", details.get("cached_tokens", 0))
        ),
    )
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "billable_input_tokens": input_tokens - cached,
        "output_tokens": output_tokens,
    }


def estimate_token_cost(
    usage: dict[str, Any],
    *,
    input_per_million: float,
    cached_input_per_million: float,
    output_per_million: float,
    native_currency: str,
    usd_per_native_unit: float,
    model: str,
    pricing_source: str,
    pricing_as_of: str,
) -> CostEstimate:
    """Apply configured token rates and retain the complete calculation provenance."""

    normalized = normalize_token_usage(usage)
    native_cost = round(
        normalized["billable_input_tokens"] / 1_000_000 * input_per_million
        + normalized["cached_input_tokens"]
        / 1_000_000
        * cached_input_per_million
        + normalized["output_tokens"] / 1_000_000 * output_per_million,
        6,
    )
    tracking: dict[str, Any] = {
        "usage": normalized,
        "native_cost": native_cost,
        "native_currency": native_currency,
        "estimate": True,
        "usage_missing": not any(normalized.values()),
        "pricing": {
            "model": model,
            "input_per_million": input_per_million,
            "cached_input_per_million": cached_input_per_million,
            "output_per_million": output_per_million,
            "source": pricing_source,
            "as_of": pricing_as_of,
        },
    }
    if native_currency != "USD":
        tracking["usd_conversion_rate"] = usd_per_native_unit
    return CostEstimate(round(native_cost * usd_per_native_unit, 6), tracking)

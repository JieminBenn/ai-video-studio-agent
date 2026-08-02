from studio_agent.costing import estimate_token_cost, normalize_token_usage


def test_normalize_openai_usage_subtracts_cached_tokens_from_regular_input():
    usage = normalize_token_usage({
        "prompt_tokens": 1_000_000,
        "completion_tokens": 500_000,
        "prompt_tokens_details": {"cached_tokens": 250_000},
    })

    assert usage == {
        "input_tokens": 1_000_000,
        "cached_input_tokens": 250_000,
        "billable_input_tokens": 750_000,
        "output_tokens": 500_000,
    }


def test_estimate_cny_token_cost_retains_native_and_normalized_amounts():
    result = estimate_token_cost(
        {"prompt_tokens": 1_000_000, "completion_tokens": 500_000},
        input_per_million=12.0,
        cached_input_per_million=1.0,
        output_per_million=24.0,
        native_currency="CNY",
        usd_per_native_unit=0.14,
        model="deepseek-v4-pro-260425",
        pricing_source="https://www.volcengine.com/docs/82379/1544106",
        pricing_as_of="2026-07-01",
    )

    assert result.cost_usd == 3.36
    assert result.tracking["native_cost"] == 24.0
    assert result.tracking["native_currency"] == "CNY"
    assert result.tracking["usd_conversion_rate"] == 0.14
    assert result.tracking["estimate"] is True


def test_missing_usage_is_explicitly_incomplete():
    result = estimate_token_cost(
        {},
        input_per_million=1.0,
        cached_input_per_million=1.0,
        output_per_million=1.0,
        native_currency="USD",
        usd_per_native_unit=1.0,
        model="unknown",
        pricing_source="config",
        pricing_as_of="2026-07-01",
    )

    assert result.cost_usd == 0.0
    assert result.tracking["usage_missing"] is True


def test_normalize_usage_clamps_invalid_and_negative_values():
    usage = normalize_token_usage({
        "input_tokens": -10,
        "output_tokens": "invalid",
        "cached_input_tokens": 50,
    })

    assert usage == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "billable_input_tokens": 0,
        "output_tokens": 0,
    }

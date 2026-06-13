"""Tests for configuration helpers."""

from adaptive_roi_codec.utils.config import merge_dicts


def test_merge_dicts_recursively_overrides_nested_keys() -> None:
    base = {"training": {"epochs": 50, "lr": 1e-4}, "quantizer": {"kappa": 2.0}}
    override = {"training": {"epochs": 10}, "quantizer": {"kappa": 1.5}}

    merged = merge_dicts(base, override)

    assert merged["training"]["epochs"] == 10
    assert merged["training"]["lr"] == 1e-4
    assert merged["quantizer"]["kappa"] == 1.5

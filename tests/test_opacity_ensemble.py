from __future__ import annotations

from tierra_plus.forward.opacity.ensemble import (
    load_benchmark_case,
    posterior_weights_for_case,
)


def test_opacity_ensemble_converges_on_expected_model() -> None:
    case = load_benchmark_case("synthetic-opacity-reference")

    result = posterior_weights_for_case(case)

    assert result.best_model == case.expected_best_model
    assert result.weights[case.expected_best_model] > 0.6

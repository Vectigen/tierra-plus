import numpy as np
import pytest

from tierra_plus.diagnostics.network import TanhRegressor, _loss_and_gradients, fit


def test_analytic_gradients_match_central_differences():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(5, 3))
    y = rng.normal(size=5)
    parameters = (
        rng.normal(scale=0.2, size=(3, 4)),
        rng.normal(scale=0.2, size=4),
        rng.normal(scale=0.2, size=4),
        rng.normal(scale=0.2, size=1),
    )
    _, gradients = _loss_and_gradients(x, y, parameters)
    epsilon = 1e-6
    for parameter, gradient in zip(parameters, gradients, strict=True):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            upper, _ = _loss_and_gradients(x, y, parameters)
            parameter[index] = original - epsilon
            lower, _ = _loss_and_gradients(x, y, parameters)
            parameter[index] = original
            assert gradient[index] == pytest.approx((upper - lower) / (2 * epsilon), abs=1e-8)


def test_deterministic_fit_and_training_only_scaling():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(128, 3))
    y = 3.0 + 0.7 * x[:, 0] - x[:, 1]
    first = fit(x, y, seed=4, hidden=8, steps=80, batch_size=32)
    second = fit(x, y, seed=4, hidden=8, steps=80, batch_size=32)
    for key, value in first.state_dict().items():
        np.testing.assert_array_equal(value, second.state_dict()[key])
    expected = first.predict(x[:1])
    combined = first.predict(np.concatenate([x[:1], 100 * x[1:2]], axis=0))
    np.testing.assert_allclose(combined[:1], expected, atol=1e-13)
    np.testing.assert_allclose(first.x_mean, x.mean(axis=0))
    assert not first.w1.flags.writeable


def test_learns_regression_and_round_trips_weights():
    rng = np.random.default_rng(14)
    x = rng.uniform(-1.0, 1.0, size=(512, 2))
    y = 4.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1]
    model = fit(x, y, seed=3, hidden=16, steps=600, batch_size=128)
    test_x = rng.uniform(-1.0, 1.0, size=(128, 2))
    test_y = 4.0 + 2.0 * test_x[:, 0] - 0.5 * test_x[:, 1]
    assert np.sqrt(np.mean((model.predict(test_x) - test_y) ** 2)) < 0.08
    restored = TanhRegressor.from_state_dict(model.state_dict())
    np.testing.assert_array_equal(model.predict(test_x), restored.predict(test_x))
    np.testing.assert_array_equal(
        model.predict(test_x), model.predict_from_hidden(model.hidden(test_x))
    )


@pytest.fixture
def separable_model():
    return TanhRegressor(
        w1=np.eye(2),
        b1=np.zeros(2),
        w2=np.array([2.0, 3.0]),
        b2=np.array([0.0]),
        x_mean=np.zeros(2),
        x_scale=np.ones(2),
        y_mean=np.array([0.0]),
        y_scale=np.array([1.0]),
    )


def test_paired_direction_and_rank_one_removal(separable_model):
    first = np.array([[0.2, -0.5], [-0.1, 0.2], [0.3, -0.2]])
    second = first + np.array([0.0, 0.4])
    direction, center = separable_model.hidden_nuisance_direction(first, second, np.full(3, 0.4))
    np.testing.assert_allclose(direction, [0.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(center, np.tanh(np.vstack([first, second])).mean(axis=0))
    expected = 2 * np.tanh(first[:, 0]) + 3 * center[1]
    np.testing.assert_allclose(separable_model.patched_predict(first, direction, center), expected)
    np.testing.assert_allclose(
        separable_model.patched_predict(first, 4 * direction, center),
        separable_model.patched_predict(second, direction, center),
    )
    state = separable_model.state_dict()
    state["w1"][0, 0] = 17
    assert separable_model.w1[0, 0] == 1.0


def test_no_hidden_nuisance_response_is_explicit(separable_model):
    x = np.ones((3, 2))
    with pytest.raises(ValueError, match="nonzero hidden nuisance response"):
        separable_model.hidden_nuisance_direction(x, x, np.ones(3))
    with pytest.raises(ValueError, match="nonzero intervention"):
        separable_model.hidden_nuisance_direction(x, x + 1, np.zeros(3))
    with pytest.raises(ValueError, match="nonzero norm"):
        separable_model.patched_predict(x, np.zeros(2), np.zeros(2))


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 0},
        {"steps": 1.5},
        {"batch_size": True},
        {"hidden": -1},
        {"seed": -1},
        {"seed": True},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
    ],
)
def test_invalid_fit_configuration(options):
    arguments = {"seed": 1, "steps": 2, **options}
    with pytest.raises(ValueError):
        fit(np.ones((3, 2)), np.ones(3), **arguments)


@pytest.mark.parametrize(
    "x,y",
    [
        (np.ones((3, 2)), np.ones((3, 1))),
        (np.ones((3, 2)), np.ones(2)),
        (np.ones((1, 2)), np.ones(1)),
        (np.ones((3, 0)), np.ones(3)),
        (np.full((3, 2), np.nan), np.ones(3)),
    ],
)
def test_invalid_training_data(x, y):
    with pytest.raises(ValueError):
        fit(x, y, seed=1, steps=2)


def test_invalid_prediction_and_patch_shapes(separable_model):
    with pytest.raises(ValueError, match="features"):
        separable_model.predict(np.ones((3, 1)))
    with pytest.raises(ValueError, match="features"):
        separable_model.predict_from_hidden(np.ones((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        separable_model.patched_predict(np.ones((3, 2)), np.ones(1), np.ones(2))
    with pytest.raises(ValueError, match="matching sample counts"):
        separable_model.hidden_nuisance_direction(np.ones((3, 2)), np.ones((2, 2)), np.ones(3))
    with pytest.raises(ValueError, match="at least two"):
        separable_model.hidden_nuisance_direction(np.ones((1, 2)), np.ones((1, 2)), np.ones(1))
    state = separable_model.state_dict()
    state["x_scale"][0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        TanhRegressor.from_state_dict(state)
    with pytest.raises(ValueError, match="exactly"):
        TanhRegressor.from_state_dict({"w1": np.eye(2)})


def test_constant_columns_and_targets_remain_finite():
    model = fit(np.ones((16, 2)), np.full(16, 7.0), seed=1, steps=20)
    np.testing.assert_allclose(model.predict(np.ones((5, 2))), 7.0)

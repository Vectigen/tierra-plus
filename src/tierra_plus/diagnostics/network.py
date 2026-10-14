"""Small NumPy regressor for controlled spectral-intervention experiments.

The model predicts a scalar conditional mean, not a Bayesian posterior. Input and
target scaling are fitted on training data only. No validation-driven stopping or
checkpoint selection is performed: every fit uses its declared update budget.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Parameters = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def _array(value: ArrayLike, name: str, ndim: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    if any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must not have empty dimensions")
    return result


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _unit_vector(vector: FloatArray, message: str) -> FloatArray:
    magnitude = float(np.max(np.abs(vector)))
    if magnitude == 0 or not np.isfinite(magnitude):
        raise ValueError(message)
    scaled = vector / magnitude
    return scaled / float(np.linalg.norm(scaled))


def _loss_and_gradients(
    x: FloatArray, y: FloatArray, parameters: Parameters
) -> tuple[float, Parameters]:
    """Mean squared loss and its analytic derivatives on standardized arrays."""
    w1, b1, w2, b2 = parameters
    activations = np.tanh(x @ w1 + b1)
    error = activations @ w2 + b2[0] - y
    output_gradient = 2.0 * error / y.size
    hidden_gradient = output_gradient[:, None] * w2 * (1.0 - activations**2)
    gradients = (
        x.T @ hidden_gradient,
        hidden_gradient.sum(axis=0),
        activations.T @ output_gradient,
        np.array([output_gradient.sum()], dtype=np.float64),
    )
    return float(np.mean(error**2)), gradients


@dataclass(frozen=True)
class TanhRegressor:
    """One hidden tanh layer with a scalar linear readout.

    All state is explicit float64 arrays. Use :meth:`state_dict` for a detached
    copy and :meth:`from_state_dict` to validate and restore a saved model.
    """

    w1: FloatArray
    b1: FloatArray
    w2: FloatArray
    b2: FloatArray
    x_mean: FloatArray
    x_scale: FloatArray
    y_mean: FloatArray
    y_scale: FloatArray

    def __post_init__(self) -> None:
        for name, ndim in (
            ("w1", 2),
            ("b1", 1),
            ("w2", 1),
            ("b2", 1),
            ("x_mean", 1),
            ("x_scale", 1),
            ("y_mean", 1),
            ("y_scale", 1),
        ):
            array = _array(getattr(self, name), name, ndim).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        features, width = self.w1.shape
        expected_shapes = {
            "b1": (width,),
            "w2": (width,),
            "b2": (1,),
            "x_mean": (features,),
            "x_scale": (features,),
            "y_mean": (1,),
            "y_scale": (1,),
        }
        for name, expected in expected_shapes.items():
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError("normalization scales must be strictly positive")

    def _inputs(self, x: ArrayLike) -> FloatArray:
        result = _array(x, "x", 2)
        if result.shape[1] != self.w1.shape[0]:
            raise ValueError(f"x must have {self.w1.shape[0]} features")
        return result

    def hidden(self, x: ArrayLike) -> FloatArray:
        """Return hidden activations using the fitted training normalization."""
        values = self._inputs(x)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                normalized = (values - self.x_mean) / self.x_scale
                return np.tanh(normalized @ self.w1 + self.b1)
            except FloatingPointError as exc:
                raise ValueError("x exceeds the model's numerical range") from exc

    def predict_from_hidden(self, activations: ArrayLike) -> FloatArray:
        """Apply the scalar readout and reverse the training target scaling."""
        values = _array(activations, "activations", 2)
        if values.shape[1] != self.w1.shape[1]:
            raise ValueError(f"activations must have {self.w1.shape[1]} features")
        with np.errstate(over="raise", invalid="raise"):
            try:
                result: FloatArray = (values @ self.w2 + self.b2[0]) * self.y_scale[
                    0
                ] + self.y_mean[0]
                return result
            except FloatingPointError as exc:
                raise ValueError("activations exceed the model's numerical range") from exc

    def predict(self, x: ArrayLike) -> FloatArray:
        return self.predict_from_hidden(self.hidden(x))

    def patched_predict(self, x: ArrayLike, direction: ArrayLike, center: ArrayLike) -> FloatArray:
        """Remove one hidden direction about a supplied discovery-set center.

        The direction is normalized here, so its original magnitude has no
        effect. This operation is an intervention, not a fitted correction.
        """
        vector = _array(direction, "direction", 1)
        origin = _array(center, "center", 1)
        expected = (self.w1.shape[1],)
        if vector.shape != expected or origin.shape != expected:
            raise ValueError(f"direction and center must have shape {expected}")
        unit = _unit_vector(vector, "direction must have a finite nonzero norm")
        values = self.hidden(x)
        patched = values - ((values - origin) @ unit)[:, None] * unit
        return self.predict_from_hidden(patched)

    def hidden_nuisance_direction(
        self, paired_x1: ArrayLike, paired_x2: ArrayLike, delta_b: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Fit a rank-one nuisance response using discovery pairs only.

        The caller must reserve these pairs for discovery. ``delta_b`` is b2-b1.
        We fit the zero-intercept least-squares slope of h(x2)-h(x1) against
        delta_b. The returned center is the pooled mean of both pair members.
        No target labels, evaluation pairs, or output readout enter this fit.
        """
        first = self._inputs(paired_x1)
        second = self._inputs(paired_x2)
        shifts = _array(delta_b, "delta_b", 1)
        if first.shape != second.shape or shifts.shape != (first.shape[0],):
            raise ValueError("paired inputs and delta_b must have matching sample counts")
        if first.shape[0] < 2:
            raise ValueError("direction fitting requires at least two discovery pairs")
        max_shift = float(np.max(np.abs(shifts)))
        if max_shift == 0:
            raise ValueError("delta_b must contain a finite nonzero intervention")
        # Rescaling leaves the slope direction unchanged and avoids squaring
        # extremely small or large physical units in the denominator.
        scaled_shifts = shifts / max_shift
        energy = float(scaled_shifts @ scaled_shifts)
        h1, h2 = self.hidden(first), self.hidden(second)
        slope = scaled_shifts @ (h2 - h1) / energy
        direction = _unit_vector(slope, "discovery pairs have no nonzero hidden nuisance response")
        return direction, 0.5 * (h1.mean(axis=0) + h2.mean(axis=0))

    def state_dict(self) -> dict[str, FloatArray]:
        """Return detached NumPy arrays suitable for ``np.savez``."""
        return {
            name: getattr(self, name).copy()
            for name in ("w1", "b1", "w2", "b2", "x_mean", "x_scale", "y_mean", "y_scale")
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, ArrayLike]) -> "TanhRegressor":
        names = {"w1", "b1", "w2", "b2", "x_mean", "x_scale", "y_mean", "y_scale"}
        if set(state) != names:
            raise ValueError(f"model state must contain exactly {sorted(names)}")
        arrays = {name: np.asarray(value, dtype=np.float64) for name, value in state.items()}
        return cls(**arrays)


def fit(
    x: ArrayLike,
    y: ArrayLike,
    seed: int,
    hidden: int = 32,
    steps: int = 800,
    batch_size: int = 256,
    learning_rate: float = 0.003,
) -> TanhRegressor:
    """Fit a deterministic scalar regressor with a fixed Adam update budget.

    Batches are sampled with replacement using a local seeded NumPy generator.
    Every declared update runs; there is no early stopping. Constant input
    columns or targets use unit scale. Determinism is within a numerical runtime;
    results across different BLAS implementations need not be bitwise identical.
    """
    inputs = _array(x, "x", 2)
    targets = _array(y, "y", 1)
    if inputs.shape[0] != targets.size or targets.size < 2:
        raise ValueError("x and y must contain at least two matching samples")
    for value, name in ((hidden, "hidden"), (steps, "steps"), (batch_size, "batch_size")):
        _positive_integer(value, name)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            x_mean, x_scale = inputs.mean(axis=0), inputs.std(axis=0)
            x_scale = np.where(x_scale > np.finfo(np.float64).eps, x_scale, 1.0)
            y_mean = np.array([targets.mean()], dtype=np.float64)
            y_scale = np.array([targets.std()], dtype=np.float64)
            y_scale = np.where(y_scale > np.finfo(np.float64).eps, y_scale, 1.0)
            normalized_x = (inputs - x_mean) / x_scale
            normalized_y = (targets - y_mean[0]) / y_scale[0]
        except FloatingPointError as exc:
            raise ValueError("training values exceed the normalization numerical range") from exc

    rng = np.random.default_rng(seed)
    features = inputs.shape[1]
    parameters: Parameters = (
        rng.normal(0.0, np.sqrt(1.0 / features), (features, hidden)),
        np.zeros(hidden, dtype=np.float64),
        rng.normal(0.0, np.sqrt(1.0 / hidden), hidden),
        np.zeros(1, dtype=np.float64),
    )
    first_moment = [np.zeros_like(value) for value in parameters]
    second_moment = [np.zeros_like(value) for value in parameters]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for step in range(1, steps + 1):
                indices = rng.integers(0, targets.size, size=batch_size)
                _, gradients = _loss_and_gradients(
                    normalized_x[indices], normalized_y[indices], parameters
                )
                for parameter, gradient, moment1, moment2 in zip(
                    parameters, gradients, first_moment, second_moment, strict=True
                ):
                    moment1 *= 0.9
                    moment1 += 0.1 * gradient
                    moment2 *= 0.999
                    moment2 += 0.001 * gradient**2
                    corrected1 = moment1 / (1.0 - 0.9**step)
                    corrected2 = moment2 / (1.0 - 0.999**step)
                    parameter -= learning_rate * corrected1 / (np.sqrt(corrected2) + 1e-8)
        except FloatingPointError as exc:
            raise ValueError("training diverged within the declared update budget") from exc
    return TanhRegressor(*parameters, x_mean, x_scale, y_mean, y_scale)

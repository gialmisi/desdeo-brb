"""Main BRB model class with an sklearn-compatible interface.

Provides the `BRBModel` class which supports fitting, predicting, and
inspecting a Belief Rule-Based inference system.
"""

from typing import Any, Callable

import numpy as np

from desdeo_brb.models import InferenceResult, RuleBase


class BRBModel:
    """A trainable Belief Rule-Based inference model.

    Implements an sklearn-compatible interface (`fit`, `predict`, `score`,
    `get_params`, `set_params`) for building and using BRB systems.

    Attributes:
        rule_base: The underlying RuleBase specification.
    """

    def __init__(self, rule_base: RuleBase | None = None) -> None:
        """Initialize the BRB model.

        Args:
            rule_base: An optional pre-configured RuleBase. If None, the model
                must be configured before fitting.
        """
        self.rule_base = rule_base

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BRBModel":
        """Fit the model by optimizing BRB parameters to minimize MSE.

        Args:
            X: Training input array of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            self
        """
        pass

    def fit_custom(
        self, loss_fn: Callable, X: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> "BRBModel":
        """Fit the model using a user-supplied loss function.

        Args:
            loss_fn: A callable loss function with signature
                ``loss_fn(y_true, y_pred) -> float``.
            X: Training input array of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            **kwargs: Additional keyword arguments passed to the optimizer.

        Returns:
            self
        """
        pass

    def predict(self, X: np.ndarray) -> InferenceResult:
        """Run inference on input data.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            An InferenceResult containing full inference details.
        """
        pass

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters (sklearn-compatible).

        Args:
            deep: If True, return nested parameters.

        Returns:
            Dictionary of parameter names to values.
        """
        pass

    def set_params(self, **params: Any) -> "BRBModel":
        """Set model parameters (sklearn-compatible).

        Args:
            **params: Parameter names and values.

        Returns:
            self
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the coefficient of determination (R²) for the prediction.

        Args:
            X: Input array of shape (n_samples, n_features).
            y: True target values of shape (n_samples,).

        Returns:
            R² score.
        """
        pass

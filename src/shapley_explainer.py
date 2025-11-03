import numpy as np
from typing import Any, Callable, Iterable
from math import factorial
from itertools import chain, combinations


class ShapleyExplainer:
    """Shapley Values implementation from scratch.

    This class provides a simple implementation of Shapley values for model
    interpretability. It estimates the contribution of each feature to the
    model's prediction by computing Shapley values across all possible
    feature subsets.

    Attributes:
        model (Callable[[np.ndarray], float]):
            The predictive model to be explained. Must accept a NumPy array
            and return a prediction vector or scalar (e.g., `.predict`).
        background_dataset (np.ndarray):
            Background dataset used for estimating feature contributions.
            Typically a representative sample of the input data.
    """

    def __init__(self, model: Callable[[np.ndarray], np.ndarray], background_dataset: np.ndarray) -> None:
        """Initializes the Shapley explainer.

        Args:
            model (Callable[[np.ndarray], np.ndarray]):
                The model to explain (e.g., `estimator.predict`).
            background_dataset (np.ndarray):
                The dataset used as a background reference for computing Shapley values.
                Shape should be (n_background, n_features).
        """
        # Store the model and background dataset
        self.model = model
        self.background_dataset = background_dataset

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute Shapley values for each instance and feature.

        Args:
            X (np.ndarray):
                Input samples for which Shapley values are computed (shape: n_instances x n_features).

        Returns:
            np.ndarray:
                A 2D array of Shapley values with the same shape as `X`.
        """
        n_instances, n_features = X.shape
        shapley_vals = np.zeros((n_instances, n_features))

        # Iterate over all instances and features
        for i in range(n_instances):
            for j in range(n_features):
                shapley_vals[i, j] = self._compute_single_shapley_value(j, X[i, :])

        return shapley_vals

    def _compute_single_shapley_value(self, feature: int, instance: np.ndarray) -> float:
        """Compute the Shapley value for a single feature in one instance.

        Implements the Shapley formula (weighted average of marginal contributions)
        across all subsets that do not include the current feature.

        Args:
            feature (int):
                Index of the feature for which the Shapley value is computed.
            instance (np.ndarray):
                The input instance (1D, length = n_features).

        Returns:
            float:
                The Shapley value for the given feature.
        """
        n_features = len(instance)
        shapley_value = 0.0

        # Iterate over all subsets S that do not include the feature of interest
        for subset in self._get_all_other_feature_subsets(n_features, feature):
            # Compute the model output with and without the feature
            w_f = self._subset_model_approximation(subset + (feature,), instance)
            wo_f = self._subset_model_approximation(subset, instance)

            # Compute the marginal contribution
            marginal_contribution = w_f - wo_f
            # Compute the permutation factor
            weight = self._permutation_factor(n_features, len(subset))

            # Accumulate the weighted marginal contribution
            shapley_value += weight * marginal_contribution
        return shapley_value

    def _get_all_subsets(self, items: list[int]) -> Iterable[tuple[int, ...]]:
        """Generate all subsets of a list.

        Args:
            items (list[int]):
                List of feature indices.

        Returns:
            Iterable[tuple[int, ...]]:
                Iterator over all possible subsets (including empty and full).
        """
        # Obtain an iterator over all subsets
        return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))

    def _get_all_other_feature_subsets(self, n_features: int, feature_of_interest: int) -> Iterable[tuple[int, ...]]:
        """Generate all subsets of features excluding the feature of interest.

        Args:
            n_features (int):
                Total number of features.
            feature_of_interest (int):
                Index of the feature to exclude.

        Returns:
            Iterable[tuple[int, ...]]:
                Iterator of feature index subsets not containing the feature of interest.
        """
        # Create a list of all feature indices and exclude the feature of interest
        indices = [i for i in range(n_features) if i != feature_of_interest]

        return self._get_all_subsets(indices)


    def _permutation_factor(self, n_features: int, n_subset: int) -> float:
        """Compute the permutation weighting factor for a subset.

        This factor ensures fair averaging across feature subsets
        in the Shapley value computation.

        Args:
            n_features (int):
                Total number of features.
            n_subset (int):
                Number of features in the subset (|S|).

        Returns:
            float:
                Permutation weight for the subset:
                |S|! * (M - |S| - 1)! / M!  where M = n_features.
        """
        # |S| = n_subset, |M| = n_features
        return factorial(n_subset) * factorial(n_features - n_subset - 1) / factorial(n_features)

    def _subset_model_approximation(self, feature_subset: tuple[int, ...], instance: np.ndarray) -> float:
        """Approximate the model output conditioned on a subset of features.

        This simulates E[f(X) | X_S = instance_S] by:
        - Copying the background dataset.
        - Overwriting the columns in S with the corresponding values from `instance`.
        - Predicting on the modified background.
        - Returning the mean prediction as the conditional expectation.

        Args:
            feature_subset (tuple[int, ...]):
                Indices of the features to condition on.
            instance (np.ndarray):
                Instance whose feature values are used to overwrite the background.

        Returns:
            float:
                The mean model output given the subset of features.
        """
        # Create a copy of the background dataset to avoid modifying the original
        backg_c = self.background_dataset.copy()

        # Overwrite the columns in feature_subset with values from instance
        for feature in feature_subset:
            backg_c[:, feature] = instance[feature]

        # Predict on the modified background dataset
        predictions = self.model(backg_c)

        # Obtain conditional expectation
        return np.mean(predictions)

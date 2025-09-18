"""
This file is used to provide an easy way for BanditWare to use different models in the backend.
It does so by providing an enum of model types called `Model` that all define
a `fit` and `predict` function, as well as a `has_fit` member variable.
BanditWare can use any model from this Model enum to create predictions.

The Model enum can easily extend to use any scikit-learn model using the `SklearnWrapper` class.
It can also use TensorFlow's Keras using the `KerasReg` class.
If this file is extended to support more models/libraries, all classes must extend `BanditModel` to:
    - accept all hyperparameters via `__init__(**kwargs)`
    - implement `fit(X: np.ndarray, y: np.ndarray) -> Self`
        (not including `Self` type annotation for compatibility with python < 3.11)
    - implement `predict(X: np.ndarray) -> np.ndarray`
    - implement `has_fit() -> bool` method that returns False until after `.fit()` is called.
Then, add the new model to the `Model` enum
"""

from functools import partial
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

# scikit-learn imports
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    SGDRegressor,
    BayesianRidge,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR as skSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Optional: TensorFlow/Keras import for neural network wrapper
# TensorFlow isn't needed for many applications, so it's okay if the user doesn't want to import it
try:
    import tensorflow as tf

    tf.config.run_functions_eagerly(True)
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
except ImportError:
    tf = None

# enum elements expect a `member` wrapper in future python versions.
# `member` was not defined until python 3.11; add a dummy member definition if on < 3.11
try:
    from enum import member
except ImportError:

    def member(value):
        return value


# ============================
# Base class: Must extend this
# ============================


class ModelInterface(ABC):
    """All model wrappers must implement these three methods."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Given the X feature matrix and y result vector (training data), fit the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Given the X feature matrix (testing data), predict the y result vector."""

    @abstractmethod
    def has_fit(self) -> bool:
        """Return True once `.fit(...)` has been called at least once."""


# ============================
#         Sub classes
# ============================


class SklearnWrapper(BaseEstimator, ModelInterface):
    """
    Wrap any scikit-learn regressor to conform to BanditWare's model interface.
    Accepts the estimator class and its hyperparameters at init time.
    In the MODEL_REGISTRY, the estimator class is set, but params are left to be set.
    """

    def __init__(self, model_cls, **params):
        self.model_cls = model_cls
        self.params = params
        self._has_fit = False
        self._model = self.model_cls(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)
        self._has_fit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert (
            self._has_fit
        ), "Model has not yet fit. `.fit()` must be called before `.predict()`"
        return self._model.predict(X)

    def has_fit(self) -> bool:
        return self._has_fit


class KerasReg(BaseEstimator, ModelInterface):
    """
    Simple feed-forward Keras Neural Network.
    Requires TensorFlow. Builds model in fit() when data shape is known.
    """

    def __init__(
        self,
        activation="relu",
        optimizer=None,
        epochs=100,
        batch_size=32,
        layers=5,
        **kwargs
    ):
        if tf is None:
            raise ImportError("TensorFlow is required for KerasReg")
        # store architecture and training params
        self.activation = activation
        self.optimizer_cls = optimizer.__class__ if optimizer else Adam
        self.optimizer_config = optimizer.get_config() if optimizer else {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.layers = layers
        self._has_fit = False
        self._model = None

    def build_model(self, n_features: int):
        # create a fresh model and optimizer each time
        model = Sequential()
        model.add(Input(shape=(n_features,)))
        for units in self.layers:
            model.add(Dense(units, activation=self.activation, **self.kwargs))
        model.add(Dense(1))
        optimizer = self.optimizer_cls(**self.optimizer_config)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self._model = self.build_model(n_features)
        self._model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self._has_fit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).flatten()

    def has_fit(self) -> bool:
        return self._has_fit


# Enum of all potential models and their constructors.
# Uses partial() to accept hyperparameters later
class Model(Enum):
    """All models as an Enum that BanditWare can use"""

    LINEAR_REGRESSION = member(partial(SklearnWrapper, LinearRegression))
    RIDGE = member(partial(SklearnWrapper, Ridge))
    LASSO = member(partial(SklearnWrapper, Lasso))
    BAYESIAN_RIDGE = member(partial(SklearnWrapper, BayesianRidge))
    SGD = member(partial(SklearnWrapper, SGDRegressor))
    # Decision tree not recommended: creates step function rather than smooth line for runtime prediction
    DECISION_TREE = member(partial(SklearnWrapper, DecisionTreeRegressor))
    RANDOM_FOREST = member(partial(SklearnWrapper, RandomForestRegressor))
    GRADIENT_BOOSTING = member(partial(SklearnWrapper, GradientBoostingRegressor))
    MLP = member(partial(SklearnWrapper, MLPRegressor))
    # KERAS = member(KerasReg)

    def create(self, **kwargs):
        """
        Instantiate the wrapped model with the given hyperparameters.
        Example:
            model = Model.RANDOM_FOREST.create(n_estimators=100, max_depth=5)
        """
        return self.value(**kwargs)

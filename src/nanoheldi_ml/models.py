"""Prespecified model families wrapped in fold-local preprocessing pipelines."""

from __future__ import annotations

from collections import OrderedDict

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import REPRODUCIBILITY_RANDOM_STATE


def _classification_pipeline(classifier: BaseEstimator, n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=n_features)),
            ("model", classifier),
        ]
    )


def classification_models(n_features: int, include_xgboost: bool = False) -> OrderedDict:
    """Return models with one fixed random state; no state is searched or ranked."""
    state = REPRODUCIBILITY_RANDOM_STATE
    models = OrderedDict(
        [
            (
                "Random",
                _classification_pipeline(
                    DummyClassifier(strategy="stratified", random_state=state), n_features
                ),
            ),
            ("NB", _classification_pipeline(GaussianNB(), n_features)),
            (
                "DT",
                _classification_pipeline(
                    DecisionTreeClassifier(class_weight="balanced", random_state=state), n_features
                ),
            ),
            (
                "LR",
                _classification_pipeline(
                    LogisticRegression(
                        class_weight="balanced", max_iter=5000, random_state=state
                    ),
                    n_features,
                ),
            ),
            (
                "GB",
                _classification_pipeline(
                    GradientBoostingClassifier(random_state=state), n_features
                ),
            ),
            (
                "SVM",
                _classification_pipeline(
                    SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced"),
                    n_features,
                ),
            ),
            (
                "RF",
                _classification_pipeline(
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced_subsample",
                        random_state=state,
                        n_jobs=1,
                    ),
                    n_features,
                ),
            ),
            (
                "MLP",
                _classification_pipeline(
                    MLPClassifier(
                        hidden_layer_sizes=(100,),
                        activation="relu",
                        solver="adam",
                        alpha=0.01,
                        max_iter=2000,
                        early_stopping=False,
                        random_state=state,
                    ),
                    n_features,
                ),
            ),
        ]
    )

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install the optional 'xgboost' dependency") from exc
        models["XGBoost"] = _classification_pipeline(
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=state,
                n_jobs=1,
                verbosity=0,
            ),
            n_features,
        )
    return models


def regression_models(n_features: int) -> OrderedDict:
    state = REPRODUCIBILITY_RANDOM_STATE
    return OrderedDict(
        [
            (
                "Mean",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("select", SelectKBest(score_func=f_regression, k=n_features)),
                        ("model", DummyRegressor(strategy="mean")),
                    ]
                ),
            ),
            (
                "MLP",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("select", SelectKBest(score_func=f_regression, k=n_features)),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=(100,),
                                activation="relu",
                                solver="adam",
                                alpha=0.01,
                                max_iter=2000,
                                early_stopping=False,
                                random_state=state,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )

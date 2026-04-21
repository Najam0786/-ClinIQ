"""
core/model_selector.py
-----------------------
Trains Random Forest, Logistic Regression, and Gradient Boosting.
Selects the best model by cross-validated ROC-AUC.
Returns predictions, probabilities, feature importances, and metrics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve,
)
from sklearn.inspection import permutation_importance


CANDIDATE_MODELS = {
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=400, max_depth=6, learning_rate=0.04,
        min_samples_leaf=20, l2_regularization=0.1,
        random_state=42, class_weight="balanced",
        early_stopping=False,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=4,
        max_features="sqrt", random_state=42,
        class_weight="balanced", n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=4,
        max_features="sqrt", random_state=42,
        class_weight="balanced", n_jobs=-1,
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000, C=0.3, solver="saga",
        random_state=42, class_weight="balanced",
    ),
}


@dataclass
class ModelResult:
    best_model_name: str
    cv_scores: Dict[str, float]         # model_name → mean CV AUC
    test_auc: float
    classification_report: str
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    feature_importances: pd.DataFrame   # feature | importance
    predictions_df: pd.DataFrame        # original index + prediction + probability
    pipeline: Pipeline
    numeric_cols: List[str]             = field(default_factory=list)
    categorical_cols: List[str]         = field(default_factory=list)
    X_train_sample: Optional[pd.DataFrame] = None  # small sample for SHAP background


def _build_sklearn_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    model,
) -> Pipeline:
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer,  numeric_cols),
        ("cat", cat_transformer,  categorical_cols),
    ], remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def _get_feature_names(pipeline: Pipeline, numeric_cols, categorical_cols) -> List[str]:
    ohe = (pipeline.named_steps["preprocessor"]
           .named_transformers_["cat"]
           .named_steps["encoder"])
    try:
        cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    except Exception:
        cat_feature_names = []
    return numeric_cols + cat_feature_names


def _get_importances(
    pipeline: Pipeline,
    feature_names: List[str],
    X_test: pd.DataFrame = None,
    y_test: pd.Series   = None,
) -> pd.DataFrame:
    clf = pipeline.named_steps["classifier"]
    imp = None

    # 1 — native tree importances (Random Forest, Extra Trees, HGBT >= sklearn 1.0)
    if hasattr(clf, "feature_importances_"):
        try:
            imp = clf.feature_importances_
            if imp is None or len(imp) == 0:
                imp = None
        except Exception:
            imp = None

    # 2 — linear model coefficients
    if imp is None and hasattr(clf, "coef_"):
        try:
            imp = np.abs(clf.coef_[0])
        except Exception:
            imp = None

    # 3 — permutation importance fallback (works for any fitted estimator)
    if imp is None and X_test is not None and y_test is not None:
        try:
            r = permutation_importance(
                pipeline, X_test, y_test,
                n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=-1,
            )
            imp = r.importances_mean
        except Exception:
            imp = None

    if imp is None or len(feature_names) == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(feature_names), len(imp))
    if n == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names[:n], "importance": imp[:n]})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _adaptive_models(n_rows: int) -> Dict[str, Any]:
    """
    Return a model dict tuned to dataset size.
    Large datasets get fewer estimators and shallower trees so CV finishes
    in seconds rather than minutes.
    """
    if n_rows > 30_000:
        # Very large: lightning-fast settings
        return {
            "Hist Gradient Boosting": HistGradientBoostingClassifier(
                max_iter=100, max_depth=4, learning_rate=0.08,
                min_samples_leaf=30, random_state=42,
                class_weight="balanced", early_stopping=False,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=80, max_depth=8, min_samples_leaf=10,
                max_features="sqrt", random_state=42,
                class_weight="balanced", n_jobs=-1,
                max_samples=min(10_000, n_rows) / n_rows,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=500, C=0.3, solver="saga",
                random_state=42, class_weight="balanced",
            ),
        }
    elif n_rows > 10_000:
        # Medium-large: reduced estimators
        return {
            "Hist Gradient Boosting": HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.06,
                min_samples_leaf=25, random_state=42,
                class_weight="balanced", early_stopping=False,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=6,
                max_features="sqrt", random_state=42,
                class_weight="balanced", n_jobs=-1,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=6,
                max_features="sqrt", random_state=42,
                class_weight="balanced", n_jobs=-1,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=0.3, solver="saga",
                random_state=42, class_weight="balanced",
            ),
        }
    else:
        return CANDIDATE_MODELS


def select_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    numeric_cols     = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    n_rows   = len(X_train)
    models   = _adaptive_models(n_rows)
    n_splits = 3 if n_rows > 15_000 else 5
    cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores: Dict[str, float] = {}

    for name, model in models.items():
        pipe = _build_sklearn_pipeline(numeric_cols, categorical_cols, model)
        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                     scoring="roc_auc", n_jobs=-1)
            cv_scores[name] = round(float(scores.mean()), 4)
        except Exception:
            cv_scores[name] = 0.0

    best_name  = max(cv_scores, key=cv_scores.get)
    best_model = models[best_name]

    final_pipe = _build_sklearn_pipeline(numeric_cols, categorical_cols, best_model)
    final_pipe.fit(X_train, y_train)

    y_pred  = final_pipe.predict(X_test)
    y_proba = final_pipe.predict_proba(X_test)[:, 1]
    auc     = round(float(roc_auc_score(y_test, y_proba)), 4)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    report = classification_report(
        y_test, y_pred,
        target_names=["Low Risk", "High Risk"],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    feature_names = _get_feature_names(final_pipe, numeric_cols, categorical_cols)
    importances   = _get_importances(final_pipe, feature_names, X_test, y_test)

    pred_df = X_test.copy()
    pred_df["prediction"]       = y_pred
    pred_df["risk_label"]       = ["High Risk" if p == 1 else "Low Risk" for p in y_pred]
    pred_df["risk_probability"] = y_proba.round(4)
    pred_df["actual"]           = y_test.values

    return ModelResult(
        best_model_name=best_name,
        cv_scores=cv_scores,
        test_auc=auc,
        classification_report=report,
        confusion_matrix=cm,
        fpr=fpr,
        tpr=tpr,
        feature_importances=importances,
        predictions_df=pred_df,
        pipeline=final_pipe,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        X_train_sample=X_train.head(150).reset_index(drop=True),
    )

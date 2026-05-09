"""Factory for classical ML models."""

from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def build_classical(name: str, n_classes: int, seed: int = 42):
    """Build a classical ML classifier.

    All parallelisation knobs (`n_jobs`, OpenMP thread counts, etc.) are left
    at their library defaults to avoid environment-specific surprises.
    """
    name = name.lower()
    if name == "lr":
        return LogisticRegression(
            max_iter=2000, C=1.0, random_state=seed,
        )
    if name == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale",
                   probability=True, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None,
            random_state=seed,
        )
    if name == "xgb":
        # Try XGBoost first; on macOS it depends on libomp which is often
        # missing (`brew install libomp`). Fall back to sklearn's GBM silently.
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                tree_method="hist", random_state=seed,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                verbosity=0,
            )
            return clf
        except Exception:
            return GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1,
                max_depth=3, random_state=seed,
            )
    raise ValueError(f"Unknown classical model '{name}'")

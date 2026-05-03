"""Heuristic Feature Selection Framework
- Accepts a pre-processed dataset (X, y).
- Profiles dataset meta-features (p/n ratio, imbalance, correlation, etc.).
- Uses rule-based heuristics to choose a feature selection (FS) family and method.
- Applies the chosen method to select a feature subset.
- Returns the reduced DataFrame and metadata about the selection.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression


# Result container
@dataclass
class HeuristicFSResult:
    X_selected: pd.DataFrame
    selected_features: List[str]
    fs_family: str
    fs_method: str
    fs_params: Dict
    meta_profile: Dict
    fs_runtime: float
    n_features_before: int
    n_features_after: int


class HeuristicFeatureSelector:
    """
    The selector:
    - Profiles the dataset.
    - Chooses a feature selection family (filter / wrapper / embedded).
    - Runs the chosen method.
    - Stores the result and exposes `transform` for new data.
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        frac_features: Optional[float] = None,
        corr_threshold: float = 0.8,
        high_dim_ratio: float = 1.5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.max_features = max_features
        self.frac_features = frac_features
        self.corr_threshold = corr_threshold
        self.high_dim_ratio = high_dim_ratio
        self.random_state = random_state
        self.verbose = verbose

        self.result_: Optional[HeuristicFSResult] = None
        self._selected_features: Optional[List[str]] = None

    def _profile_dataset(self, X: pd.DataFrame, y) -> Dict:
        n_samples, n_features = X.shape

        # class distribution
        y_series = pd.Series(y)
        class_counts = y_series.value_counts(normalize=True)
        class_imbalance = float(class_counts.max())

        # numeric columns for correlation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:  
            corr = X[numeric_cols].corr().abs()
            mask = ~np.eye(corr.shape[0], dtype=bool)
            mean_corr = float(corr.where(mask).stack().mean())
        else:
            mean_corr = 0.0

        # dimensionality ratio p/n
        p_to_n_ratio = n_features / n_samples if n_samples > 0 else np.inf
        task_type = "binary" if class_counts.size == 2 else "multiclass"

        # variance structure (mean variance across features)
        variances = X.var(ddof=0)
        mean_variance = float(variances.mean())

        # sparsity (fraction of zeros)
        total_entries = n_samples * n_features
        zero_fraction = float((X == 0).sum().sum() / total_entries) if total_entries > 0 else 0.0

        meta = {
            "n_samples": n_samples,
            "n_features": n_features,
            "p_to_n_ratio": p_to_n_ratio,
            "task_type": task_type,
            "class_imbalance": class_imbalance,
            "mean_abs_correlation": mean_corr,
            "mean_variance": mean_variance,
            "sparsity": zero_fraction,
        }

        if self.verbose:
            print("[HeuristicFS] Meta-profile:", meta)

        return meta

    def _determine_k(self, meta: Dict) -> int:
        n_features = meta["n_features"]
        p_to_n = meta["p_to_n_ratio"]

        if self.max_features is not None:
            k = min(self.max_features, n_features)

        elif self.frac_features is not None:
            k = int(np.ceil(self.frac_features * n_features))

        elif p_to_n > self.high_dim_ratio:
            k = int(max(50, min(0.05 * n_features, 300)))

            if self.verbose:
                print(f"[HeuristicFS] High-dimensional dataset detected (p/n={p_to_n:.2f}). "
                    f"Selecting k={k} features (~5%).")

        else:
            k = max(30, int(0.5 * n_features))

        k = min(k, n_features)

        if self.verbose:
            print(f"[HeuristicFS] Final k={k} for FS")

        return k

    def _choose_strategy(self, meta: Dict) -> Tuple[str, str, Dict]:
        k = self._determine_k(meta)
        p_to_n = meta["p_to_n_ratio"]
        mean_corr = meta["mean_abs_correlation"]

        if p_to_n > self.high_dim_ratio:
            fs_family = "filter"
            fs_method = "f_classif_kbest"
            fs_params = {"k": k}

        elif mean_corr >= self.corr_threshold:
            fs_family = "embedded"
            fs_method = "lasso_logreg"
            fs_params = {"k": k, "C": 1.0}

        else:
            fs_family = "wrapper"
            fs_method = "rfe_logreg"
            fs_params = {"n_features_to_select": k, "step": 0.1}

        if self.verbose:
            print(f"[HeuristicFS] Chosen strategy: {fs_family} / {fs_method} with params {fs_params}")

        return fs_family, fs_method, fs_params

    def _run_filter(self, X: pd.DataFrame, y, method: str, params: Dict):
        k = params.get("k", X.shape[1])

        if method == "chi2_kbest":
            selector = SelectKBest(score_func=chi2, k=k)
        elif method == "f_classif_kbest":
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == "mutual_info_kbest":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown filter method: {method}")

        selector.fit(X, y)
        mask = selector.get_support()
        selected_features = list(X.columns[mask])
        X_selected = X[selected_features].copy()
        return X_selected, selected_features

    def _run_wrapper(self, X: pd.DataFrame, y, method: str, params: Dict):
        if method != "rfe_logreg":
            raise ValueError(f"Unknown wrapper method: {method}")

        n_features_to_select = params.get("n_features_to_select", X.shape[1])
        step = params.get("step", 0.1)

        base_estimator = LogisticRegression(
            penalty="l2",
               solver="lbfgs",
            random_state=self.random_state,
            max_iter=1000,
        )
        selector = RFE(
            estimator=base_estimator,
            n_features_to_select=n_features_to_select,
            step=step,
        )
        selector.fit(X, y)
        mask = selector.get_support()
        selected_features = list(X.columns[mask])
        X_selected = X[selected_features].copy()
        return X_selected, selected_features

    def _run_embedded(self, X: pd.DataFrame, y, method: str, params: Dict):
        if method != "lasso_logreg":
            raise ValueError(f"Unknown embedded method: {method}")

        C = params.get("C", 1.0)

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            random_state=self.random_state,
            max_iter=2000,
        )
        model.fit(X, y)

        coefs = model.coef_
        if coefs.ndim == 2:
            importance = np.mean(np.abs(coefs), axis=0)
        else:
            importance = np.abs(coefs)

        importance = np.asarray(importance)
        n_features = X.shape[1]

        meta_tmp = {"n_features": n_features}
        k = self._determine_k(meta_tmp)

        idx_sorted = np.argsort(importance)[::-1]
        selected_idx = idx_sorted[:k]
        selected_features = list(X.columns[selected_idx])
        X_selected = X[selected_features].copy()

        return X_selected, selected_features

    def _run_fs(self, X: pd.DataFrame, y, fs_family: str, fs_method: str, fs_params: Dict):
        if fs_family == "filter":
            return self._run_filter(X, y, fs_method, fs_params)
        elif fs_family == "wrapper":
            return self._run_wrapper(X, y, fs_method, fs_params)
        elif fs_family == "embedded":
            return self._run_embedded(X, y, fs_method, fs_params)
        else:
            raise ValueError(f"Unknown FS family: {fs_family}")

    def fit_transform(self, X: pd.DataFrame, y) -> HeuristicFSResult:
        start = time.time()

        meta = self._profile_dataset(X, y)
        fs_family, fs_method, fs_params = self._choose_strategy(meta)
        X_selected, selected_features = self._run_fs(X, y, fs_family, fs_method, fs_params)

        runtime = time.time() - start

        result = HeuristicFSResult(
            X_selected=X_selected,
            selected_features=selected_features,
            fs_family=fs_family,
            fs_method=fs_method,
            fs_params=fs_params,
            meta_profile=meta,
            fs_runtime=runtime,
            n_features_before=X.shape[1],
            n_features_after=len(selected_features),
        )

        self.result_ = result
        self._selected_features = selected_features

        if self.verbose:
            print(
                f"[HeuristicFS] Selected {len(selected_features)} / {X.shape[1]} features "
                f"in {runtime:.3f} seconds."
            )

        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            raise RuntimeError("HeuristicFeatureSelector has not been fitted yet.")

        missing = set(self._selected_features) - set(X.columns)
        if missing:
            raise ValueError(f"Input X is missing features: {missing}")

        return X[self._selected_features].copy()


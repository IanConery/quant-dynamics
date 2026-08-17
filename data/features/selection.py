"""Feature selection, permutation importance, multi-threshold search, and scale invariance auditing."""

import io
import multiprocessing
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from config.settings import FEATURE_SELECTION
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _perm_importance_worker(args):
    """Module-level worker for parallel permutation importance."""
    import joblib
    idx, X_test, y_test, baseline_score, model_bytes = args
    model = joblib.load(io.BytesIO(model_bytes))
    X_perm = X_test.copy()
    np.random.seed(42)
    X_perm[:, idx] = np.random.permutation(X_perm[:, idx])
    return idx, baseline_score - model.score(X_perm, y_test)


def select_features(X: np.ndarray, y_reg: np.ndarray, feature_names: List[str],
                     method: str = None, n_top: int = None) -> Tuple[np.ndarray, List[str]]:
    """Select most predictive features via permutation importance."""
    if method is None:
        method = FEATURE_SELECTION["method"]
    if method == "none":
        return X, feature_names

    if method == "permutation":
        split = int(len(X) * 0.85)
        X_train, X_test = X[:split], X[split:]
        y_train = y_reg[:split]
        y_test = y_reg[split:]

        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        baseline_score = model.score(X_test, y_test)
        n_features = X.shape[1]

        import joblib
        buf = io.BytesIO()
        joblib.dump(model, buf)
        model_bytes = buf.getvalue()
        n_jobs = min(multiprocessing.cpu_count(), max(2, n_features // 2))

        if n_jobs <= 1:
            importances = []
            for i in range(n_features):
                X_perm = X_test.copy()
                np.random.seed(42)
                X_perm[:, i] = np.random.permutation(X_perm[:, i])
                perm_score = model.score(X_perm, y_test)
                importances.append(baseline_score - perm_score)
        else:
            tasks = [(i, X_test, y_test, baseline_score, model_bytes) for i in range(n_features)]
            with multiprocessing.Pool(processes=n_jobs) as pool:
                results = pool.map(_perm_importance_worker, tasks)
            results.sort(key=lambda x: x[0])
            importances = [r[1] for r in results]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        threshold = FEATURE_SELECTION.get("min_importance_threshold", 0.001)
        kept = importance_df[importance_df["importance"] >= threshold]
        if n_top is not None:
            kept = kept.head(n_top)

        selected = kept["feature"].tolist()
        selected_indices = [feature_names.index(f) for f in selected]
        logger.info(f"Feature selection ({method}): kept {len(selected)}/{len(feature_names)} features")
        return X[:, selected_indices], selected

    return X, feature_names


def select_features_multi_threshold(
    X: np.ndarray,
    y_reg: np.ndarray,
    feature_names: List[str],
    thresholds: List[int] = None,
) -> pd.DataFrame:
    """Evaluate performance across multiple feature counts."""
    if thresholds is None:
        max_features = min(len(feature_names), 50)
        thresholds = [8, 10, 15, 20, 30, max_features]
        thresholds = sorted(set([t for t in thresholds if t <= max_features]))

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_reg[:split], y_reg[split:]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    baseline_score = model.score(X_test, y_test)
    n_features = X.shape[1]
    import joblib
    buf = io.BytesIO()
    joblib.dump(model, buf)
    model_bytes = buf.getvalue()
    n_jobs = min(multiprocessing.cpu_count(), max(2, n_features // 2))

    if n_jobs <= 1:
        importances = []
        for i in range(n_features):
            X_perm = X_test.copy()
            np.random.seed(42)
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
            perm_score = model.score(X_perm, y_test)
            importances.append(baseline_score - perm_score)
    else:
        tasks = [(i, X_test, y_test, baseline_score, model_bytes) for i in range(n_features)]
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = pool.map(_perm_importance_worker, tasks)
        results.sort(key=lambda x: x[0])
        importances = [r[1] for r in results]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    ranked_features = importance_df["feature"].tolist()
    results = []

    for n_top in thresholds:
        n_top = min(n_top, len(feature_names))
        selected = ranked_features[:n_top]
        indices = [feature_names.index(f) for f in selected if f in feature_names]

        X_sel_train = X_train[:, indices]
        X_sel_test = X_test[:, indices]

        try:
            eval_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            eval_model.fit(X_sel_train, y_train)
            r2 = float(eval_model.score(X_sel_test, y_test))
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Feature selection scoring failed for {n_top} features: {e}")
            r2 = float("nan")

        results.append({
            "n_features": n_top,
            "r2": round(r2, 6),
            "features_kept": "; ".join(selected),
        })

    report = pd.DataFrame(results)
    best_idx = report["r2"].idxmax() if not report["r2"].isna().all() else 0
    best_n = int(report.loc[best_idx, "n_features"])
    best_r2 = float(report.loc[best_idx, "r2"])
    logger.info(f"Multi-threshold feature selection: best = top {best_n} features (R² = {best_r2:.6f})")
    return report


def audit_scale_invariance(
    df: pd.DataFrame,
    close_col: str = "close",
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Audit features for scale invariance by checking correlation with raw price."""
    price = df[close_col].dropna()

    exclude_cols = {
        close_col, "timestamp", "open", "high", "low", "volume",
        "btc_close", "eth_close",
    }
    exclude_prefixes = [
        "reg_target_", "clf_target_", "ternary_target_", "vol_reg_target_",
    ]

    results = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if df[col].dtype not in (np.float64, np.float32, np.int64, np.int32, float, int, bool):
            continue

        series = df[col].dropna()
        common_idx = price.index.intersection(series.index)
        if len(common_idx) < 30:
            continue

        p = price.loc[common_idx].values.astype(np.float64)
        s = series.loc[common_idx].values.astype(np.float64)

        if np.std(s) == 0 or np.std(p) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(p, s)[0, 1])

        abs_corr = abs(corr)
        flagged = abs_corr > threshold

        if flagged:
            rec = "Convert to ratio/return. E.g., (close - sma) / sma instead of sma raw."
        else:
            rec = "OK — scale-invariant"

        results.append({
            "feature": col,
            "abs_corr": round(abs_corr, 4),
            "flagged": flagged,
            "recommendation": rec,
        })

    audit_df = pd.DataFrame(results)
    if len(audit_df) > 0:
        audit_df = audit_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    n_flagged = int(audit_df["flagged"].sum()) if len(audit_df) > 0 else 0
    logger.info(f"Scale-invariance audit: {n_flagged}/{len(audit_df)} features flagged (|corr| > {threshold})")
    return audit_df

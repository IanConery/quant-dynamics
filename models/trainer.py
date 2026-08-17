import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_percentage_error,
)
from tabulate import tabulate

from data.loader import temporal_cross_validation
from models.calibration import calibrate_platt

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner

from config.settings import MODEL_DIR, PREDICTION_WINDOWS, CLASSICAL_PARAMS, OPTUNA_CONFIG, DEEP_PARAMS, DEVICE
from data.loader import get_datasets
from models.classical import ClassicalModel, get_all_classical_models
from models.deep import DeepModel, get_all_deep_models
from models.ensemble import StackingEnsemble
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ─── Regression Metrics ─────────────────────────────────────────────────────


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # SMAPE instead of MAPE: robust to near-zero true values
    eps = 1e-8
    abs_err = np.abs(y_true - y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom < eps, eps, denom)
    smape = float(np.mean(abs_err / denom) * 100)
    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "SMAPE_%": round(smape, 4),
        "R2": round(r2, 4),
    }


# ─── Classification Metrics ──────────────────────────────────────────────────


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None, n_classes: int = 2) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    if n_classes == 2:
        try:
            auc = roc_auc_score(y_true, y_proba.ravel())
        except (ValueError, TypeError):
            auc = float("nan")
        bal_acc = float(np.mean([
            recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        ]))
        return {
            "Accuracy": round(acc, 4),
            "Bal_Acc": round(bal_acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1_macro, 4),
            "AUC-ROC": round(auc, 4),
 }
    else:
        # Multi-class: use macro-averaged metrics
        from sklearn.metrics import balanced_accuracy_score
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", zero_division=0)
        except (ValueError, TypeError):
            auc = float("nan")
        return {
            "Accuracy": round(acc, 4),
            "Bal_Acc": round(bal_acc, 4),
            "Precision_M": round(prec, 4),
            "Recall_M": round(rec, 4),
            "F1_Macro": round(f1_macro, 4),
            "F1_Wt": round(f1_weighted, 4),
            "AUC-ROC": round(auc, 4),
        }


# ─── Evaluate Single Model ───────────────────────────────────────────────────


def evaluate_model(
    model: ClassicalModel,
    X_test: np.ndarray,
    y_reg_test: np.ndarray,
    y_clf_test: np.ndarray,
    task: str,
    interval: str,
    window: str,
    n_classes: int = 2,
) -> Dict[str, float]:
    if task == "regression":
        y_pred = model.predict(X_test)
        return _regression_metrics(y_reg_test, y_pred)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        return _classification_metrics(y_clf_test, y_pred, y_proba, n_classes=n_classes)


# ─── Train All Classical Models ──────────────────────────────────────────────


def train_classical(
    interval: str,
    target_window: str,
    data: Dict,
    output_dir: str = None,
    n_classes: int = 2,
    suffix: str = "",
) -> Tuple[Dict[str, ClassicalModel], pd.DataFrame]:
    if output_dir is None:
        tw = target_window if "h" in str(target_window) else f"{target_window}h"
        output_dir = os.path.join(MODEL_DIR, interval, tw)
    os.makedirs(output_dir, exist_ok=True)

    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    logger.info(f"\n{'='*60}")
    logger.info(f"Training classical models: {interval} / {tw} (n_classes={n_classes})")
    logger.info(f"{'='*60}")

    results = []
    trained = {}

    for task in ["regression", "classification"]:
        short = "reg" if task == "regression" else "clf"
        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        y_train = data[f"y_{short}_train"]
        y_val = data[f"y_{short}_val"]

        models = get_all_classical_models(task, n_classes=n_classes if task == "classification" else 2)
        for model in models:
            model.fit(X_train, y_train, X_val, y_val)
            model.feature_names = list(data["feature_names"])
            metrics = evaluate_model(model, X_test, data["y_reg_test"], data["y_clf_test"],
                                      task, interval, tw, n_classes=n_classes if task == "classification" else 2)

            key = f"{model.model_type}_{short}{suffix}"
            trained[key] = model

            path = os.path.join(output_dir, f"{key}.pkl")
            model.save(path)

            row = {
                "Model": key,
                "Task": task,
                **metrics,
            }
            results.append(row)

    df = pd.DataFrame(results)
    logger.info("\n" + tabulate(df, headers="keys", tablefmt="simple", showindex=False))

    return trained, df


def train_deep(
    interval: str,
    target_window: str,
    data: Dict,
    output_dir: str = None,
    n_classes: int = 2,
    suffix: str = "",
) -> Tuple[Dict[str, DeepModel], pd.DataFrame]:
    if output_dir is None:
        tw = target_window if "h" in str(target_window) else f"{target_window}h"
        output_dir = os.path.join(MODEL_DIR, interval, tw)
    os.makedirs(output_dir, exist_ok=True)

    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    logger.info(f"\n{'='*60}")
    logger.info(f"Training deep models: {interval} / {tw} (n_classes={n_classes})")
    logger.info(f"{'='*60}")

    input_size = data["X_train_seq"].shape[2]
    results = []
    trained = {}

    X_train = data["X_train_seq"]
    X_val = data["X_val_seq"]
    X_test = data["X_test_seq"]

    # TB mode: use TB labels as ternary targets for TFT multi-task training
    if data.get("tb_used"):
        y_ternary_train = data.get("y_clf_train_seq")
        y_ternary_val = data.get("y_clf_val_seq")
    else:
        y_ternary_train = data.get("y_ternary_train_seq")
        y_ternary_val = data.get("y_ternary_val_seq")

    trained_tft = None

    for task in ["regression", "classification"]:
        short = "reg" if task == "regression" else "clf"
        y_train = data[f"y_{short}_train_seq"]
        y_val = data[f"y_{short}_val_seq"]

        models = get_all_deep_models(input_size, task, n_classes=n_classes if task == "classification" else 2)
        for model in models:
            is_tft = model.model_type == "tft"

            if is_tft:
                if trained_tft is not None:
                    model = trained_tft
                else:
                    # TB mode: regression-only (binary head doesn't fit 3-class targets)
                    if data.get("tb_used"):
                        model.fit(
                            X_train, y_train, X_val, y_val,
                            None, None,
                        )
                    else:
                        model.fit(
                            X_train, y_train, X_val, y_val,
                            y_ternary_train, y_ternary_val,
                        )
                    trained_tft = model

                reg_pred = model.predict_regression(X_test)
                clf_pred = model.predict_proba(X_test)

                n_reg = min(len(data["y_reg_test"]), len(reg_pred))
                reg_metrics = _regression_metrics(
                    data["y_reg_test"][:n_reg], reg_pred[:n_reg]
                )

                if model.n_classes > 2 and not data.get("tb_used"):
                    clf_labels = np.argmax(clf_pred, axis=1)
                    n_clf = min(len(data["y_ternary_test"]) if data.get("y_ternary_test") is not None else len(data["y_clf_test"]), len(clf_labels))
                    y_true_clf = (data["y_ternary_test"] if data.get("y_ternary_test") is not None else data["y_clf_test"])[:n_clf]
                    clf_metrics = _classification_metrics(
                        y_true_clf, clf_labels[:n_clf], clf_pred[:n_clf], n_classes=model.n_classes
                    )
                elif data.get("tb_used") and clf_pred.ndim > 1:
                    # TB mode: TFT trained regression-only, ternary head is random
                    # Use regression output as proxy for classification (skip if 1D)
                    clf_labels = np.argmax(clf_pred, axis=1)
                    n_clf = min(len(data["y_clf_test"]), len(clf_labels))
                    y_true_clf = data["y_clf_test"][:n_clf]
                    clf_metrics = _classification_metrics(
                        y_true_clf, clf_labels[:n_clf], clf_pred[:n_clf], n_classes=model.n_classes
                    )
                elif data.get("tb_used"):
                    # TFT trained regression-only, skip classification evaluation
                    clf_metrics = {"Bal_Acc": float("nan"), "Accuracy": float("nan"),
                                    "F1_Macro": float("nan"), "F1_Wt": float("nan"),
                                    "Precision_M": float("nan"), "Recall_M": float("nan"),
                                    "AUC-ROC": float("nan")}
                else:
                    clf_labels = (clf_pred[:, 0] if clf_pred.ndim > 1 else clf_pred) > 0.5
                    n_clf = min(len(data["y_clf_test"]), len(clf_labels))
                    clf_metrics = _classification_metrics(
                        data["y_clf_test"][:n_clf],
                        clf_labels[:n_clf].astype(int),
                        clf_pred[:n_clf] if clf_pred.ndim == 1 else clf_pred[:n_clf, 0],
                    )

                tft_reg_key = f"tft_reg{suffix}"
                tft_clf_key = f"tft_clf{suffix}"
                model.save(os.path.join(output_dir, f"{tft_reg_key}.pth"))
                orig_task = model.task
                model.task = "classification"
                model.save(os.path.join(output_dir, f"{tft_clf_key}.pth"))
                model.task = orig_task
                trained[tft_reg_key] = model
                trained[tft_clf_key] = model

                results.append({"Model": tft_reg_key, "Task": "regression", **reg_metrics})
                results.append({"Model": tft_clf_key, "Task": "classification", **clf_metrics})
            else:
                model.fit(X_train, y_train, X_val, y_val)

                pred = model.predict(X_test)
                n_overlap = min(len(data[f"y_{short}_test"]), len(pred))
                y_test_aligned = data[f"y_{short}_test"][:n_overlap]
                pred_aligned = pred[:n_overlap]

                if task == "regression":
                    metrics = _regression_metrics(y_test_aligned, pred_aligned)
                else:
                    if data.get("tb_used"):
                        # TB mode: 3-class targets, LSTM/iTransformer output binary prob
                        # Convert targets to binary: UP (2) vs NOT_UP (0,1)
                        y_test_bin = (y_test_aligned == 2).astype(int)
                        y_pred_clf = (pred_aligned > 0.5).astype(int)
                        metrics = _classification_metrics(y_test_bin, y_pred_clf, pred_aligned)
                    else:
                        y_pred_clf = (pred_aligned > 0.5).astype(int)
                        metrics = _classification_metrics(y_test_aligned, y_pred_clf, pred_aligned)

                key = f"{model.model_type}_{short}{suffix}"
                trained[key] = model
                model.save(os.path.join(output_dir, f"{key}.pth"))

                results.append({"Model": key, "Task": task, **metrics})

    df = pd.DataFrame(results)
    logger.info("\n" + tabulate(df, headers="keys", tablefmt="simple", showindex=False))

    return trained, df


# ─── Ensemble ──────────────────────────────────────────────────────────────


def train_ensemble(
    interval: str,
    target_window: str,
    data: Dict,
    output_dir: str = None,
    n_classes: int = 2,
    suffix: str = "",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if output_dir is None:
        tw = target_window if "h" in str(target_window) else f"{target_window}h"
        output_dir = os.path.join(MODEL_DIR, interval, tw)
    os.makedirs(output_dir, exist_ok=True)

    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    logger.info(f"\n{'='*60}")
    logger.info(f"Training ensemble: {interval} / {tw} (n_classes={n_classes})")
    logger.info(f"{'='*60}")

    X_val = data["X_val"]
    X_val_seq = data["X_val_seq"]
    X_test = data["X_test"]
    X_test_seq = data["X_test_seq"]

    results = []
    trained = {}

    for task in ["regression", "classification"]:
        short = "reg" if task == "regression" else "clf"
        y_val = data[f"y_{short}_val"]
        y_test_flat = data[f"y_{short}_test"]

        base_models = {}
        for key_file in sorted(os.listdir(output_dir)):
            if not (key_file.endswith(".pkl") or key_file.endswith(".pth")):
                continue
            fname_no_ext = key_file.replace(".pkl", "").replace(".pth", "")
            if fname_no_ext.endswith(f"_{short}{suffix}") and "ensemble" not in fname_no_ext:
                path = os.path.join(output_dir, key_file)
                try:
                    if key_file.endswith(".pkl"):
                        model = ClassicalModel.load(path)
                    else:
                        model = DeepModel.load(path)
                    base_models[key_file] = model
                except Exception as e:
                    logger.warning(f"  Could not load {key_file}: {e}")

        if len(base_models) < 2:
            logger.warning(f"  Only {len(base_models)} base models for {task}, skipping")
            continue

        meta_type = "ridge" if task == "regression" else "logistic_regression"
        ensemble = StackingEnsemble(task=task, meta_learner_type=meta_type)
        ensemble.fit(base_models, X_val, X_val_seq, y_val)

        n_overlap = min(len(y_test_flat), len(X_test_seq))
        pred = ensemble.predict(base_models, X_test, X_test_seq)[:n_overlap]
        y_aligned = y_test_flat[:n_overlap]

        if task == "regression":
            metrics = _regression_metrics(y_aligned, pred)
        else:
            if n_classes > 2:
                y_clf_pred = np.argmax(pred, axis=1) if pred.ndim > 1 else (pred > 0.5).astype(int)
                metrics = _classification_metrics(y_aligned, y_clf_pred, pred, n_classes=n_classes)
            else:
                y_clf_pred = (pred > 0.5).astype(int)
                metrics = _classification_metrics(y_aligned, y_clf_pred, pred)

        key = f"ensemble_{short}{suffix}"
        trained[key] = (ensemble, base_models)
        ensemble.save(os.path.join(output_dir, f"{key}.pkl"))

        results.append({"Model": key, "Task": task, "Base_Models": len(base_models), **metrics})

    df = pd.DataFrame(results)
    logger.info("\n" + tabulate(df, headers="keys", tablefmt="simple", showindex=False))
    return trained, df


# ─── Evaluate Trained Models ────────────────────────────────────────────────


def evaluate_all(
    interval: str,
    target_window: str,
    output_dir: str = None,
) -> pd.DataFrame:
    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, interval, tw)

    data = get_datasets(interval, target_window, feature_selection_enabled=True)

    results = []
    for fname in sorted(os.listdir(output_dir)):
        if not (fname.endswith(".pkl") or fname.endswith(".pth")):
            continue
        path = os.path.join(output_dir, fname)

        # Ensemble models
        if "ensemble" in fname:
            try:
                from models.ensemble import StackingEnsemble
                ensemble = StackingEnsemble.load(path)
                task = ensemble.task
                short = "reg" if task == "regression" else "clf"
                base_models = {}
                for bf in sorted(os.listdir(output_dir)):
                    if bf == fname:
                        continue
                    if short in bf and "ensemble" not in bf and (bf.endswith(".pkl") or bf.endswith(".pth")):
                        bpath = os.path.join(output_dir, bf)
                        try:
                            if bf.endswith(".pkl"):
                                base_models[bf] = ClassicalModel.load(bpath)
                            else:
                                base_models[bf] = DeepModel.load(bpath)
                        except Exception as e:
                            logger.warning(f"Could not load base model {bf}: {e}")
                n_overlap = min(len(data["X_test"]), len(data["X_test_seq"]))
                pred = ensemble.predict(base_models, data["X_test"][:n_overlap], data["X_test_seq"][:n_overlap])
                y_aligned = data[f"y_{short}_test"][:n_overlap]

                if task == "regression":
                    metrics = _regression_metrics(y_aligned, pred)
                else:
                    y_clf_pred = (pred > 0.5).astype(int)
                    metrics = _classification_metrics(y_aligned, y_clf_pred, pred)

                row = {"Model": fname.replace(".pkl", ""), "Task": task, **metrics}
                results.append(row)
            except Exception as e:
                logger.warning(f"Could not evaluate {fname}: {e}")
            continue

        # Classical models (.pkl)
        if fname.endswith(".pkl"):
            try:
                model = ClassicalModel.load(path)
                try:
                    if model.task == "regression":
                        metrics = evaluate_model(
                            model, data["X_test"], data["y_reg_test"], data["y_clf_test"],
                            "regression", interval, tw,
                        )
                    else:
                        metrics = evaluate_model(
                            model, data["X_test"], data["y_reg_test"], data["y_clf_test"],
                            "classification", interval, tw,
                        )
                    row = {"Model": fname.replace(".pkl", ""), "Task": model.task, **metrics}
                    results.append(row)
                except ValueError as e:
                    logger.warning(f"Feature mismatch in {fname}: {e} — skipping")
            except Exception as e:
                logger.warning(f"Could not evaluate {fname}: {e}")

        # Deep models (.pth)
        if fname.endswith(".pth"):
            try:
                model = DeepModel.load(path)
                task = model.task
                short = "reg" if task == "regression" else "clf"
                n_overlap = min(len(data[f"y_{short}_test"]), len(data["X_test_seq"]))
                pred = model.predict(data["X_test_seq"])[:n_overlap]
                y_aligned = data[f"y_{short}_test"][:n_overlap]

                if task == "regression":
                    metrics = _regression_metrics(y_aligned, pred)
                else:
                    y_clf_pred = (pred > 0.5).astype(int)
                    metrics = _classification_metrics(y_aligned, y_clf_pred, pred)

                row = {"Model": fname.replace(".pth", ""), "Task": task, **metrics}
                results.append(row)
            except Exception as e:
                logger.warning(f"Could not evaluate {fname}: {e}")

    df = pd.DataFrame(results)
    logger.info(f"\nEvaluation: {interval} / {tw}")
    logger.info(tabulate(df, headers="keys", tablefmt="simple", showindex=False))
    return df


# ─── Feature Importance Report ───────────────────────────────────────────────


def feature_importance_report(
    interval: str,
    target_window: str,
    output_dir: str = None,
    top_n: int = 20,
) -> pd.DataFrame:
    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, interval, tw)
    data = get_datasets(interval, target_window, feature_selection_enabled=True)
    feature_names = data["feature_names"]

    all_importances = {}
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(output_dir, fname)
        try:
            model = ClassicalModel.load(path)
            fi = model.get_feature_importances()
            if len(fi) == len(feature_names):
                all_importances[fname.replace(".pkl", "")] = fi
        except Exception as e:
            logger.warning(f"Could not load model {fname} for importance: {e}")

    if not all_importances:
        logger.warning("No models found for importance report")
        return pd.DataFrame()

    fi_df = pd.DataFrame(all_importances, index=feature_names)
    fi_df["mean_importance"] = fi_df.mean(axis=1)
    fi_df = fi_df.sort_values("mean_importance", ascending=False).head(top_n)

    logger.info(f"\nTop {top_n} features ({interval} / {tw}):")
    logger.info(tabulate(fi_df, headers="keys", tablefmt="simple"))
    return fi_df


# ─── Optuna Hyperparameter Tuning ────────────────────────────────────────────


def _walk_forward_splits(X: np.ndarray, y: np.ndarray, n_folds: int = 5, min_val_size: int = 100) -> list:
    """Create walk-forward (expanding window) train/val splits.

    Adapts fold size to actual data: uses min(n // (n_folds + 1), n // 2) as fold size
    and ensures minimum validation size.
    """
    n = len(X)
    fold_size = max(n // (n_folds + 1), min_val_size)
    fold_size = min(fold_size, n // 2)

    splits = []
    train_end = fold_size
    for i in range(n_folds):
        val_end = train_end + fold_size
        if val_end > n:
            val_end = n
            train_end = val_end - max(fold_size, min_val_size)
            if train_end <= 0:
                break
        if val_end - train_end < min_val_size:
            break
        splits.append((
            X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
        ))
        train_end = val_end
    return splits


def _build_classical_params(model_type: str, trial: optuna.Trial, task: str) -> dict:
    """Sample hyperparameters from search space for classical models."""
    if model_type == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
            "n_jobs": -1,
            "random_state": 42,
        }
        if task == "regression":
            params["objective"] = "reg:squarederror"
        else:
            params["objective"] = "binary:logistic"
        return params

    elif model_type == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 10, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
            "n_jobs": -1,
            "verbose": -1,
            "random_state": 42,
        }
        if task == "regression":
            params["objective"] = "regression"
        else:
            params["objective"] = "binary"
        return params

    elif model_type == "random_forest":
        feat_idx = trial.suggest_categorical("max_features", ["sqrt", "log2", "None"])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": feat_idx,
            "n_jobs": -1,
            "random_state": 42,
        }

    raise ValueError(f"Unknown model type: {model_type}")


def _build_classical_params_from(best_params: dict, model_type: str, task: str) -> dict:
    """Reconstruct classical params dict from Optuna best_params."""
    params = dict(best_params)
    params["n_jobs"] = -1
    params["random_state"] = 42
    if model_type == "xgboost":
        if task == "regression":
            params["objective"] = "reg:squarederror"
        else:
            params["objective"] = "binary:logistic"
    elif model_type == "lightgbm":
        params["verbose"] = -1
        params["n_jobs"] = -1
        if task == "regression":
            params["objective"] = "regression"
        else:
            params["objective"] = "binary"
    return params


def _build_deep_params(model_type: str, trial: optuna.Trial, task: str, input_size: int) -> dict:
    """Sample hyperparameters from search space for deep models."""
    shared = {
        "input_size": input_size,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "epochs": 200,
        "patience": 20,
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "seq_length": 60,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }
    if model_type == "lstm":
        shared.update({
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512]),
            "num_layers": trial.suggest_int("num_layers", 1, 6),
        })
    elif model_type == "itransformer":
        shared.update({
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256, 512]),
            "nhead": trial.suggest_categorical("nhead", [4, 8, 16]),
            "num_layers": trial.suggest_int("num_layers", 1, 6),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 512, 1024]),
        })
    elif model_type == "tft":
        shared.update({
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            "num_gru_layers": trial.suggest_int("num_gru_layers", 1, 4),
            "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
            "n_static_features": min(trial.suggest_int("n_static_features", 4, 12), input_size - 1),
            "mc_dropout_samples": 10,
        })
    else:
        raise ValueError(f"Unknown deep model type: {model_type}")
    return shared


def _objective_classical(
    trial: optuna.Trial,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    y_test: np.ndarray,
    model_type: str, task: str, n_folds: int,
) -> float:
    """Optuna objective for classical models with walk-forward CV."""
    splits = _walk_forward_splits(X_train, y_train, n_folds, min_val_size=100)
    if len(splits) < 2:
        raise optuna.TrialPruned()

    scores = []
    last_model = None
    last_train_X = None
    last_train_y = None
    for fold_i, (X_tr, y_tr, X_vl, y_vl) in enumerate(splits):
        try:
            params = _build_classical_params(model_type, trial, task)
            model = ClassicalModel(model_type, task, params)
            model.fit(X_tr, y_tr, X_vl, y_vl)
            pred = model.predict(X_vl)
            if task == "regression":
                score = r2_score(y_vl, pred)
            else:
                score = f1_score(y_vl, (pred > 0.5).astype(int), zero_division=0)
            scores.append(score)
            last_model = model
            last_train_X = X_tr
            last_train_y = y_tr

            if len(scores) >= 2 and trial.should_prune():
                intermediate = sum(scores) / len(scores)
                trial.report(intermediate, fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except Exception as e:
            logger.debug(f"  Fold {fold_i} error: {e}")
            continue

    if len(scores) == 0:
        raise optuna.TrialPruned()

    val_score = float(np.mean(scores))

    # Compute train score for gap detection — reuse last fold model
    # instead of training a new one (saves ~50% of Optuna training time)
    try:
        if last_model is not None and last_train_X is not None:
            train_pred = last_model.predict(last_train_X)
            if task == "regression":
                train_score = r2_score(last_train_y, train_pred)
            else:
                train_score = f1_score(last_train_y, (train_pred > 0.5).astype(int), zero_division=0)
        else:
            train_score = val_score
    except Exception:
        train_score = val_score

    train_val_gap = train_score - val_score

    # Prune if gap > 0.15 (model is overfitting)
    if train_val_gap > 0.15:
        trial.set_user_attr("train_val_gap", round(train_val_gap, 4))
        raise optuna.TrialPruned()

    trial.set_user_attr("train_score", round(train_score, 4))
    trial.set_user_attr("val_score", round(val_score, 4))
    trial.set_user_attr("train_val_gap", round(train_val_gap, 4))

    return val_score


def _objective_deep(
    trial: optuna.Trial,
    X_train_seq: np.ndarray, y_train: np.ndarray,
    X_val_seq: np.ndarray, y_val: np.ndarray,
    model_type: str, task: str, input_size: int, n_folds: int,
) -> float:
    """Optuna objective for deep models with walk-forward CV."""
    splits = _walk_forward_splits(X_train_seq, y_train, n_folds, min_val_size=120)
    if len(splits) < 2:
        raise optuna.TrialPruned()

    scores = []
    last_model = None
    last_train_X = None
    last_train_y = None
    for fold_i, (X_tr, y_tr, X_vl, y_vl) in enumerate(splits):
        try:
            params = _build_deep_params(model_type, trial, task, input_size)
            model = DeepModel(model_type, input_size, task, params)
            model.fit(X_tr, y_tr, X_vl, y_vl)
            pred = model.predict(X_vl)
            if len(pred) == 0 or len(y_vl) == 0:
                continue
            if task == "regression":
                score = r2_score(y_vl, pred)
            else:
                score = f1_score(y_vl, (pred > 0.5).astype(int), zero_division=0)
            scores.append(score)
            last_model = model
            last_train_X = X_tr
            last_train_y = y_tr

            if len(scores) >= 2 and trial.should_prune():
                intermediate = sum(scores) / len(scores)
                trial.report(intermediate, fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except Exception as e:
            logger.debug(f"  Fold {fold_i} error: {e}")
            continue

    if len(scores) == 0:
        raise optuna.TrialPruned()

    val_score = float(np.mean(scores))

    # Compute train score for gap detection — reuse last fold model
    # instead of training a new one (saves GPU memory + ~50% training time)
    try:
        if last_model is not None and last_train_X is not None:
            train_pred = last_model.predict(last_train_X)
            if len(train_pred) > 0 and len(last_train_y) > 0:
                if task == "regression":
                    train_score = r2_score(last_train_y, train_pred)
                else:
                    train_score = f1_score(last_train_y, (train_pred > 0.5).astype(int), zero_division=0)
            else:
                train_score = val_score
        else:
            train_score = val_score
    except Exception:
        train_score = val_score

    train_val_gap = train_score - val_score

    if train_val_gap > 0.15:
        trial.set_user_attr("train_val_gap", round(train_val_gap, 4))
        raise optuna.TrialPruned()

    trial.set_user_attr("train_score", round(train_score, 4))
    trial.set_user_attr("val_score", round(val_score, 4))
    trial.set_user_attr("train_val_gap", round(train_val_gap, 4))

    return val_score


def optuna_tune(
    interval: str,
    target_window: str,
    model_type: str = "lightgbm",
    task: str = "regression",
    data: Dict = None,
    n_folds: int = 5,
    n_trials: int = None,
    timeout_minutes: int = None,
    output_dir: str = None,
) -> Dict:
    """
    Run Optuna hyperparameter search with walk-forward validation.

    Args:
        interval: Data interval (15m, 1h, 1d)
        target_window: Prediction window (e.g. 24h)
        model_type: One of xgboost, lightgbm, random_forest, lstm, itransformer
        task: "regression" or "classification"
        data: Dict from get_datasets(). If None, will be loaded automatically.
        n_folds: Number of internal walk-forward folds per trial
        n_trials: Max number of trials (default from OPTUNA_CONFIG)
        timeout_minutes: Timeout in minutes (default from OPTUNA_CONFIG)
        output_dir: Where to save results (default: MODEL_DIR/{interval}/{window}/)

    Returns:
        Dict with best_params, best_score.
        Saves optuna_results.json to output_dir.
    """
    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, interval, tw)
    os.makedirs(output_dir, exist_ok=True)

    if n_trials is None:
        n_trials = OPTUNA_CONFIG.get("n_trials", 100)
    if timeout_minutes is None:
        timeout_minutes = OPTUNA_CONFIG.get("timeout_minutes", 120)

    if data is None:
        data = get_datasets(interval, tw)

    is_deep = model_type in ("lstm", "itransformer", "tft")
    study_name = f"{OPTUNA_CONFIG.get('study_name_prefix', 'xrp_')}{model_type}_{task}_{interval}_{tw}"

    sampler_name = OPTUNA_CONFIG.get("sampler", "tpe")
    sampler = TPESampler() if sampler_name == "tpe" else RandomSampler()

    short = "reg" if task == "regression" else "clf"

    if is_deep:
        input_size = data["X_train_seq"].shape[2]
        objective = lambda trial: _objective_deep(
            trial,
            data["X_train_seq"], data[f"y_{short}_train_seq"],
            data["X_val_seq"], data[f"y_{short}_val_seq"],
            model_type, task, input_size, n_folds,
        )
    else:
        objective = lambda trial: _objective_classical(
            trial,
            data["X_train"], data[f"y_{short}_train"],
            data["X_val"], data[f"y_{short}_val"],
            data[f"y_{short}_test"],
            model_type, task, n_folds,
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"Optuna tuning: {model_type}/{task} | {interval}/{tw}")
    logger.info(f"  Trials: {n_trials}, Timeout: {timeout_minutes}min, Folds: {n_folds}")
    logger.info(f"{'='*60}")

    storage = OPTUNA_CONFIG.get("storage", None)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60, show_progress_bar=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) == 0:
        logger.warning(f"No trials completed ({len(study.trials)} all pruned). Falling back to defaults.")
        if is_deep:
            best_params = {k: v for k, v in DEEP_PARAMS.get(model_type, {}).items()}
            best_score = 0.0
        else:
            best_params = CLASSICAL_PARAMS.get(model_type, {}).get(task, {})
            best_score = 0.0
    else:
        best_params = study.best_params
        best_score = study.best_value

    logger.info(f"\nBest trial: score={best_score:.6f}")
    logger.info(f"Best params: {json.dumps(best_params, indent=2)}")

    # Train final model with best params on full data
    logger.info(f"\nTraining final model with best params...")
    if is_deep:
        input_size = data["X_train_seq"].shape[2]
        final_params = {
            "input_size": input_size,
            "learning_rate": best_params["learning_rate"],
            "batch_size": best_params["batch_size"],
            "epochs": 200,
            "patience": 20,
            "weight_decay": best_params.get("weight_decay", 1e-5),
            "seq_length": 60,
            "dropout": best_params["dropout"],
        }
        if model_type == "lstm":
            final_params["hidden_size"] = best_params["hidden_size"]
            final_params["num_layers"] = best_params["num_layers"]
        elif model_type == "tft":
            final_params["hidden_size"] = best_params["hidden_size"]
            final_params["num_gru_layers"] = best_params["num_gru_layers"]
            final_params["nhead"] = best_params["nhead"]
            final_params["n_static_features"] = best_params.get("n_static_features", 8)
            final_params["mc_dropout_samples"] = 10
        else:
            final_params["d_model"] = best_params["d_model"]
            final_params["nhead"] = best_params["nhead"]
            final_params["num_layers"] = best_params["num_layers"]
            final_params["dim_feedforward"] = best_params["dim_feedforward"]

        y_ternary_train = data.get("y_ternary_train_seq")
        y_ternary_val = data.get("y_ternary_val_seq")

        final_model = DeepModel(model_type, input_size, task, final_params)
        if model_type == "tft":
            final_model.fit(
                data["X_train_seq"], data[f"y_{short}_train_seq"],
                data["X_val_seq"], data[f"y_{short}_val_seq"],
                y_ternary_train, y_ternary_val,
            )
            final_model.save(os.path.join(output_dir, f"{model_type}_{short}_tuned.pth"))
            final_model.save(os.path.join(output_dir, f"{model_type}_clf_tuned.pth"))
        else:
            final_model.fit(
                data["X_train_seq"], data[f"y_{short}_train_seq"],
                data["X_val_seq"], data[f"y_{short}_val_seq"],
            )
            final_model.save(os.path.join(output_dir, f"{model_type}_{short}_tuned.pth"))
    else:
        final_params = _build_classical_params_from(best_params, model_type, task)
        final_model = ClassicalModel(model_type, task, final_params)
        final_model.fit(
            data["X_train"], data[f"y_{short}_train"],
            data["X_val"], data[f"y_{short}_val"],
        )
        final_model.save(os.path.join(output_dir, f"{model_type}_{short}_tuned.pkl"))

        # Evaluate final model on test
        if task == "regression":
            test_pred = final_model.predict(data["X_test"])
            test_metrics = _regression_metrics(data[f"y_{short}_test"], test_pred)
        else:
            test_pred = final_model.predict(data["X_test"])
            test_metrics = _classification_metrics(
                data[f"y_{short}_test"], (test_pred > 0.5).astype(int), test_pred
            )
        logger.info(f"Test metrics: {test_metrics}")

    # Save results
    results = {
        "study_name": study_name,
        "model_type": model_type,
        "task": task,
        "interval": interval,
        "target_window": tw,
        "n_trials": len(study.trials),
        "best_score": best_score,
        "best_params": best_params,
        "direction": "maximize",
        "n_folds": n_folds,
    }

    # Gap statistics from completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        gaps = [t.user_attrs.get("train_val_gap", 0.0) for t in completed_trials]
        results["gap_stats"] = {
            "mean_gap": round(float(np.mean(gaps)), 4),
            "max_gap": round(float(np.max(gaps)), 4),
            "median_gap": round(float(np.median(gaps)), 4),
            "n_pruned_by_gap": len([t for t in study.trials
                                     if t.state == optuna.trial.TrialState.PRUNED
                                     and t.user_attrs.get("train_val_gap", 0) > 0.15]),
        }

    results_path = os.path.join(output_dir, f"optuna_{model_type}_{short}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Optuna results saved to {results_path}")

    return results


def temporal_cv_evaluate(
    interval: str,
    target_window: str,
    model_type: str = "lightgbm",
    task: str = "regression",
    n_folds: int = 5,
    gap: int = 0,
    min_train_samples: int = 500,
    min_val_samples: int = 100,
    calibrate: bool = False,
) -> pd.DataFrame:
    """Evaluate a model using temporal (walk-forward) cross-validation.

    Trains the model on each fold's training data, evaluates on the validation fold.
    Optionally calibrates probabilities using Platt scaling.

    Args:
        interval: Data interval (15m, 1h, 1d)
        target_window: Prediction window (e.g., 24h)
        model_type: Model type (lightgbm, xgboost, random_forest)
        task: "regression" or "classification"
        n_folds: Number of walk-forward folds
        gap: Samples to skip between train and val (prevents leakage)
        min_train_samples: Minimum training samples per fold
        min_val_samples: Minimum validation samples per fold
        calibrate: Whether to calibrate classification probabilities

    Returns:
        DataFrame with per-fold metrics.
    """
    tw = target_window if "h" in str(target_window) else f"{target_window}h"

    from data.loader import load_processed
    df = load_processed(interval)
    folds = temporal_cross_validation(
        df, tw, n_folds=n_folds, gap=gap,
        min_train_samples=min_train_samples, min_val_samples=min_val_samples,
    )

    short = "reg" if task == "regression" else "clf"
    results = []

    for fold_data in folds:
        params = dict(CLASSICAL_PARAMS[model_type][task])
        model = ClassicalModel(model_type, task, params)
        model.fit(
            fold_data["X_train"], fold_data[f"y_{short}_train"],
            fold_data["X_val"], fold_data[f"y_{short}_val"],
        )

        pred = model.predict(fold_data["X_test"])

        if task == "regression":
            metrics = _regression_metrics(fold_data["y_reg_test"], pred)
        else:
            proba = model.predict_proba(fold_data["X_test"])
            if calibrate:
                proba = calibrate_platt(proba, fold_data[f"y_{short}_val"], method="sigmoid")

            y_pred = (proba > 0.5).astype(int)
            metrics = _classification_metrics(
                fold_data["y_clf_test"], y_pred, proba
            )

        row = {
            "fold": fold_data["fold"],
            "train_size": fold_data["train_size"],
            "val_size": fold_data["val_size"],
            **metrics,
        }
        results.append(row)
        logger.info(f"  Fold {fold_data['fold']}: {row}")

    results_df = pd.DataFrame(results)
    logger.info(f"\nTemporal CV ({model_type}/{task}, {n_folds} folds, gap={gap}):")
    if task == "regression":
        metric_cols = [c for c in ['R2', 'MAE', 'SMAPE_%'] if c in results_df.columns]
    else:
        metric_cols = [c for c in ['Bal_Acc', 'F1', 'AUC-ROC', 'Accuracy'] if c in results_df.columns]
    logger.info(f"  Mean metrics: {results_df[metric_cols].mean().to_dict()}")
    logger.info(f"  Std  metrics: {results_df[metric_cols].std().to_dict()}")
    logger.info("\n" + tabulate(results_df, headers="keys", tablefmt="simple", showindex=False))
    return results_df


def calibrate_model(
    interval: str,
    target_window: str,
    model_type: str = "lightgbm",
    method: str = "sigmoid",
) -> None:
    """Calibrate a model's probabilities on validation data.

    Loads the trained model, computes raw probabilities on validation set,
    trains a calibrator, then applies it to test set predictions.
    Compares calibration quality (ECE) before and after.

    Args:
        interval: Data interval
        target_window: Prediction window
        model_type: Model type (lightgbm, xgboost, random_forest)
        method: 'sigmoid' (Platt) or 'isotonic'
    """
    from models.calibration import ProbabilityCalibrator
    from data.loader import load_processed

    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    data = get_datasets(interval, tw)

    model_dir = os.path.join(MODEL_DIR, interval, tw)
    proba_val = None
    proba_test = None

    mpath_pkl = os.path.join(model_dir, f"{model_type}_clf.pkl")
    mpath_tuned = os.path.join(model_dir, f"{model_type}_clf_tuned.pkl")
    mpath = mpath_tuned if os.path.exists(mpath_tuned) else mpath_pkl

    if os.path.exists(mpath):
        try:
            model = ClassicalModel.load(mpath)
            proba_val = model.predict_proba(data["X_val"])
            proba_test = model.predict_proba(data["X_test"])
        except ValueError as e:
            logger.warning(f"Feature mismatch loading {mpath}: {e}")
            logger.info(f"Retraining {model_type} on current data for calibration...")
            params = dict(CLASSICAL_PARAMS[model_type]["classification"])
            model = ClassicalModel(model_type, "classification", params)
            model.fit(data["X_train"], data["y_clf_train"], data["X_val"], data["y_clf_val"])
            proba_val = model.predict_proba(data["X_val"])
            proba_test = model.predict_proba(data["X_test"])
    else:
        logger.error(f"No model found for calibration: {model_type} at {model_dir}")
        return

    if proba_val is None:
        logger.error(f"Model produced no predictions for calibration")
        return

    y_val = data["y_clf_val"]
    y_test = data["y_clf_test"]

    logger.info(f"\nCalibration: {model_type}/{tw} | method={method}")
    from models.calibration import _ece_binary
    ece_before = _ece_binary(proba_val, y_val)
    logger.info(f"  ECE before calibration: {ece_before:.4f}")

    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(proba_val, y_val)

    proba_calibrated = calibrator.calibrate(proba_test)
    ece_after = _ece_binary(proba_calibrated, y_test)
    logger.info(f"  ECE after calibration:  {ece_after:.4f}")
    logger.info(f"  Improvement: {ece_before - ece_after:+.4f}")

    y_pred_raw = (proba_test > 0.5).astype(int)
    y_pred_cal = (proba_calibrated > 0.5).astype(int)
    metrics_raw = _classification_metrics(y_test, y_pred_raw, proba_test)
    metrics_cal = _classification_metrics(y_test, y_pred_cal, proba_calibrated)

    logger.info(f"\nMetrics before calibration: {metrics_raw}")
    logger.info(f"Metrics after calibration:  {metrics_cal}")

    cal_path = os.path.join(model_dir, f"{model_type}_calibrator_{method}.pkl")
    import joblib
    joblib.dump(calibrator, cal_path)
    logger.info(f"Calibrator saved to {cal_path}")

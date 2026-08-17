import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engine import compare_strategies, run_backtest, walk_forward_backtest  # noqa: E402
from bot_tracker.cli import register_bot_parser, run_bot_command  # noqa: E402
from data.derivatives_fetcher import fetch_all_derivatives, load_derivatives_data  # noqa: E402
from data.external_fetcher import fetch_all_external  # noqa: E402
from data.fetcher import (  # noqa: E402
    load_cross_asset_data,
    load_external_data,
    load_raw_data,
    refresh_all,
    run_full_fetch,
)
from data.loader import chronological_split, get_datasets, get_tb_datasets, load_processed  # noqa: E402
from data.processor import (  # noqa: E402
    create_targets,
    create_targets_triple_barrier,
    meta_label_filter,
    process_raw_data,
    save_processed_data,
)
from models.classical import ClassicalModel  # noqa: E402
from models.deep import DeepModel  # noqa: E402
from models.garch import GARCHEnsemble  # noqa: E402
from models.regime_trainer import RegimeAwareTrainer  # noqa: E402
from models.state_space import KalmanFilter  # noqa: E402
from models.trainer import (  # noqa: E402
    calibrate_model,
    evaluate_all,
    feature_importance_report,
    optuna_tune,
    temporal_cv_evaluate,
    train_classical,
    train_deep,
    train_ensemble,
)
from streaming.drift_monitor import DriftMonitor  # noqa: E402
from streaming.engine import run_stream  # noqa: E402
from streaming.predictor import StreamPredictor  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)


def run_data_fetch(args):
    intervals = args.intervals if args.intervals else None
    force = getattr(args, "force", False)
    refresh = getattr(args, "refresh", False)
    fetch_binance = getattr(args, "binance", False)
    fetch_derivatives = getattr(args, "derivatives", False)
    fetch_external = getattr(args, "external", False)

    if force:
        import glob as glib
        raw_dir = "artifacts/raw_data"
        for f in glib.glob(f"{raw_dir}/*.parquet"):
            os.remove(f)
        for f in glib.glob(f"{raw_dir}/external/*.parquet"):
            os.remove(f)
        for f in glib.glob(f"{raw_dir}/derivatives/*.parquet"):
            os.remove(f)
        logger.info("Cleared existing raw data")
        run_full_fetch(intervals, fetch_binance=fetch_binance)
        if fetch_derivatives:
            fetch_all_derivatives()
        if fetch_external:
            fetch_all_external()
    elif fetch_binance:
        run_full_fetch(intervals, fetch_binance=True)
        if fetch_derivatives:
            fetch_all_derivatives()
        if fetch_external:
            fetch_all_external()
    elif refresh:
        refresh_all(intervals)
        if fetch_derivatives:
            fetch_all_derivatives()
        if fetch_external:
            fetch_all_external()
    elif fetch_derivatives:
        fetch_all_derivatives()
    elif fetch_external:
        fetch_all_external()
    else:
        run_full_fetch(intervals)


def run_data_process(args):
    from config.settings import INTERVALS
    intervals = args.intervals if args.intervals else INTERVALS

    logger.info("=" * 60)
    logger.info("Starting data processing")
    logger.info("=" * 60)

    # Load external data (FNG)
    try:
        fng_df = load_external_data("fear_greed_index")
    except FileNotFoundError:
        fng_df = None
        logger.warning("No FNG data found, skipping external features")

    # Load derivatives data
    funding_df = None
    oi_df = None
    liq_df = None
    try:
        funding_df = load_derivatives_data("funding_rates")
    except FileNotFoundError:
        logger.warning("No funding rate data found, skipping")
    try:
        oi_df = load_derivatives_data("open_interest")
    except FileNotFoundError:
        logger.warning("No open interest data found, skipping")
    try:
        liq_df = load_derivatives_data("liquidations")
    except FileNotFoundError:
        logger.warning("No liquidation data found, skipping")

    # Load on-chain data
    active_addr_df = None
    exchange_flow_df = None
    try:
        active_addr_df = load_external_data("onchain_active_addresses")
    except FileNotFoundError:
        logger.warning("No active address data found, skipping")
    try:
        exchange_flow_df = load_external_data("onchain_exchange_net_flow")
    except FileNotFoundError:
        logger.warning("No exchange flow data found, skipping")

    # Load macro data
    vix_df = None
    dxy_df = None
    us10y_df = None
    try:
        vix_df = load_external_data("macro_vix")
    except FileNotFoundError:
        logger.warning("No VIX data found, skipping")
    try:
        dxy_df = load_external_data("macro_dxy")
    except FileNotFoundError:
        logger.warning("No DXY data found, skipping")
    try:
        us10y_df = load_external_data("macro_us10y")
    except FileNotFoundError:
        logger.warning("No US10Y data found, skipping")

    # Load order book data
    ob_df = None
    try:
        ob_df = load_external_data("order_book_snapshot")
    except FileNotFoundError:
        logger.warning("No order book data found, skipping")

    for interval in intervals:
        logger.info(f"\n--- Processing {interval} ---")
        xrp_df = load_raw_data(interval)

        try:
            btc_df = load_cross_asset_data("BTC/USDT", interval)
        except FileNotFoundError:
            btc_df = None
            logger.warning(f"No BTC data for {interval}")

        try:
            eth_df = load_cross_asset_data("ETH/USDT", interval)
        except FileNotFoundError:
            eth_df = None
            logger.warning(f"No ETH data for {interval}")

        df = process_raw_data(
            xrp_df, btc_df, eth_df, fng_df,
            funding_df, oi_df, liq_df,
            active_addr_df, exchange_flow_df,
            vix_df, dxy_df, us10y_df, ob_df,
        )
        df = create_targets(df, interval)

        df = df.dropna()
        logger.info(f"After dropna: {len(df)} rows, {len(df.columns)} columns")
        save_processed_data(df, interval)

    logger.info("\n" + "=" * 60)
    logger.info("Data processing complete!")
    logger.info("=" * 60)


def run_train(args):
    from config.settings import MODEL_DIR
    interval = args.interval
    window = args.window

    tw = window if "h" in str(window) else f"{window}h"
    use_tb = getattr(args, "triple_barrier", False)
    use_meta = getattr(args, "meta_filter", False)

    if use_tb:
        data = get_tb_datasets(interval, tw, apply_meta_filter=use_meta)
        n_classes = 3
        suffix = "_tb"
    else:
        data = get_datasets(interval, tw)
        n_classes = 2
        suffix = ""

    if use_tb:
        logger.info(f"Training with triple barrier targets (3-class: DOWN/SIDEWAYS/UP)" +
                     (" + meta-label filter" if use_meta else ""))

    output_dir = os.path.join(MODEL_DIR, interval, tw)
    os.makedirs(output_dir, exist_ok=True)

    if args.tune:
        model_types = ["lightgbm", "xgboost", "random_forest"]
        if args.model == "classical":
            model_types = ["lightgbm", "xgboost", "random_forest"]
        elif args.model == "deep":
            model_types = ["lstm", "itransformer", "tft"]
        elif args.model == "ensemble":
            model_types = ["lightgbm"]

        for mt in model_types:
            if mt == "tft":
                optuna_tune(interval, tw, model_type=mt, task="regression", data=data, n_trials=50, timeout_minutes=60)
            else:
                for task in ["regression", "classification"]:
                    optuna_tune(interval, tw, model_type=mt, task=task, data=data, n_trials=50, timeout_minutes=60)
    else:
        if args.model in ("all", "classical"):
            trained, results = train_classical(interval, tw, data, n_classes=n_classes, suffix=suffix)

        if args.model in ("all", "deep"):
            trained, results = train_deep(interval, tw, data, n_classes=n_classes, suffix=suffix)

        if args.model in ("all", "ensemble"):
            trained, results = train_ensemble(interval, tw, data, n_classes=n_classes, suffix=suffix)

    if args.evaluate:
        evaluate_all(interval, tw)
        feature_importance_report(interval, tw)


def run_evaluate(args):
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"
    evaluate_all(interval, tw)
    feature_importance_report(interval, tw)


def run_temporal_cv(args):
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"
    temporal_cv_evaluate(
        interval=interval, target_window=tw,
        model_type=args.model, task=args.task,
        n_folds=args.folds, gap=args.gap,
        calibrate=args.calibrate,
    )


def run_calibrate(args):
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"
    calibrate_model(
        interval=interval, target_window=tw,
        model_type=args.model, method=args.method,
    )


def _load_model_for_predict(model_name: str, model_dir: str, data: dict):
    """Load a model and return (model, probs, reg_preds) tuple."""
    from models.ensemble import StackingEnsemble
    import os

    if model_name == "ensemble":
        ens_path = os.path.join(model_dir, "ensemble_clf.pkl")
        if os.path.exists(ens_path):
            ensemble = StackingEnsemble.load(ens_path)
            base_models = {}
            for fname in sorted(os.listdir(model_dir)):
                if fname.endswith(".pkl") or fname.endswith(".pth"):
                    key = fname.replace(".pkl", "").replace(".pth", "")
                    if "clf" in key and key != "ensemble_clf":
                        fpath = os.path.join(model_dir, fname)
                        try:
                            if fname.endswith(".pkl"):
                                base_models[fname] = ClassicalModel.load(fpath)
                            else:
                                base_models[fname] = DeepModel.load(fpath)
                        except Exception as e:
                            logger.warning(f"Could not load {fname}: {e}")
            n = min(len(data["X_test"]), len(data["X_test_seq"]))
            probs = ensemble.predict(base_models, data["X_test"][:n], data["X_test_seq"][:n])

            # Also get regression predictions
            ens_reg_path = os.path.join(model_dir, "ensemble_reg.pkl")
            if os.path.exists(ens_reg_path):
                ens_reg = StackingEnsemble.load(ens_reg_path)
                base_reg = {}
                for fname in sorted(os.listdir(model_dir)):
                    if fname.endswith(".pkl") or fname.endswith(".pth"):
                        key = fname.replace(".pkl", "").replace(".pth", "")
                        if "reg" in key and key != "ensemble_reg":
                            fpath = os.path.join(model_dir, fname)
                            try:
                                if fname.endswith(".pkl"):
                                    base_reg[fname] = ClassicalModel.load(fpath)
                                else:
                                    base_reg[fname] = DeepModel.load(fpath)
                            except Exception as e:
                                logger.warning(f"Could not load {fname}: {e}")
                reg_preds = ens_reg.predict(base_reg, data["X_test"][:n], data["X_test_seq"][:n])
            else:
                reg_preds = None
            return ("ensemble", probs, reg_preds)
        logger.warning("No ensemble model found, falling back to lightgbm")

    # Individual model
    for ext in [".pkl", ".pth"]:
        mpath = os.path.join(model_dir, f"{model_name}_clf{ext}")
        if os.path.exists(mpath):
            if ext == ".pkl":
                model = ClassicalModel.load(mpath)
                probs = model.predict_proba(data["X_test"])
            else:
                model = DeepModel.load(mpath)
                probs = model.predict(data["X_test_seq"])

            # Also load regression model
            reg_path = os.path.join(model_dir, f"{model_name}_reg{ext}")
            reg_preds = None
            if os.path.exists(reg_path):
                if ext == ".pkl":
                    reg_model = ClassicalModel.load(reg_path)
                    reg_preds = reg_model.predict(data["X_test"])
                else:
                    reg_model = DeepModel.load(reg_path)
                    reg_preds = reg_model.predict(data["X_test_seq"])
            return (model_name, probs, reg_preds)

    logger.error(f"No model found for {model_name} in {model_dir}")
    return (None, None, None)


def run_predict(args):
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"
    model_name = args.model
    latest_n = args.latest

    model_dir = os.path.join("artifacts/models", interval, tw)
    data = get_datasets(interval, tw)

    _, probs, reg_preds = _load_model_for_predict(model_name, model_dir, data)
    if probs is None:
        return

    # Use test set timestamps (they represent the most recent period)
    df = load_processed(interval)
    _, _, test_df = chronological_split(df)

    n = min(latest_n, len(probs), len(test_df))
    latest = test_df.tail(n).reset_index(drop=True)

    # Align predictions to latest rows
    all_test_probs = probs[-len(test_df):] if len(probs) >= len(test_df) else probs
    offset = len(test_df) - len(all_test_probs)
    selected_probs = all_test_probs[offset + len(test_df) - n:]
    if len(selected_probs) == 0:
        selected_probs = all_test_probs[-n:]

    if reg_preds is not None:
        all_reg = reg_preds[-len(test_df):] if len(reg_preds) >= len(test_df) else reg_preds
        r_offset = len(test_df) - len(all_reg)
        selected_reg = all_reg[r_offset + len(test_df) - n:]
        if len(selected_reg) == 0:
            selected_reg = all_reg[-n:]
    else:
        selected_reg = None

    from tabulate import tabulate
    rows = []
    for i in range(n):
        ts = str(latest.iloc[i].get("timestamp", "?"))
        price = latest.iloc[i].get("close", 0)
        p = float(selected_probs[i]) if i < len(selected_probs) else 0.5
        direction = "UP" if p >= 0.5 else "DOWN"
        conf = max(p, 1 - p)
        if i < len(selected_reg) and selected_reg is not None:
            ret = float(selected_reg[i])
            pred_price = price * (1 + ret)
            rows.append([ts, f"{price:.4f}", f"{p:.4f}", direction,
                         f"{conf:.4f}", f"{ret:+.4f}", f"{pred_price:.4f}"])
        else:
            rows.append([ts, f"{price:.4f}", f"{p:.4f}", direction, f"{conf:.4f}", "N/A", "N/A"])

    headers = ["Timestamp", "Close", "P(UP)", "Direction", "Confidence", "Pred_Return", "Pred_Close"]
    logger.info(f"\n{'='*80}")
    logger.info(f"Predictions ({interval}/{tw} | model={model_name} | latest {n})")
    logger.info(f"{'='*80}")
    logger.info(tabulate(rows, headers=headers, tablefmt="simple"))


def run_backtest_cmd(args):
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"
    model_name = args.model
    strategy = args.strategy
    zero_commission = args.zero_commission
    wf_method = args.wf_method
    min_confidence = args.min_confidence
    min_holding_bars = args.min_holding_bars
    regime_aware = getattr(args, "regime_aware", False)
    bayesian = getattr(args, "bayesian", False)

    if args.walk_forward:
        logger.info("=" * 60)
        logger.info(f"Walk-forward backtest | method={wf_method} | zero_commission={zero_commission}")
        logger.info(f"  min_confidence={min_confidence} | min_holding_bars={min_holding_bars}")
        logger.info("=" * 60)
        trades, metrics = walk_forward_backtest(
            interval=interval, target_window=tw, model_type=model_name,
            strategy=strategy if strategy != "all" else "signal_based",
            zero_commission=zero_commission,
            wf_method=wf_method,
            min_confidence=min_confidence,
            min_holding_bars=min_holding_bars,
            regime_aware=regime_aware,
            bayesian=bayesian,
        )
        import json
        logger.info(f"\nAggregate metrics:")
        logger.info(json.dumps(metrics, indent=2, default=str))
    elif strategy == "all":
        logger.info("=" * 60)
        logger.info("Strategy comparison")
        logger.info("=" * 60)
        df = compare_strategies(interval=interval, target_window=tw, model_type=model_name)
    else:
        logger.info("=" * 60)
        logger.info(f"Backtest: {interval}/{tw} | model={model_name} | strategy={strategy}")
        logger.info("=" * 60)
        data = get_datasets(interval, tw)

        model_dir = os.path.join("artifacts/models", interval, tw)
        model = None
        if model_name == "ensemble":
            from models.ensemble import StackingEnsemble
            ens_path = os.path.join(model_dir, "ensemble_clf.pkl")
            if os.path.exists(ens_path):
                model = StackingEnsemble.load(ens_path)
                base_models = {}
                for fname in os.listdir(model_dir):
                    if fname.endswith(".pkl") or fname.endswith(".pth"):
                        key = fname.replace(".pkl", "").replace(".pth", "")
                        if "clf" in key and key != "ensemble_clf":
                            fpath = os.path.join(model_dir, fname)
                            try:
                                if fname.endswith(".pkl"):
                                    bm = ClassicalModel.load(fpath)
                                    try:
                                        _ = bm.predict_proba(data["X_test"][:1])
                                    except ValueError as e:
                                        logger.warning(f"Retraining {key}: {e}")
                                        from config.settings import CLASSICAL_PARAMS
                                        mt = key.split("_")[0]
                                        params = dict(CLASSICAL_PARAMS[mt]["classification"])
                                        bm = ClassicalModel(mt, "classification", params)
                                        bm.fit(data["X_train"], data["y_clf_train"], data["X_val"], data["y_clf_val"])
                                        bm.save(fpath)
                                    base_models[fname] = bm
                                else:
                                    base_models[fname] = DeepModel.load(fpath)
                            except Exception as e:
                                logger.warning(f"Could not load {fname}: {e}")
                model = (model, base_models)

        if model is None:
            mpath = os.path.join(model_dir, f"{model_name}_clf.pkl")
            if not os.path.exists(mpath):
                mpath = os.path.join(model_dir, f"{model_name}_clf.pth")
            if os.path.exists(mpath):
                try:
                    if mpath.endswith(".pkl"):
                        model = ClassicalModel.load(mpath)
                        # Check feature compatibility
                        try:
                            _ = model.predict_proba(data["X_test"][:1])
                        except ValueError as e:
                            logger.warning(f"Feature mismatch: {e}")
                            logger.info(f"Retraining {model_name} on current data...")
                            from config.settings import CLASSICAL_PARAMS
                            params = dict(CLASSICAL_PARAMS[model_name]["classification"])
                            model = ClassicalModel(model_name, "classification", params)
                            model.fit(data["X_train"], data["y_clf_train"], data["X_val"], data["y_clf_val"])
                            model.feature_names = list(data["feature_names"])
                            model.save(os.path.join(model_dir, f"{model_name}_clf.pkl"))
                            logger.info(f"Retrained and saved {model_name}")
                    else:
                        model = DeepModel.load(mpath)
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    return
            else:
                logger.error(f"No model found: {mpath}")
                return

        from data.loader import load_processed
        df = load_processed(interval)
        train_df, val_df, test_df = chronological_split(df)

        regime_labels = None
        if regime_aware:
            from backtest.engine import _predict_fold_regimes
            combined_train = pd.concat([train_df, val_df], ignore_index=True)
            regime_labels = _predict_fold_regimes(combined_train, test_df)

        trades, equity, metrics = run_backtest(
            model, model_name, test_df,
            data["X_test"], data["X_test_seq"],
            strategy=strategy, interval=interval, target_window=tw,
            zero_commission=zero_commission,
            min_confidence=min_confidence,
            min_holding_bars=min_holding_bars,
            regime_labels=regime_labels,
            bayesian=bayesian,
        )
        import json
        logger.info(f"\nBacktest results:")
        logger.info(json.dumps(metrics, indent=2, default=str))
        if len(trades) > 0:
            logger.info(f"\nFirst 10 trades:")
            logger.info(trades.head(10).to_string(index=False))


def run_regime_train(args):
    """Train regime classifier and regime-specific models."""
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"

    logger.info("=" * 60)
    logger.info(f"Regime-aware training ({interval}/{tw})")
    logger.info("=" * 60)

    df = load_processed(interval)
    data = get_datasets(interval, tw)

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    trainer = RegimeAwareTrainer()

    logger.info("\n--- Step 1: Training regime classifier ---")
    trainer.train_regime_classifier(df, train_end, val_end)

    logger.info("\n--- Step 2: Training volatility models ---")
    trainer.train_volatility_models(df)

    logger.info("\n--- Step 3: Training default (fallback) model ---")
    for task in ["regression", "classification"]:
        logger.info(f"  Task: {task}")
        trainer.train_default_model(data, model_type=args.model, task=task)

    logger.info("\n--- Step 4: Training regime-specific models ---")
    for task in ["regression", "classification"]:
        logger.info(f"  Task: {task}")
        trainer.train_regime_specific_models(
            df, data, interval, tw,
            model_type=args.model, task=task,
        )

    trainer.save(interval, tw)

    logger.info("\n--- Step 5: Evaluating regime-aware predictions ---")
    test_regime_labels = trainer.regime_classifier.predict(df.iloc[val_end:])
    regime_names = ["bear", "sideways", "bull"]
    from collections import Counter
    regime_dist = Counter(test_regime_labels)
    logger.info(f"  Test regime distribution: {dict(regime_dist)}")

    test_predictions = trainer.predict(df.iloc[val_end:], data["X_test"])
    from models.trainer import _regression_metrics
    from tabulate import tabulate

    reg_col = f"reg_target_{tw}"
    y_true = df.iloc[val_end:][reg_col].values

    n_overlap = min(len(y_true), len(test_predictions["predictions"]))
    metrics = _regression_metrics(
        y_true[:n_overlap],
        test_predictions["predictions"][:n_overlap]
    )
    logger.info(f"  Regime-aware regression metrics: {metrics}")

    logger.info("\n" + "=" * 60)
    logger.info("Regime-aware training complete!")
    logger.info("=" * 60)


def run_regime_predict(args):
    """Make regime-aware predictions."""
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"

    trainer = RegimeAwareTrainer.load(interval, tw)
    df = load_processed(interval)
    data = get_datasets(interval, tw)

    predictions = trainer.predict(df.iloc[int(len(df) * 0.85):], data["X_test"])

    test_df = df.iloc[int(len(df) * 0.85):]
    n = min(args.latest, len(predictions["predictions"]))
    latest = test_df.tail(n).reset_index(drop=True)

    regime_names = ["bear", "sideways", "bull"]
    from tabulate import tabulate
    rows = []
    for i in range(n):
        ts = str(latest.iloc[i].get("timestamp", "?"))
        price = latest.iloc[i].get("close", 0)
        pred_ret = float(predictions["predictions"][i])
        pred_price = price * (1 + pred_ret)
        regime = regime_names[int(predictions["regime_labels"][i])]
        conf = float(predictions["regime_confidence"][i])
        rows.append([ts, f"{price:.4f}", f"{pred_ret:+.6f}",
                      f"{pred_price:.4f}", regime, f"{conf:.4f}"])

    headers = ["Timestamp", "Close", "Pred_Return", "Pred_Close", "Regime", "Confidence"]
    logger.info(f"\nRegime-aware predictions ({interval}/{tw} | latest {n}):")
    logger.info(tabulate(rows, headers=headers, tablefmt="simple"))


def run_volatility(args):
    """Train and evaluate volatility models (GARCH + Kalman Filter)."""
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"

    df = load_processed(interval)
    returns = df["close"].pct_change().dropna().values
    prices = df["close"].dropna().values

    logger.info("=" * 60)
    logger.info(f"Volatility analysis ({interval}/{tw})")
    logger.info("=" * 60)

    logger.info("\n--- GARCH Ensemble ---")
    try:
        garch = GARCHEnsemble()
        garch.fit(returns)
        forecasts = garch.forecast(horizon=5)
        median_fc = garch.median_forecast(horizon=5)
        logger.info(f"  GARCH forecasts (5 periods): {median_fc}")

        regime_probs = garch.models.get("GARCH").get_regime_probabilities()
        for level, probs in regime_probs.items():
            logger.info(f"    {level} vol regime: mean={probs.mean():.4f}, "
                         f"recent={probs[-1]:.4f}")
    except Exception as e:
        logger.warning(f"GARCH failed: {e}")

    logger.info("\n--- Kalman Filter ---")
    try:
        kf = KalmanFilter()
        kf.fit(prices)
        signal = kf.get_signal()
        trend = kf.get_trend()
        lower, upper = kf.get_confidence_bounds(prices)

        pred, pred_var = kf.predict(horizon=5)
        logger.info(f"  Signal (last 5): {signal[-5:]}")
        logger.info(f"  Trend  (last 5): {trend[-5:]}")
        logger.info(f"  Prediction (5 ahead): {pred}")
        logger.info(f"  Prediction variance: {pred_var}")
    except Exception as e:
        logger.warning(f"Kalman Filter failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Volatility analysis complete!")
    logger.info("=" * 60)


def run_label_triple_barrier(args):
    """Create triple barrier targets and optionally meta-label filter."""
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"

    logger.info("=" * 60)
    logger.info(f"Triple Barrier + Meta-Label labeling ({interval}/{tw})")
    logger.info("=" * 60)

    df = load_processed(interval)
    logger.info(f"Loaded {len(df)} rows")

    # Override config if provided
    from config.settings import TRIPLE_BARRIER as TB_CFG
    config = dict(TB_CFG)
    if args.upper is not None:
        config["upper_barrier"] = args.upper
    if args.lower is not None:
        config["lower_barrier"] = args.lower
    if args.horizon is not None:
        config["time_horizon_bars"] = args.horizon

    logger.info(f"Barrier config: TP={config['upper_barrier']:.1%}, SL={abs(config['lower_barrier']):.1%}, "
                f"horizon={config['time_horizon_bars']} bars")

    # Step 1: Triple barrier targets
    logger.info("\n--- Step 1: Creating triple barrier targets ---")
    df = create_targets_triple_barrier(df, interval, config=config)

    # Step 2: Meta-label filter
    if args.meta_label:
        logger.info("\n--- Step 2: Running meta-label filter ---")
        target_col = f"tb_target_{tw}"
        df = meta_label_filter(df, target_col)
    else:
        logger.info("\n--- Step 2: Skipping meta-label filter (use --meta-label to enable) ---")

    # Save
    save_processed_data(df, interval)

    # Summary
    from collections import Counter
    tb_col = f"tb_target_{tw}"
    dist = Counter(df[tb_col].dropna())
    logger.info(f"\nTriple barrier distribution ({tw}):")
    for label in [0, 1, 2]:
        pct = dist[label] / len(df) * 100 if len(df) > 0 else 0
        name = {0: "DOWN (SL)", 1: "SIDEWAYS (time)", 2: "UP (TP)"}[label]
        logger.info(f"  {name}: {dist[label]} ({pct:.1f}%)")

    if "trade_filter" in df.columns:
        n_trade = int(df["trade_filter"].sum())
        logger.info(f"\nMeta-label filter: {n_trade}/{len(df)} samples as TRADE ({n_trade/len(df)*100:.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info("Labeling complete! Processed data saved.")
    logger.info("=" * 60)


def run_stream_start(args):
    """Start real-time streaming prediction pipeline."""
    import asyncio
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"
    model_type = args.model
    duration = getattr(args, "duration", 0)

    asyncio.run(run_stream(
        interval=interval,
        target_window=tw,
        model_type=model_type,
        duration=duration,
    ))


def run_stream_predict(args):
    """Run a single prediction on latest data (no WebSocket).

    Loads the model, computes features from the last processed data,
    and prints the prediction.
    """
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"
    model_type = args.model

    logger.info("=" * 60)
    logger.info(f"Single prediction: {interval}/{tw} | model={model_type}")
    logger.info("=" * 60)

    predictor = StreamPredictor(
        interval=interval,
        target_window=tw,
        model_type=model_type,
    )

    if not predictor.load_models():
        logger.error("Failed to load models")
        return

    df = load_processed(interval)
    logger.info(f"Loaded processed data: {len(df)} rows")

    for _, row in df.iterrows():
        candle = {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        predictor.update_feature_buffer(candle)

    X = predictor.compute_features()
    if X is None:
        logger.error("Could not compute features (insufficient data)")
        return

    prediction = predictor.predict(X)
    if prediction:
        from tabulate import tabulate
        rows = [[
            str(prediction["timestamp"]),
            f"{prediction['P_UP']:.4f}",
            prediction["direction"],
            f"{prediction['confidence']:.4f}",
        ]]
        headers = ["Timestamp", "P(UP)", "Direction", "Confidence"]
        logger.info(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        logger.error("Prediction failed")

    logger.info(f"\nRecent history ({len(predictor.get_prediction_history())} predictions)")


def run_stream_drift(args):
    """Check feature drift on processed data."""
    interval = args.interval
    window = args.window
    tw = window if "h" in str(window) else f"{window}h"
    psi_threshold = getattr(args, "psi_threshold", 0.1)

    logger.info("=" * 60)
    logger.info(f"Drift check: {interval}/{tw} | threshold={psi_threshold}")
    logger.info("=" * 60)

    data = get_datasets(interval, tw)
    feature_names = data["feature_names"]

    monitor = DriftMonitor()
    monitor.set_reference(data["X_train"], feature_names)

    n_test = len(data["X_test"])
    for i in range(n_test):
        monitor.add_sample(data["X_test"][i], feature_names)

    monitor._sample_count = n_test
    drift = monitor.check_drift(force=True)

    if drift:
        logger.info(f"\nOverall PSI: {drift['overall_psi']:.6f}")
        logger.info(f"Threshold:   {drift['threshold']:.4f}")
        logger.info(f"Drift:       {drift['drift_detected']}")
        logger.info(f"Action:      {drift['action']}")

        if drift["drift_detected"]:
            logger.info("\nTop drifting features:")
            sorted_features = sorted(
                drift["feature_psi"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            from tabulate import tabulate
            rows = [[f, round(p, 6)] for f, p in sorted_features]
            logger.info(tabulate(rows, headers=["Feature", "PSI"], tablefmt="simple"))
    else:
        logger.info("Insufficient data for drift check")


def run_stream_config(args):
    """Show or update streaming configuration."""
    import json
    from config.settings import STREAMING_CONFIG, ALERT_CONFIG, DRIFT_CONFIG

    logger.info("=" * 60)
    logger.info("Streaming Configuration")
    logger.info("=" * 60)

    logger.info("\n--- WebSocket Sources ---")
    logger.info(json.dumps(STREAMING_CONFIG.get("sources", {}), indent=2))

    logger.info("\n--- Alert Config ---")
    logger.info(json.dumps(ALERT_CONFIG, indent=2, default=str))

    logger.info("\n--- Drift Monitor Config ---")
    logger.info(json.dumps(DRIFT_CONFIG, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Quant Dynamics — Systematic ML & Market Regime Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- DATA COMMAND ---
    data_parser = subparsers.add_parser("data", help="Data pipeline operations")
    data_sub = data_parser.add_subparsers(dest="subcommand")

    fetch_p = data_sub.add_parser("fetch", help="Fetch raw OHLCV data")
    fetch_p.add_argument("--intervals", nargs="+", default=None, help="Intervals to fetch (default: all)")
    fetch_p.add_argument("--force", action="store_true", help="Re-fetch all data from scratch")
    fetch_p.add_argument("--refresh", action="store_true", help="Fetch only new candles since last fetch")
    fetch_p.add_argument("--binance", action="store_true",
                          help="Download Binance Vision historical klines (2019+) and merge with Kraken")
    fetch_p.add_argument("--derivatives", action="store_true", help="Also fetch derivatives data (funding, OI, liqs)")
    fetch_p.add_argument("--external", action="store_true", help="Also fetch external data (on-chain, macro, OB)")

    process_p = data_sub.add_parser("process", help="Process raw data with features and targets")
    process_p.add_argument("--intervals", nargs="+", default=None, help="Intervals to process")

    # --- TRAIN COMMAND ---
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--interval", required=True, help="Data interval (15m, 1h, 1d)")
    train_parser.add_argument("--window", required=True, help="Prediction window (e.g., 1h, 4h, 24h)")
    train_parser.add_argument("--model", choices=["all", "classical", "deep", "ensemble"], default="all")
    train_parser.add_argument("--evaluate", action="store_true", help="Also evaluate on test set")
    train_parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    train_parser.add_argument("--triple-barrier", action="store_true",
                               help="Train using triple barrier targets (3-class: DOWN/SIDEWAYS/UP)")
    train_parser.add_argument("--meta-filter", action="store_true",
                               help="Apply meta-label filter (only train on TRADE samples)")

    # --- PREDICT COMMAND ---
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--interval", required=True)
    predict_parser.add_argument("--window", required=True)
    predict_parser.add_argument("--model", default="ensemble", help="Which model to use")
    predict_parser.add_argument("--latest", type=int, default=10, help="Number of latest predictions to show")

    # --- BACKTEST COMMAND ---
    bt_parser = subparsers.add_parser("backtest", help="Run backtests")
    bt_parser.add_argument("--interval", required=True)
    bt_parser.add_argument("--window", required=True)
    bt_parser.add_argument("--model", default="ensemble", help="Model to backtest")
    bt_parser.add_argument("--strategy", choices=["signal_based", "confidence_based", "all"], default="all")
    bt_parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward validation")
    bt_parser.add_argument("--zero-commission", action="store_true", help="Run with 0%% commission and 0%% slippage (diagnostic)")
    bt_parser.add_argument("--wf-method", choices=["expanding", "rolling"], default="expanding",
                           help="Walk-forward window method (default: expanding)")
    bt_parser.add_argument("--min-confidence", type=float, default=None,
                           help="Only enter when model confidence >= X (reduces trade frequency)")
    bt_parser.add_argument("--min-holding-bars", type=int, default=0,
                            help="Cooldown bars after exit before next entry (reduces churn)")
    bt_parser.add_argument("--regime-aware", action="store_true",
                            help="Wire regime classifier to backtest for regime-specific risk parameters")
    bt_parser.add_argument("--bayesian", action="store_true",
                            help="Apply Bayesian belief updating to model predictions")

    # --- EVALUATE COMMAND ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument("--interval", required=True)
    eval_parser.add_argument("--window", required=True)

    # --- TEMPORAL CV COMMAND ---
    cv_parser = subparsers.add_parser("cv", help="Temporal cross-validation evaluation")
    cv_parser.add_argument("--interval", required=True)
    cv_parser.add_argument("--window", required=True)
    cv_parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    cv_parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    cv_parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    cv_parser.add_argument("--gap", type=int, default=0, help="Gap between train and val (prevents leakage)")
    cv_parser.add_argument("--calibrate", action="store_true", help="Also calibrate probabilities")

    # --- CALIBRATE COMMAND ---
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate model probabilities (Platt scaling)")
    cal_parser.add_argument("--interval", required=True)
    cal_parser.add_argument("--window", required=True)
    cal_parser.add_argument("--model", default="ensemble", help="Model to calibrate")
    cal_parser.add_argument("--method", default="sigmoid", choices=["sigmoid", "isotonic"], help="Calibration method")

    # --- REGIME TRAIN COMMAND ---
    regime_parser = subparsers.add_parser("regime-train", help="Train regime classifier + regime-specific models")
    regime_parser.add_argument("--interval", required=True)
    regime_parser.add_argument("--window", required=True)
    regime_parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])

    # --- REGIME PREDICT COMMAND ---
    regime_pred_parser = subparsers.add_parser("regime-predict", help="Make regime-aware predictions")
    regime_pred_parser.add_argument("--interval", required=True)
    regime_pred_parser.add_argument("--window", required=True)
    regime_pred_parser.add_argument("--latest", type=int, default=10)

    # --- LABEL-TRIPLE-BARRIER COMMAND ---
    tb_parser = subparsers.add_parser("label-triple-barrier", help="Create triple barrier + meta-label targets")
    tb_parser.add_argument("--interval", required=True, help="Data interval (15m, 1h, 4h, 1d)")
    tb_parser.add_argument("--window", required=True, help="Prediction window (e.g., 24h)")
    tb_parser.add_argument("--upper", type=float, default=None, help="Upper barrier %% (default: from config)")
    tb_parser.add_argument("--lower", type=float, default=None, help="Lower barrier %% (default: from config)")
    tb_parser.add_argument("--horizon", type=int, default=None, help="Time horizon in bars (default: from config)")
    tb_parser.add_argument("--meta-label", action="store_true", help="Also run meta-label filter")

    # --- VOLATILITY COMMAND ---
    vol_parser = subparsers.add_parser("volatility", help="Train and evaluate GARCH + Kalman Filter")
    vol_parser.add_argument("--interval", required=True)
    vol_parser.add_argument("--window", required=True)

    # --- STREAM COMMAND ---
    stream_parser = subparsers.add_parser("stream", help="Real-time streaming pipeline")
    stream_sub = stream_parser.add_subparsers(dest="subcommand")

    stream_start_p = stream_sub.add_parser("start", help="Start WebSocket streaming prediction")
    stream_start_p.add_argument("--interval", default="1h")
    stream_start_p.add_argument("--window", default="24h")
    stream_start_p.add_argument("--model", default="ensemble")
    stream_start_p.add_argument("--duration", type=int, default=0, help="Run duration in seconds (0=unlimited)")

    stream_predict_p = stream_sub.add_parser("predict", help="Single prediction on latest data (no WebSocket)")
    stream_predict_p.add_argument("--interval", required=True)
    stream_predict_p.add_argument("--window", required=True)
    stream_predict_p.add_argument("--model", default="ensemble")

    stream_drift_p = stream_sub.add_parser("drift", help="Check feature distribution drift (PSI)")
    stream_drift_p.add_argument("--interval", required=True)
    stream_drift_p.add_argument("--window", required=True)
    stream_drift_p.add_argument("--psi-threshold", type=float, default=0.1)

    stream_config_p = stream_sub.add_parser("config", help="Show streaming configuration")

    register_bot_parser(subparsers)

    args = parser.parse_args()

    if args.command == "data":
        if args.subcommand == "fetch":
            run_data_fetch(args)
        elif args.subcommand == "process":
            run_data_process(args)
        else:
            data_parser.print_help()
    elif args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "backtest":
        run_backtest_cmd(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "cv":
        run_temporal_cv(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    elif args.command == "regime-train":
        run_regime_train(args)
    elif args.command == "regime-predict":
        run_regime_predict(args)
    elif args.command == "volatility":
        run_volatility(args)
    elif args.command == "label-triple-barrier":
        run_label_triple_barrier(args)
    elif args.command == "stream":
        if args.subcommand == "start":
            run_stream_start(args)
        elif args.subcommand == "predict":
            run_stream_predict(args)
        elif args.subcommand == "drift":
            run_stream_drift(args)
        elif args.subcommand == "config":
            run_stream_config(args)
        else:
            stream_parser.print_help()
    elif args.command == "bot":
        run_bot_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

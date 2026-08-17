"""Feature engineering subpackage."""

from data.features.cross_asset import (
    add_btc_lead_lag_features,
    add_cross_asset_features,
    add_momentum_features,
    add_vol_adj_momentum_features,
)
from data.features.external import (
    add_derivatives_features,
    add_external_features,
    add_macro_features,
    add_onchain_features,
    add_orderbook_features,
    add_regime_features,
    add_taker_flow_features,
    add_time_features,
)
from data.features.selection import (
    audit_scale_invariance,
    select_features,
    select_features_multi_threshold,
)
from data.features.targets import (
    _interval_to_hours,
    adjust_targets_for_cost,
    create_pbr_targets,
    create_targets,
    create_targets_triple_barrier,
    meta_label_filter,
)
from data.features.technical import (
    _compute_adx,
    add_adx,
    add_atr,
    add_bollinger,
    add_ema,
    add_lag_features,
    add_macd,
    add_obv,
    add_price_patterns,
    add_rsi,
    add_sma,
    add_stochastic,
    add_volume_sma,
    add_vwap,
)

__all__ = [
    # Technical
    "add_sma",
    "add_ema",
    "add_rsi",
    "add_macd",
    "add_bollinger",
    "add_atr",
    "add_stochastic",
    "add_adx",
    "add_obv",
    "add_vwap",
    "add_volume_sma",
    "add_lag_features",
    "add_price_patterns",
    "_compute_adx",
    # Cross-Asset
    "add_cross_asset_features",
    "add_btc_lead_lag_features",
    "add_momentum_features",
    "add_vol_adj_momentum_features",
    # External
    "add_regime_features",
    "add_time_features",
    "add_external_features",
    "add_derivatives_features",
    "add_onchain_features",
    "add_macro_features",
    "add_orderbook_features",
    "add_taker_flow_features",
    # Targets
    "_interval_to_hours",
    "create_pbr_targets",
    "adjust_targets_for_cost",
    "create_targets_triple_barrier",
    "meta_label_filter",
    "create_targets",
    # Selection
    "select_features",
    "select_features_multi_threshold",
    "audit_scale_invariance",
]

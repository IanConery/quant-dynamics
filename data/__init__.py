"""Data acquisition, feature engineering, and dataset loader package."""

from data.derivatives_fetcher import (
    fetch_all_derivatives,
    fetch_funding_rates,
    fetch_liquidations,
    fetch_open_interest,
    load_derivatives_data,
    save_derivatives_data,
)
from data.external_fetcher import (
    fetch_all_external,
    fetch_macro_data,
    fetch_on_chain_data,
    fetch_order_book_data,
)
from data.fetcher import (
    fetch_all_binance_historical,
    fetch_all_intervals,
    fetch_binance_historical,
    fetch_cross_assets,
    fetch_external_data,
    fetch_ohlcv,
    load_cross_asset_data,
    load_raw_data,
    merge_all_binance,
    merge_binance_with_kraken,
    refresh_all,
    refresh_data,
    run_full_fetch,
    save_raw_data,
)
from data.loader import (
    chronological_split,
    create_sequences,
    get_datasets,
    get_tb_datasets,
    load_processed,
    scale_features,
    temporal_cross_validation,
)
from data.processor import (
    build_feature_matrix,
    build_feature_matrix_triple_barrier,
    process_raw_data,
    save_processed_data,
    select_features,
    select_features_multi_threshold,
)
from data.websocket import (
    WebSocketConnection,
    WebSocketManager,
)

__all__ = [
    # Fetcher
    "fetch_ohlcv",
    "fetch_cross_assets",
    "fetch_all_intervals",
    "fetch_external_data",
    "run_full_fetch",
    "save_raw_data",
    "load_raw_data",
    "load_cross_asset_data",
    "refresh_data",
    "refresh_all",
    "fetch_binance_historical",
    "fetch_all_binance_historical",
    "merge_binance_with_kraken",
    "merge_all_binance",
    # Derivatives
    "fetch_funding_rates",
    "fetch_open_interest",
    "fetch_liquidations",
    "save_derivatives_data",
    "load_derivatives_data",
    "fetch_all_derivatives",
    # External
    "fetch_on_chain_data",
    "fetch_macro_data",
    "fetch_order_book_data",
    "fetch_all_external",
    # Loader
    "load_processed",
    "chronological_split",
    "create_sequences",
    "scale_features",
    "get_datasets",
    "get_tb_datasets",
    "temporal_cross_validation",
    # Processor
    "process_raw_data",
    "build_feature_matrix",
    "build_feature_matrix_triple_barrier",
    "save_processed_data",
    "select_features",
    "select_features_multi_threshold",
    # WebSocket
    "WebSocketConnection",
    "WebSocketManager",
]

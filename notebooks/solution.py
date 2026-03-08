#!/usr/bin/env python3
from __future__ import annotations

"""Baseline ML pipeline in script form (7-stage workflow).

Script-first version of the baseline model development flow.

Submission naming convention for manual output names in docs/comments:
`submission_<PIPELINE>_<VARIANT>_CV<LOCAL_CV>.csv`
"""

import glob
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_NAMING_CONVENTION = "submission_<PIPELINE>_<VARIANT>_CV<LOCAL_CV>.csv"


class CONFIG:
    """Global settings copied from the notebook baseline."""

    SEED = 42
    SPLIT_RATIO = 0.90
    if os.path.exists("/kaggle/input"):
        DATA_PATH = "/kaggle/input/trademaster-cup-2025/"
    else:
        DATA_PATH = str(PROJECT_ROOT / "data" / "raw")

    # Champion model parameters from notebook.
    XGB_PARAMS = {
        "objective": "reg:absoluteerror",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_jobs": -1,
        "verbosity": 0,
        "random_state": 42,
    }


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across Python/NumPy/PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def detect_device() -> torch.device:
    """Print and return the compute device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS (Metal Performance Shaders) Acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using NVIDIA CUDA Acceleration")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (Might be slow)")
    return device


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, int, np.ndarray]:
    """Load train/test and create the same global split used in notebook."""
    print("🚀 Loading Data...")
    train_df = pd.read_csv(os.path.join(CONFIG.DATA_PATH, "train_v2.csv")).sort_values(["date_id", "minute_id"])
    test_df = pd.read_csv(os.path.join(CONFIG.DATA_PATH, "test_v2.csv")).sort_values(["date_id", "minute_id"])

    split_idx = int(len(train_df) * CONFIG.SPLIT_RATIO)
    val_ids = train_df["id"].iloc[split_idx:].values

    print(f"✅ Data Loaded. Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"🔒 Global Split Index: {split_idx} (Ratio: {CONFIG.SPLIT_RATIO})")
    print(f"🔒 Validation Rows: {len(val_ids)}")
    return train_df, test_df, split_idx, val_ids


def create_lags_fast(df: pd.DataFrame, features: list[str], lags: list[int] | None = None) -> pd.DataFrame:
    """Create per-day lagged copies of selected features."""
    if lags is None:
        lags = [1, 2, 3, 5]
    print("   Generating Lags...")
    new_cols = []
    for col in features:
        for lag in lags:
            s = df.groupby("date_id")[col].shift(lag)
            s.name = f"{col}_lag{lag}"
            new_cols.append(s)
    return pd.concat([df] + new_cols, axis=1)


def create_vip_features(df: pd.DataFrame, vip_features: list[str]) -> pd.DataFrame:
    """Add VIP rolling stats and core feature interaction terms."""
    print("   Generating VIP Interactions & Rolling Stats...")
    new_cols = []
    windows = [5, 10, 20]
    for col in vip_features:
        for w in windows:
            s_mean = df.groupby("date_id")[col].transform(lambda x: x.rolling(w).mean())
            s_mean.name = f"{col}_mean_{w}"
            new_cols.append(s_mean)
            s_std = df.groupby("date_id")[col].transform(lambda x: x.rolling(w).std())
            s_std.name = f"{col}_std_{w}"
            new_cols.append(s_std)

    f19 = df["feature_19"]
    for col in ["feature_5", "feature_27"]:
        s_mult = f19 * df[col]
        s_mult.name = f"feat_19_x_{col}"
        new_cols.append(s_mult)
    return pd.concat([df] + new_cols, axis=1)


def create_rank_features(df: pd.DataFrame, vip_cols: list[str]) -> pd.DataFrame:
    """Add rolling percentile ranks for VIP features within each day."""
    df_eng = df.copy()
    print(f"   Generating Rolling Ranks for {len(vip_cols)} VIPs...")
    for col in vip_cols:
        df_eng[f"{col}_rank"] = df_eng.groupby("date_id")[col].transform(
            lambda x: x.rolling(window=60, min_periods=10).rank(pct=True)
        )
    return df_eng


def create_delta_features(df: pd.DataFrame, vip_cols: list[str]) -> pd.DataFrame:
    """Add first-order velocity deltas using lag-1 columns."""
    df_eng = df.copy()
    print("   Generating Velocity Deltas...")
    for col in vip_cols:
        lag1 = f"{col}_lag1"
        if lag1 in df_eng.columns:
            df_eng[f"{col}_delta"] = df_eng[col] - df_eng[lag1]
    return df_eng


def create_market_features(df: pd.DataFrame, vip_cols: list[str]) -> pd.DataFrame:
    """Add expanding mean/std context and divergence features."""
    df_eng = df.copy()
    print(f"   Generating Global Context for {len(vip_cols)} VIPs...")
    for col in vip_cols:
        market_mean = df_eng.groupby("date_id")[col].transform(lambda x: x.expanding().mean())
        df_eng[f"global_mean_{col}"] = market_mean
        df_eng[f"divergence_{col}"] = df_eng[col] - market_mean
        df_eng[f"global_std_{col}"] = df_eng.groupby("date_id")[col].transform(lambda x: x.expanding().std())
    return df_eng


def create_intraday_features(df: pd.DataFrame, vip_cols: list[str]) -> pd.DataFrame:
    """Add clock-position and cumulative day-progress features."""
    df_eng = df.copy()
    print("   Generating Clock & Pseudo-Price Features...")
    df_eng["dist_from_open"] = df_eng["minute_id"]
    max_min = df_eng["minute_id"].max()
    df_eng["dist_from_close"] = max_min - df_eng["minute_id"]

    for col in vip_cols:
        df_eng[f"cum_{col}"] = df_eng.groupby("date_id")[col].cumsum()
        day_max = df_eng.groupby("date_id")[col].cummax()
        day_min = df_eng.groupby("date_id")[col].cummin()
        range_vals = day_max - day_min
        range_vals = np.where(range_vals == 0, 1, range_vals)
        df_eng[f"day_position_{col}"] = (df_eng[col] - day_min) / range_vals
    return df_eng


def run_sniper_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_idx: int,
    ignore_cols: list[str],
    base_features: list[str],
    vip_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run sniper-stage feature engineering and robust scaling."""
    print("🚀 Starting Sniper Data Pipeline (Safe Mode)...")

    train_eng = create_vip_features(create_lags_fast(train_df, base_features), vip_features)
    test_eng = create_vip_features(create_lags_fast(test_df, base_features), vip_features)

    vips = ["feature_19", "feature_5", "feature_27", "feature_2", "feature_13"]
    vips_context = ["feature_19", "feature_5", "feature_27"]

    train_eng = create_intraday_features(
        create_market_features(create_delta_features(create_rank_features(train_eng, vips), vips), vips_context),
        vips_context,
    )
    test_eng = create_intraday_features(
        create_market_features(create_delta_features(create_rank_features(test_eng, vips), vips), vips_context),
        vips_context,
    )

    final_features = [c for c in train_eng.columns if c not in ignore_cols]
    print(f"✅ Final Feature Count: {len(final_features)}")

    target_cols = ["target_short", "target_medium", "target_long"]
    x = train_eng[final_features].values
    y = train_eng[target_cols].values
    x_test = test_eng[final_features].values

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"   Splitting {len(x)} rows at index {split_idx}...")
    x_tr_raw = x[:split_idx]
    x_val_raw = x[split_idx:]
    y_tr = y[:split_idx]
    y_val = y[split_idx:]

    print("   🛡️ Fitting RobustScaler on TRAIN set only...")
    scaler = RobustScaler()
    x_tr_scaled = scaler.fit_transform(x_tr_raw)
    x_val_scaled = scaler.transform(x_val_raw)
    x_test_scaled = scaler.transform(x_test)

    x_tr_scaled = np.nan_to_num(x_tr_scaled, nan=0.0)
    x_val_scaled = np.nan_to_num(x_val_scaled, nan=0.0)
    x_test_scaled = np.nan_to_num(x_test_scaled, nan=0.0)

    print("✅ SNIPER PIPELINE COMPLETE.")
    print(f"   X_tr_scaled shape: {x_tr_scaled.shape}")
    print(f"   X_val_scaled shape: {x_val_scaled.shape}")

    return x_tr_scaled, x_val_scaled, x_test_scaled, y_tr, y_val, np.asarray(final_features), scaler.center_


def add_clusters(x_data: np.ndarray, clusters: np.ndarray, n_clusters: int = 7) -> np.ndarray:
    """Append one-hot cluster indicators to the feature matrix."""
    one_hot = np.eye(n_clusters)[clusters]
    return np.hstack([x_data, one_hot])


def get_cluster_stats(x_data: np.ndarray, cluster_ids_train: np.ndarray, target_indices: list[int]) -> dict[int, list[float]]:
    """Compute train-cluster mean per selected feature index."""
    stats: dict[int, list[float]] = {}
    for feat_idx in target_indices:
        feat_stats = []
        for c in range(7):
            mask = cluster_ids_train == c
            mean_val = x_data[mask, feat_idx].mean() if mask.sum() > 0 else 0
            feat_stats.append(mean_val)
        stats[feat_idx] = feat_stats
    return stats


def apply_cluster_deltas(
    x_data: np.ndarray, cluster_ids: np.ndarray, stats: dict[int, list[float]], target_indices: list[int]
) -> np.ndarray:
    """Add feature-minus-cluster-mean deltas for selected indices."""
    new_feats = []
    for feat_idx in target_indices:
        means = stats[feat_idx]
        cluster_means = np.array([means[c] for c in cluster_ids])
        delta = x_data[:, feat_idx] - cluster_means
        new_feats.append(delta.reshape(-1, 1))
    return np.hstack([x_data] + new_feats)


def add_king_interactions(x_data: np.ndarray, king_idx: int, other_idxs: list[int]) -> np.ndarray:
    """Add interaction terms between top scout feature and top peers."""
    new_feats = []
    king_col = x_data[:, king_idx]
    for idx in other_idxs:
        interact = king_col * x_data[:, idx]
        new_feats.append(interact.reshape(-1, 1))
    return np.hstack([x_data] + new_feats)


def run_refinery(
    x_tr_scaled: np.ndarray, x_val_scaled: np.ndarray, x_test_scaled: np.ndarray, y_tr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build refinery features via clustering, scout-selected top features, and interaction deltas."""
    print("⚖️ REFINERY: Starting Feature Engineering (Smart Mode)...")
    print("🚀 Generating Cluster Features (Honest Mode)...")

    kmeans = KMeans(n_clusters=7, random_state=CONFIG.SEED, n_init=10)
    kmeans.fit(x_tr_scaled[::10])

    tr_clusters = kmeans.predict(x_tr_scaled)
    val_clusters = kmeans.predict(x_val_scaled)
    test_clusters = kmeans.predict(x_test_scaled)

    x_tr_clustered = add_clusters(x_tr_scaled, tr_clusters)
    x_val_clustered = add_clusters(x_val_scaled, val_clusters)
    x_test_clustered = add_clusters(x_test_scaled, test_clusters)
    print(f"✅ Clusters Added. Intermediate Shape: {x_tr_clustered.shape}")

    print("   🦅 Training Scout to identify Top Features...")
    dtrain_scout = xgb.DMatrix(x_tr_clustered, label=y_tr[:, 1])
    scout = xgb.train({"tree_method": "hist", "max_depth": 4, "random_state": CONFIG.SEED}, dtrain_scout, num_boost_round=100)
    scores = scout.get_score(importance_type="total_gain")

    sorted_feats = sorted(scores, key=scores.get, reverse=True)
    top_10_indices = [int(f[1:]) for f in sorted_feats[:10]]
    top_3_indices = [int(f[1:]) for f in sorted_feats[:3]]
    best_feat_idx = top_3_indices[0]
    print(f"      👉 Top 10 Features: {top_10_indices}")

    print("   ➖ Generating Cluster Deltas (Using Top 10 Features)...")
    train_stats = get_cluster_stats(x_tr_clustered, tr_clusters, top_10_indices)
    x_tr_new = apply_cluster_deltas(x_tr_clustered, tr_clusters, train_stats, top_10_indices)
    x_val_new = apply_cluster_deltas(x_val_clustered, val_clusters, train_stats, top_10_indices)
    x_test_new = apply_cluster_deltas(x_test_clustered, test_clusters, train_stats, top_10_indices)

    print("   👑 Generating Kingmaker Interactions...")
    x_tr_refinery = add_king_interactions(x_tr_new, best_feat_idx, top_3_indices)
    x_val_refinery = add_king_interactions(x_val_new, best_feat_idx, top_3_indices)
    x_test_refinery = add_king_interactions(x_test_new, best_feat_idx, top_3_indices)
    print(f"✅ Refinery Features Ready. Final Shape: {x_tr_refinery.shape}")

    return x_tr_refinery, x_val_refinery, x_test_refinery


def weighted_cv_to_display(cv_scores: list[float]) -> float:
    """Notebook convention: weighted MAE * 100 for filename display."""
    wmae = cv_scores[0] * 0.5 + cv_scores[1] * 0.3 + cv_scores[2] * 0.2
    return wmae * 100


def center_submission_targets(sub: pd.DataFrame) -> pd.DataFrame:
    """Keep original centering behavior from notebook."""
    out = sub.copy()
    for c in out.columns[1:]:
        out[c] = out[c] - out[c].mean()
    return out


def train_champion_refinery(
    x_tr_refinery: np.ndarray,
    x_val_refinery: np.ndarray,
    x_test_refinery: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    test_ids: pd.Series,
    submissions_dir: Path,
) -> tuple[pd.DataFrame, float]:
    """Train the champion chained XGBoost model on refinery features."""
    print("🦁 Training Champion XGBoost (Legal Features) [NaN-Safe Mode]...")

    x_tr_curr = x_tr_refinery.copy()
    x_val_curr = x_val_refinery.copy()
    x_test_curr = x_test_refinery.copy()

    final_preds = np.zeros((len(x_test_curr), 3))
    cv_scores: list[float] = []

    for i in range(3):
        print(f"   🔗 Target {i}...")
        honest_feat = None
        if i < 2:
            honest_feat = np.full(len(x_tr_curr), np.nan)
            tscv = TimeSeriesSplit(n_splits=5)
            for t_idx, v_idx in tscv.split(x_tr_curr):
                m = xgb.train(CONFIG.XGB_PARAMS, xgb.DMatrix(x_tr_curr[t_idx], y_tr[t_idx, i]), num_boost_round=100)
                honest_feat[v_idx] = m.predict(xgb.DMatrix(x_tr_curr[v_idx]))

        dtrain = xgb.DMatrix(x_tr_curr, label=y_tr[:, i])
        dval = xgb.DMatrix(x_val_curr, label=y_val[:, i])
        model = xgb.train(
            CONFIG.XGB_PARAMS,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        score = model.best_score
        cv_scores.append(score)
        print(f"      ✅ Target {i} Best Score: {score:.6f}")

        test_pred = model.predict(xgb.DMatrix(x_test_curr))
        val_pred = model.predict(dval)
        final_preds[:, i] = test_pred

        if i < 2 and honest_feat is not None:
            x_tr_curr = np.hstack([x_tr_curr, honest_feat.reshape(-1, 1)])
            x_val_curr = np.hstack([x_val_curr, val_pred.reshape(-1, 1)])
            x_test_curr = np.hstack([x_test_curr, test_pred.reshape(-1, 1)])

    sub = pd.DataFrame({"id": test_ids})
    sub["target_short"] = final_preds[:, 0]
    sub["target_medium"] = final_preds[:, 1]
    sub["target_long"] = final_preds[:, 2]
    sub = center_submission_targets(sub)

    avg_cv = weighted_cv_to_display(cv_scores)
    # Keep timestamp for run uniqueness; docs/examples use SUBMISSION_NAMING_CONVENTION.
    filename = f"submission_XGB_Refinery_CV{avg_cv:.5f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = submissions_dir / filename
    sub.to_csv(output_path, index=False)
    print(f"🚀 SAVED: {output_path}")
    return sub, avg_cv


def train_purist(
    x_tr_scaled: np.ndarray,
    x_val_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    test_ids: pd.Series,
    submissions_dir: Path,
) -> tuple[pd.DataFrame, float]:
    """Train the alternative 'purist' XGBoost model."""
    print("   🌲 Training 'Purist' Model (Base Features Only)...")
    final_preds = np.zeros((len(x_test_scaled), 3))
    cv_scores: list[float] = []

    for i in range(3):
        dtrain = xgb.DMatrix(x_tr_scaled, label=y_tr[:, i])
        dval = xgb.DMatrix(x_val_scaled, label=y_val[:, i])
        params = CONFIG.XGB_PARAMS.copy()
        params["max_depth"] = 4
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        cv_scores.append(model.best_score)
        print(f"      ✅ Target {i} Best Score: {model.best_score:.6f}")
        final_preds[:, i] = model.predict(xgb.DMatrix(x_test_scaled))

    sub = pd.DataFrame({"id": test_ids})
    sub["target_short"] = final_preds[:, 0]
    sub["target_medium"] = final_preds[:, 1]
    sub["target_long"] = final_preds[:, 2]
    sub = center_submission_targets(sub)

    avg_cv = weighted_cv_to_display(cv_scores)
    # Keep timestamp for run uniqueness; docs/examples use SUBMISSION_NAMING_CONVENTION.
    filename = f"submission_XGB_Purist_CV{avg_cv:.5f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = submissions_dir / filename
    sub.to_csv(output_path, index=False)
    print(f"      🚀 Saved Purist: {filename}")
    return sub, avg_cv


def train_robust(
    x_tr_refinery: np.ndarray,
    x_val_refinery: np.ndarray,
    x_test_refinery: np.ndarray,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    test_ids: pd.Series,
    submissions_dir: Path,
) -> tuple[pd.DataFrame, float]:
    """Train the alternative 'robust' XGBoost model."""
    print("   🛡️ Training 'Robust' Model (Deep Trees)...")
    final_preds = np.zeros((len(x_test_refinery), 3))
    cv_scores: list[float] = []

    params = CONFIG.XGB_PARAMS.copy()
    params["max_depth"] = 8
    params["learning_rate"] = 0.01
    params["subsample"] = 0.6

    for i in range(3):
        dtrain = xgb.DMatrix(x_tr_refinery, label=y_tr[:, i])
        dval = xgb.DMatrix(x_val_refinery, label=y_val[:, i])
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        cv_scores.append(model.best_score)
        print(f"      ✅ Target {i} Best Score: {model.best_score:.6f}")
        final_preds[:, i] = model.predict(xgb.DMatrix(x_test_refinery))

    sub = pd.DataFrame({"id": test_ids})
    sub["target_short"] = final_preds[:, 0]
    sub["target_medium"] = final_preds[:, 1]
    sub["target_long"] = final_preds[:, 2]
    sub = center_submission_targets(sub)

    avg_cv = weighted_cv_to_display(cv_scores)
    # Keep timestamp for run uniqueness; docs/examples use SUBMISSION_NAMING_CONVENTION.
    filename = f"submission_XGB_Robust_CV{avg_cv:.5f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = submissions_dir / filename
    sub.to_csv(output_path, index=False)
    print(f"      🚀 Saved Robust: {filename}")
    return sub, avg_cv


def blend_refinery_and_robust(
    sub_refinery: pd.DataFrame,
    sub_robust: pd.DataFrame | None,
    cv_ref: float | None,
    cv_rob: float | None,
    submissions_dir: Path,
) -> Path:
    """Build the final 50/50 ensemble from refinery and robust submissions."""
    print("⚗️ MIXING ENSEMBLE: 50% Refinery + 50% Robust...")

    if sub_robust is None:
        robust_files = glob.glob(str(submissions_dir / "submission_XGB_Robust_*.csv"))
        if not robust_files:
            raise ValueError("⚠️ Missing Robust predictions. Run robust training or ensure robust CSV exists.")
        latest_rob = max(robust_files, key=os.path.getctime)
        print(f"   📂 Loaded Robust from disk: {latest_rob}")
        sub_robust = pd.read_csv(latest_rob)

    w_ref = 0.5
    w_rob = 0.5
    cols = ["target_short", "target_medium", "target_long"]
    sub_ens = pd.DataFrame({"id": sub_refinery["id"]})
    for c in cols:
        sub_ens[c] = sub_refinery[c] * w_ref + sub_robust[c] * w_rob

    cv_ref_val = cv_ref if cv_ref is not None else 0.67309
    cv_rob_val = cv_rob if cv_rob is not None else 0.67037
    ens_cv = cv_ref_val * w_ref + cv_rob_val * w_rob

    # Keep timestamp for run uniqueness; docs/examples use SUBMISSION_NAMING_CONVENTION.
    filename = f"submission_Ensemble_Ref50_Rob50_CV{ens_cv:.5f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = submissions_dir / filename
    sub_ens.to_csv(output_path, index=False)

    print(f"🚀 SAVED ENSEMBLE: {filename}")
    print(f"   (Weights: Refinery {w_ref} | Robust {w_rob})")
    return output_path


def main() -> None:
    """Run the full converted baseline pipeline."""
    print("# --- Stage 1: Global Setup and Determinism ---")
    print(f"📎 Naming convention (docs/examples): {SUBMISSION_NAMING_CONVENTION}")
    seed_everything(CONFIG.SEED)
    _device = detect_device()
    print(f"🔧 Global Configuration Loaded. Data Path: {CONFIG.DATA_PATH}")

    print("\n# --- Stage 2: Data Loading and Split Definition ---")
    train_df, test_df, split_idx, _val_ids = load_data()

    target_cols = ["target_short", "target_medium", "target_long"]
    ignore_cols = ["id", "date_id", "minute_id", "row_id"] + target_cols
    base_features = [c for c in train_df.columns if c not in ignore_cols]
    vip_features = ["feature_19", "feature_5", "feature_27", "feature_2", "feature_13"]

    print("\n# --- Stage 3: Sniper Feature Pipeline (Engineering + Scaling) ---")
    x_tr_scaled, x_val_scaled, x_test_scaled, y_tr, y_val, _final_features, _scaler_center = run_sniper_pipeline(
        train_df=train_df,
        test_df=test_df,
        split_idx=split_idx,
        ignore_cols=ignore_cols,
        base_features=base_features,
        vip_features=vip_features,
    )

    print("\n# --- Stage 4: Refinery Feature Injection (Clustering + Scout) ---")
    x_tr_refinery, x_val_refinery, x_test_refinery = run_refinery(
        x_tr_scaled=x_tr_scaled,
        x_val_scaled=x_val_scaled,
        x_test_scaled=x_test_scaled,
        y_tr=y_tr,
    )

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    print("\n# --- Stage 5: Champion Chained XGBoost on Refinery Features ---")
    sub_refinery, avg_cv_refinery = train_champion_refinery(
        x_tr_refinery=x_tr_refinery,
        x_val_refinery=x_val_refinery,
        x_test_refinery=x_test_refinery,
        y_tr=y_tr,
        y_val=y_val,
        test_ids=test_df["id"],
        submissions_dir=submissions_dir,
    )

    print("\n# --- Stage 6: Alternative Models for Diversity (Purist + Robust) ---")
    print("🧪 Training Alternative Models (Purist & Robust)...")
    _sub_purist, _avg_cv_purist = train_purist(
        x_tr_scaled=x_tr_scaled,
        x_val_scaled=x_val_scaled,
        x_test_scaled=x_test_scaled,
        y_tr=y_tr,
        y_val=y_val,
        test_ids=test_df["id"],
        submissions_dir=submissions_dir,
    )
    sub_robust, avg_cv_robust = train_robust(
        x_tr_refinery=x_tr_refinery,
        x_val_refinery=x_val_refinery,
        x_test_refinery=x_test_refinery,
        y_tr=y_tr,
        y_val=y_val,
        test_ids=test_df["id"],
        submissions_dir=submissions_dir,
    )

    print("\n# --- Stage 7: Final 50/50 Ensemble (Refinery + Robust) ---")
    blend_refinery_and_robust(
        sub_refinery=sub_refinery,
        sub_robust=sub_robust,
        cv_ref=avg_cv_refinery,
        cv_rob=avg_cv_robust,
        submissions_dir=submissions_dir,
    )


if __name__ == "__main__":
    main()

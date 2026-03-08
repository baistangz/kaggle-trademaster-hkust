#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET_COLS = ["target_short", "target_medium", "target_long"]
VIP_FEATURES = ["feature_19", "feature_5", "feature_27", "feature_2", "feature_13"]
SEEDS = [42, 43, 44, 101, 777]
N_FOLDS = 5
EPOCHS = 25
BATCH_SIZE = 2048
LEARNING_RATE = 5e-4
TARGET_SCALE = 100.0
KMEANS_CLUSTERS = 7
KMEANS_SAMPLE_STRIDE = 10
LAGS = [1, 2, 3, 5]
ROLL_WINDOWS = [5, 10, 20]
OUTPUT_NAME = "submission_resnet_honest.csv"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the archival neural-network baseline submission. "
            "Default output: submissions/submission_resnet_honest.csv"
        )
    )
    parser.add_argument(
        "--output-name",
        default=OUTPUT_NAME,
        help="Submission filename written into the project submissions folder.",
    )
    return parser.parse_args()


def resolve_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]

    if Path("/kaggle/input").exists():
        search_path = Path("/kaggle/input")
        for root, _dirs, files in os.walk(search_path):
            if "train_v2.csv" in files:
                data_path = Path(root)
                break
        else:
            raise FileNotFoundError("Could not locate train_v2.csv under /kaggle/input")
    else:
        data_path = project_root / "data" / "raw"

    return project_root, data_path


def load_raw_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("   Loading raw train/test data...")
    train_df = pd.read_csv(data_path / "train_v2.csv")
    test_df = pd.read_csv(data_path / "test_v2.csv")

    # Sorting is necessary because all engineered lags and rolling windows are
    # defined in day/minute order.
    train_df = train_df.sort_values(["date_id", "minute_id"]).reset_index(drop=True)
    test_df = test_df.sort_values(["date_id", "minute_id"]).reset_index(drop=True)
    return train_df, test_df


def engineer_safe_features(df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    """Build the original notebook's "safe mode" NN feature set."""
    df_eng = df.copy()

    for col in base_features:
        grouped = df_eng.groupby("date_id")[col]
        for lag in LAGS:
            df_eng[f"{col}_lag{lag}"] = grouped.shift(lag)

    for window in ROLL_WINDOWS:
        for col in VIP_FEATURES:
            grouped = df_eng.groupby("date_id")[col]
            df_eng[f"{col}_mean_{window}"] = grouped.transform(
                lambda x: x.rolling(window).mean()
            )
            df_eng[f"{col}_std_{window}"] = grouped.transform(
                lambda x: x.rolling(window).std()
            )

    # The original notebook ranked each value only against a recent rolling
    # context rather than against the entire day.
    for col in VIP_FEATURES:
        grouped = df_eng.groupby("date_id")[col]
        df_eng[f"{col}_rank"] = grouped.transform(
            lambda x: x.rolling(window=60, min_periods=10).rank(pct=True)
        ) - 0.5

    return df_eng


def build_feature_matrices(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ignore = ["id", "date_id", "minute_id"] + TARGET_COLS
    base_features = [c for c in train_df.columns if c not in ignore]

    print("   Engineering archival NN features...")
    train_eng = engineer_safe_features(train_df, base_features)
    test_eng = engineer_safe_features(test_df, base_features)

    y = train_eng[TARGET_COLS].values
    final_cols = [c for c in train_eng.columns if c not in ignore]
    x_train = train_eng[final_cols].values
    x_test = test_eng[final_cols].values

    del train_eng, test_eng, train_df, test_df
    gc.collect()

    # Preserve notebook behavior: aggressively zero-fill missing/inf values.
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)
    return x_train, x_test, y, final_cols


def scale_and_cluster(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("   Robust scaling features...")
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("   Generating KMeans market-state embeddings...")
    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(x_train_scaled[::KMEANS_SAMPLE_STRIDE])
    train_clusters = kmeans.predict(x_train_scaled)
    test_clusters = kmeans.predict(x_test_scaled)
    return x_train_scaled, x_test_scaled, train_clusters, test_clusters


class WeightedMAELoss(nn.Module):
    def __init__(self, weights: list[float] | None = None) -> None:
        super().__init__()
        if weights is None:
            weights = [0.5, 0.3, 0.2]
        self.weights = torch.tensor(weights, device=DEVICE)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        abs_err = torch.abs(inputs - targets)
        weighted_err = torch.sum(abs_err * self.weights, dim=1)
        return torch.mean(weighted_err)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SniperResNet(nn.Module):
    def __init__(self, num_inputs: int, hidden_dim: int = 512, output_dim: int = 3, dropout: float = 0.25) -> None:
        super().__init__()
        self.embedding = nn.Embedding(KMEANS_CLUSTERS, 16)
        self.entry = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        )
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim + 16, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x_cat)
        x = self.entry(x_num)
        x = self.res_blocks(x)
        concat = torch.cat([x, emb], dim=1)
        return self.merge(concat)


def train_ensemble(
    x_train_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    print(f"Training archival ResNet on {DEVICE}...")

    x_num_tensor = torch.FloatTensor(x_train_scaled).to(DEVICE)
    x_test_num_tensor = torch.FloatTensor(x_test_scaled).to(DEVICE)
    cluster_train_tensor = torch.LongTensor(train_clusters).to(DEVICE)
    cluster_test_tensor = torch.LongTensor(test_clusters).to(DEVICE)
    y_tensor = torch.FloatTensor(y * TARGET_SCALE).to(DEVICE)

    ensemble_preds = np.zeros((len(x_test_scaled), len(TARGET_COLS)))

    for seed in SEEDS:
        print(f"   Seed {seed}...")
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        seed_preds = np.zeros((len(x_test_scaled), len(TARGET_COLS)))

        for train_idx, val_idx in kfold.split(x_num_tensor, y_tensor):
            train_ds = TensorDataset(
                x_num_tensor[train_idx],
                cluster_train_tensor[train_idx],
                y_tensor[train_idx],
            )
            val_ds = TensorDataset(
                x_num_tensor[val_idx],
                cluster_train_tensor[val_idx],
                y_tensor[val_idx],
            )

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

            model = SniperResNet(num_inputs=x_num_tensor.shape[1]).to(DEVICE)
            criterion = WeightedMAELoss(weights=[0.5, 0.3, 0.2])
            optimizer = optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=LEARNING_RATE,
                steps_per_epoch=len(train_loader),
                epochs=EPOCHS,
            )

            best_wmae = float("inf")
            best_weights = copy.deepcopy(model.state_dict())

            for _epoch in range(EPOCHS):
                model.train()
                for bx_num, bx_cat, by in train_loader:
                    optimizer.zero_grad()
                    out = model(bx_num, bx_cat)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx_num, bx_cat, by in val_loader:
                        out = model(bx_num, bx_cat)
                        val_loss += criterion(out, by).item() * bx_num.size(0)

                avg_wmae = val_loss / len(val_ds)
                if avg_wmae < best_wmae:
                    best_wmae = avg_wmae
                    best_weights = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_weights)
            model.eval()

            with torch.no_grad():
                fold_outputs = []
                test_loader = DataLoader(
                    TensorDataset(x_test_num_tensor, cluster_test_tensor),
                    batch_size=4096,
                )
                for bx_num, bx_cat in test_loader:
                    fold_outputs.append(model(bx_num, bx_cat).cpu().numpy())
                seed_preds += np.concatenate(fold_outputs) / N_FOLDS

            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ensemble_preds += seed_preds / len(SEEDS)
        print(f"      completed seed {seed}")

    return ensemble_preds / TARGET_SCALE


def save_submission(data_path: Path, output_path: Path, preds: np.ndarray) -> None:
    submission = pd.read_csv(data_path / "test_v2.csv", usecols=["id"])
    submission["target_short"] = preds[:, 0]
    submission["target_medium"] = preds[:, 1]
    submission["target_long"] = preds[:, 2]

    # Preserve notebook behavior: center each target around zero before export.
    for col in TARGET_COLS:
        submission[col] = submission[col] - submission[col].mean()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved archival NN submission to {output_path}")


def main() -> None:
    args = parse_args()
    project_root, data_path = resolve_paths()
    output_path = project_root / "submissions" / args.output_name

    print(f"Initializing archival neural baseline on {DEVICE}...")
    train_df, test_df = load_raw_data(data_path)
    x_train, x_test, y, _feature_cols = build_feature_matrices(train_df, test_df)
    x_train_scaled, x_test_scaled, train_clusters, test_clusters = scale_and_cluster(
        x_train, x_test
    )
    preds = train_ensemble(
        x_train_scaled, x_test_scaled, train_clusters, test_clusters, y
    )
    save_submission(data_path, output_path, preds)


if __name__ == "__main__":
    main()

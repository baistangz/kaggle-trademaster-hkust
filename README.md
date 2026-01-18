# TradeMaster Cup 2025 by HKUST(GZ) FinTech Thrust

**Current Best Local CV:** -- | **Leaderboard:** 0.7827  
**Status:** testing

## Objective
Predicting short, medium, and long-term stock movements using High-Frequency Trading (HFT) data. The metric is **Mean Absolute Error (MAE)**.

## Project Structure

* `notebooks/`:
    * `solution.ipynb`: The main XGBoost/LightGBM/CatBoost ensemble.
* `data/`:
    * `raw/`: Immutable original competition data.
    * `processed/`: Feature-engineered datasets (NPZ format).
* `submissions/`: Final CSVs ready for Kaggle upload.

## 🛠️ Key Strategies
1.  **Sniper Pipeline:** Aggressive feature engineering targeting specific "VIP" features (Vol, Trend, Spread).
2.  **Titanium Blend:** A weighted ensemble of XGBoost (40%), LightGBM (30%), and CatBoost (30%).
3.  **RankGauss:** Post-processing calibration to force normal distribution on predictions.

## 🚀 How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run `solution.ipynb` to generate the base models.
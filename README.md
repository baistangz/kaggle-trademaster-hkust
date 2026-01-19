# TradeMaster Cup 2025 by HKUST(GZ) FinTech Thrust

**Current Best Local CV:** -- | **Leaderboard:** 0.7827  
**Status:** testing

## Objective
--. The metric is **Mean Absolute Error (MAE)**.

## Project Structure

* `data/`:
    * `raw/`: Immutable original competition data.
    * `processed/`: Feature-engineered datasets (NPZ format).
* `notebooks/`:
    * `solution.ipynb`: The main XGBoost/LightGBM/CatBoost ensemble.
    * `neuralnetworks.ipynb`: MLP model.
    * `ensemble.ipynb`: Blend trees and NN.
* `submissions/`: Final CSVs ready for Kaggle upload.

## Key Strategies
1.  **Sniper Pipeline:** Aggressive feature engineering targeting specific "VIP" features (Vol, Trend, Spread).
2.  **Titanium Blend:** A weighted ensemble of XGBoost (40%), LightGBM (30%), and CatBoost (30%).
3.  **RankGauss:** Post-processing calibration to force normal distribution on predictions.

## How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run `solution.ipynb` to generate the base models.
3.  Run `neuralnetworks.ipynb` to generate the NN models.
4.  Run `ensemble.ipynb` to generate the final blended model.
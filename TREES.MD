# Tree Models Experiments Log (TradeMaster Cup 2025)

## 🏆 Best Submission (Current Baseline)
* **Model:** Titanium Blend (Classic / Restored)
* **Configuration:**
    * **Models:** XGBoost (Hist) + LightGBM (GBDT) + CatBoost.
    * **Parameters:** Manual "Purist" settings (LR=0.03, Depth=6).
    * **Logic:** Chained Predictions (Short $\to$ Medium $\to$ Long).
    * **Weights:** XGB (0.50) | LGB (0.30) | CAT (0.20).
* **LB Score:** **0.7848** (WMAE - Lower is Better).
* **Status:** Secured & Safe.

---

## 🧪 Experiment History (Chronological)

### 1. The "Titanium Blend" (Baseline)
* **Concept:** Standard ensemble with manual weights (0.5/0.3/0.2).
* **LB Score:** **0.7848**.
* **Verdict:** The Gold Standard. Robust and reliable.

### 2. The "Filtered" Trees (Failed)
* **Concept:** Removing constant features.
* **LB Score:** **0.7880** (+0.0032 degradation).
* **Verdict:** Trees handle noise better than we thought. Don't touch the features.

# experiment 1:

### 3. Weight Optimization (Failed)
* **Concept:** Using `scipy.optimize` / Random Search to find "perfect" weights on Validation data.
* **Result:** Weights favored LGB/CAT, dropped XGB to ~0%.
* **LB Score:** **0.7929** (+0.0081 degradation).
* **Diagnosis:** **Overfitting.** The validation set hated XGBoost, but the Public LB needs it. Dropping the XGB anchor caused the error to rise.

# experiment 2

### 4. Optuna Tuning (Failed)
* **Concept:** Optimizing Hyperparameters for all 3 models using 3-fold TimeSeries CV.
    * XGB: "Low & Slow" (LR 0.01, Depth 5).
    * LGB: High leaves (136).
    * CAT: Deep trees (Depth 9).
* **Result:** Blended with "Optimized Weights".
* **LB Score:** **0.7864** (+0.0016 degradation).
* **Diagnosis:** **Underfitting / Fragility.** The "Low & Slow" XGBoost likely underfitted the trend. The manual parameters (LR 0.03) struck a better balance between signal and noise.

# experiment 3

### 5. Seed Averaging - Broken Chain (CRITICAL FAILURE)
* **Concept:** Train the "Classic" models with 3 seeds each to reduce noise.
* **Implementation Mistake:** Simplified the training loop and **removed the Chaining logic** (didn't use Target Short to predict Target Medium).
* **LB Score:** **0.9139** (+0.1291 explosion).
* **Diagnosis:** **The Chain is Vital.** Losing the "Trend Signal" (previous targets) destroyed the model's ability to predict longer horizons.

### 6. The Restoration (Success)
* **Concept:** Revert code to the exact logic of Experiment 1.
* **LB Score:** **0.7848**.
* **Verdict:** Baseline secured. We learned that **Chaining** and **Manual Weights** are the load-bearing pillars of this solution.

# experiment 4

### 7. Seeded Chains (In Progress)
* **Concept:** Run the "Classic" logic (Experiment 1) but train 3 seeds per model.
* **Crucial Detail:** The "Chaining" is preserved *inside* the seed loop (Seed 42 T0 $\to$ Seed 42 T1).
* **Goal:** Reduce error from `0.7848` $\to$ `~0.7840` by averaging out tree variance.

# experiment 5

### 8. Residual Learning / Delta Strategy (Failed)
* **Concept:** Instead of predicting total return, predict the *change* from the previous timeframe (Short → Medium, Medium → Long).
* **LB Score:** **0.7890** (+0.0042 Degradation).
* **Diagnosis:** **Error Compounding.** Small errors in Model 1 (Short) were passed to Model 2 and 3, amplifying the total error. The "Absolute" prediction method (Baseline) allows models to self-correct.

---

## 🔑 Key Learnings

1.  **Metric Reality:** **Lower is Better (WMAE).** We initially thought higher was better, which led to confusion. `0.7848` is our lowest error.
2.  **Chaining is Non-Negotiable:** Predicting `Target Short` $\to$ `Target Medium` $\to$ `Target Long` adds massive value. Removing this dependency spiked the error from 0.78 to 0.91.
3.  **XGBoost is the Anchor:** Despite lower validation scores, XGBoost (at 50% weight) provides critical stability on the Public LB. Optimization attempts that minimize XGBoost consistently fail.
4.  **Manual > Optuna (So Far):** Our manual parameters (Depth 6, LR 0.03) are outperforming Optuna's "Low & Slow" suggestions. The market noise might be punishing overly delicate models.

---

## 📝 Next Steps
1.  **Submit Seeded Chains:** If this improves the score even slightly (e.g., 0.7845), it becomes the new baseline.
2.  **Feature Interaction:** Instead of changing models, we should try adding **Interaction Features** (e.g., `feature_1 * feature_2`) to give the trees explicit signals.
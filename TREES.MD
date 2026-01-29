# Tree Models Experiments Log (TradeMaster Cup 2025)

## 🏆 Best Submission (Current Champion)
* **Model:** Optimized Titanium Blend (XGBoost + LightGBM + CatBoost)
* **Weights:** Target-Dependent (Found via Random Search)
    * **Short:** XGB (0.002) | LGB (0.610) | CAT (0.387)
    * **Medium:** XGB (0.044) | LGB (0.203) | CAT (0.753)
    * **Long:** XGB (0.260) | LGB (0.144) | CAT (0.595)
* **Data:** Full Tree Dataset (221 Features, Clustered) - **NO filtering applied**.
* **Processing:** Features generated in `solution.ipynb` (Cell 8) and loaded directly from `../data/processed/trees/`.
* **LB Score:** 0.7929 (Note: This score is slightly worse than the 0.7848 baseline, indicating potential overfitting to the validation set or a regime shift in the test set).

---

## 🧪 Experiment History

### 1. The "Titanium Blend" (Baseline)
* **Concept:** A robust ensemble of the three major Gradient Boosting libraries using "Purist" Chain logic (predicting T0 -> T1 -> T2).
* **Weights:** Manual Fixed Weights: XGB (0.50) | LGB (0.30) | CAT (0.20).
* **Data:** Raw Clustered Data (221 Features).
* **LB Score:** **0.7848** (Best Score So Far).
* **Verdict:** **Gold Standard.** The manual weights, while theoretically suboptimal, provided the most robust generalization to the Public LB. XGBoost's high weight likely stabilized the ensemble despite its lower validation performance in later experiments.

### 2. The "Filtered" Trees (Failed Experiment)
* **Concept:** Remove "constant" (Zero Variance) features to clean the dataset before training.
* **Method:** Applied `VarianceThreshold(threshold=0.0)` in Cell 9.
* **Features:** Dropped from 221 -> 209 features.
* **LB Score:** **0.7880** (Worse by +0.0032).
* **Diagnosis:** **Column Misalignment / Information Loss.**
    * **Reason 1:** `VarianceThreshold` fits on Train but transforms Validation/Test blindly. A feature might be constant in Train but non-constant in Test (rare events). Dropping it removes signal.
    * **Reason 2:** XGBoost/LightGBM rely on column indices if feature names aren't perfectly preserved. Removing columns shifted the indices, potentially confusing the models if they expected "Feature 50" to be at index 50.
* **Fix:** Reverted to loading the raw `X_tr_final` (221 features) directly. Score returned to baseline.

### 3. Weight Optimization (Random Search)
* **Concept:** Use the Validation Set (last 10% of time) to find the mathematically optimal weights for each target instead of guessing.
* **Method:** Random Search (10,000 iterations) using `scipy.optimize` and Dirichlet distributions.
* **Resulting Weights:**
    * **Short:** Heavily favored **LightGBM** (61%).
    * **Medium/Long:** Heavily favored **CatBoost** (75% / 60%).
    * **XGBoost:** Almost completely discarded (< 5% weight for Short/Medium).
* **LB Score:** **0.7929** (Worse than Baseline 0.7848).
* **Diagnosis:** **Validation Overfitting.**
    * The optimizer aggressively maximized the score on the *specific* 13,940 rows of the validation set.
    * Because the validation set is just the "last 10%" of the training data, it likely has a specific market regime. CatBoost and LightGBM overfitted to this specific regime more than XGBoost.
    * XGBoost, being more conservative (and heavily regularized in our config), performed "worse" on the validation set but generalized "better" to the unseen Public LB (Test Set).
* **Lesson:** "Optimal" on local validation does not always mean "Optimal" on the Leaderboard. A conservative blend (50% XGB) acted as a safety anchor.

---

## 🔑 Key Learnings

1.  **Do Not Filter Features for Trees:** Even "useless" constant features should be kept. Trees handle them naturally (by ignoring them). Removing them introduces risk of misalignment or leakage without adding performance.
2.  **XGBoost is the Anchor:** Despite lower standalone validation scores compared to CatBoost/LGBM in specific splits, XGBoost provides stability. Dropping its weight below 20-30% hurts generalization.
3.  **Validation vs. Leaderboard Gap:** There is a divergence between our Local Validation (last 10%) and the Public LB.
    * Local Val prefers CatBoost/LGBM.
    * Public LB prefers XGBoost/Blends.
    * **Action:** Trust the 0.7848 baseline weights (50/30/20) or try a milder optimization (e.g., constrained optimization where no model drops below 20%).

---

## 📝 Next Steps (Back to Trees)
1.  **Revert Weights:** Go back to the **0.7848** weights (XGB 0.5 | LGB 0.3 | CAT 0.2) or try a balanced mix (0.33 each).
2.  **Hyperparameter Tuning:** We used fixed params. We should run Optuna on the *trees* to squeeze performance, rather than just adjusting weights.
3.  **Feature Engineering:** The 0.7848 score came from the "Cluster" features. We can try adding "Interaction" features (e.g., `feature_X * feature_Y`) specifically for the trees.
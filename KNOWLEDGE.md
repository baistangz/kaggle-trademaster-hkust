# Neural Network Experiments Log (TradeMaster Cup 2025)

## 🏆 Current Baseline (The Target)
* **Model:** Titanium Blend (XGBoost 50% + LightGBM 30% + CatBoost 20%)
* **Data:** Full Tree Dataset (221 Features, Clustered)
* **Performance:** ~0.0028 - 0.0030 MAE (Validation)
* **LB Score:** 0.7848

---

## 🧪 Experiment History

### 1. The "Clean" MLP (High-Signal Diet)
* **Concept:** Feed the NN only the most important features to avoid noise.
* **Data:** Top 40 Features selected by LightGBM (Variance Threshold + Twin Removal).
* **Architecture:** 3-Layer MLP, Dropout 0.3.
* **Target:** Z-Scored (StandardScaler).
* **Result:** ~0.8032 (Score)
* **Verdict:** **Starvation.** The model was stable but lacked the information density of the 221-feature Tree models.

### 2. The "Full" MLP (All Features)
* **Concept:** Give the NN the exact same 221 features as the Trees.
* **Data:** Full 221 Features (Trees dataset).
* **Architecture:** `RobustMLP` (3-Layer), Dropout 0.15, L1 Loss.
* **Result:** ~0.8042 (Score)
* **Verdict:** **Noise Intolerance.** Performance degraded compared to Experiment 1. The MLP could not filter out the noise in the extra 180 features as effectively as decision trees do.

### 3. The "Smart" MLP (Middle Path)
* **Concept:** Compromise between starvation and noise.
* **Data:** Top 90 High-Variance Features + 7 Cluster Features.
* **Architecture:** `SmartMLP` (3-Layer), Dropout 0.2.
* **Result:**
    * **Correlation with Trees:** -0.0321 (Zero)
    * **Score:** Poor.
* **Verdict:** **Model Collapse.** The model failed to find any signal and likely collapsed to predicting the global mean (all zeros), resulting in zero correlation with the actual price movements predicted by Trees.

### 4. ResNet MLP (Deep Residual Network)
* **Concept:** Use Residual connections (Skip connections) to help the model ignore noise in the Full 221 dataset.
* **Data:** Full 221 Features.
* **Architecture:** `ResNetMLP` (3 Residual Blocks).
* **Result:**
    * **Val MAE:** Stuck at ~0.500 (Scaled).
    * **Correlation:** 0.0287.
* **Verdict:** **Convergence Failure.** The deep network could not propagate the gradient effectively through the noise. It learned to predict "0" (the average) for everything.

### 5. Wide & Shallow MLP (No Scaling)
* **Concept:** Use a single massive layer to act as a "fuzzy lookup table" and predict raw targets to avoid scaling artifacts.
* **Data:** Full 221 Features.
* **Architecture:** `WideMLP` (1 Layer, 2048 Units), Raw Targets (No Scaling).
* **Result:**
    * **Val MAE:** ~0.0094 (Raw).
    * **Correlation:** 0.0132.
* **Verdict:** **Scale Mismatch.** While better than ResNet, the error (0.009) was still **3x worse** than Trees (0.003). The NN struggled to predict tiny floating-point numbers directly.

### 6. RankGauss + Wide MLP (Distribution Fix)
* **Concept:** Neural Networks prefer Gaussian (Normal) distributions. We forced all 221 features into a Bell Curve using `QuantileTransformer`.
* **Data:** `nn_rankgauss` (Full 221 Features, RankGauss transformed).
* **Architecture:** `WideMLP` (1 Layer), Raw Targets.
* **Result:**
    * **Val MAE:** ~0.0104.
    * **Correlation:** 0.2743.
* **Verdict:** **Partial Success (Correlation), Failure (Accuracy).** The correlation jumped to 0.27, proving the model finally started learning *some* patterns. However, the accuracy (0.0104) remained unacceptably poor compared to Trees.

### 7. Final Attempt (RankGauss + Scaled Targets)
* **Concept:** Combine RankGauss inputs with Scaled Targets to give the NN the easiest possible learning task.
* **Data:** `nn_rankgauss`.
* **Architecture:** `WideScaledMLP` (2 Layers), StandardScaled Targets.
* **Result:**
    * **Val MAE:** Exploded from 0.51 (Epoch 5) to 0.86 (Epoch 35).
    * **Correlation:** 0.2967.
* **Verdict:** **Divergence.** The model started memorizing noise, causing the validation error to rise (overfitting/hallucination).

---

## 🛑 Final Decision
**Abandon Neural Networks.**

1.  **Performance Gap:** The best NN achieved ~0.009 MAE, while the baseline Trees achieve ~0.003 MAE. Blending a 3x worse model will dilute the score.
2.  **Data Nature:** This dataset appears to be low-signal/high-noise tabular data, which is the specific domain where Gradient Boosted Trees (XGB/LGB/Cat) massively outperform standard Neural Networks.
3.  **Strategy:** Submit the **Titanium Blend** (Trees Only).
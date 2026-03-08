# Extracted from notebooks/solution.py
# Included as baseline ML proof artifact for report appendix

# ===== BASELINE FUNCTION: train_champion_refinery =====
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



# ===== BASELINE FUNCTION: train_purist =====
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



# ===== BASELINE FUNCTION: train_robust =====
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



# ===== BASELINE FUNCTION: blend_refinery_and_robust =====
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




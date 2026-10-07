"""External reader vocabulary; persisted artifact keys remain unchanged."""

PROCEDURES = {
    "joint": "Complete selected procedure",
    "values_only": "Measurements only",
    "values_and_indicators": "Measurements + missing flags",
    "missingness_only": "Missing flags only",
    "all_pass": "Always predict pass",
    "strict": "Measurements only",
    "with_missing_indicators": "Measurements + missing flags",
    "logreg_final_primary": "Final primary logistic regression",
    "logreg_final_challenger": "Final comparison logistic regression",
}
CLASSIFIERS = {"krr": "Kernel ridge", "logreg": "Logistic regression"}

TABLE_HEADERS = {
    "selector": "Selection method",
    "classifier": "Model",
    "replication_mode": "Inputs",
    "mode": "Inputs",
    "procedure": "Procedure",
    "role": "Model role",
    "fold_index": "Test period",
    "eval_scope": "Evaluation samples",
    "evaluation_region": "Score source",
    "threshold_policy": "Threshold rule",
    "BER": "Balanced error",
    "mean_BER": "Mean balanced error",
    "std_BER": "Fold SD",
    "min_BER": "Minimum error",
    "max_BER": "Maximum error",
    "True+": "Failure recall",
    "True-": "Pass specificity",
    "TPR": "Failure recall",
    "TNR": "Pass specificity",
    "mean_True+": "Mean failure recall",
    "mean_True-": "Mean pass specificity",
    "ROC_AUC": "Ranking AUC",
    "PR_AUC": "Precision-recall AUC",
    "TP": "Failures caught",
    "TN": "Passes unflagged",
    "FP": "False alerts",
    "FN": "Failures missed",
    "pooled_TP": "Failures caught",
    "pooled_TN": "Passes unflagged",
    "pooled_FP": "False alerts",
    "pooled_FN": "Failures missed",
    "C": "LR inverse regularization",
    "alpha": "KRR regularization",
    "gamma": "RBF gamma",
    "gamma_multiplier": "Gamma multiplier",
    "gamma_values": "RBF gamma settings",
    "gamma_multiplier_values": "Gamma multiplier settings",
    "n_neighbors": "ReliefF neighbors",
    "k": "Selected inputs",
    "k_values": "Feature-budget count",
    "evaluated_configs": "Distinct configurations",
    "calibration_n": "Calibration samples",
    "calibration_fails": "Calibration failures",
    "calibration_passes": "Calibration passes",
    "BER_step_failure": "Error step per failure",
    "BER_step_pass": "Error step per pass",
    "lofo_available": "Recalibration available",
    "lofo_threshold_min": "Recalibrated threshold min",
    "lofo_threshold_max": "Recalibrated threshold max",
    "lofo_flagged_fraction_min": "Recalibrated flagged fraction min",
    "lofo_flagged_fraction_max": "Recalibrated flagged fraction max",
    "TNR90_semantics": "90% specificity interpretation",
    "observed_mean_inter_alarm_spacing": "Observed samples between alarms",
    "calibration_selected_MSPC_TPR_at_TNR90": "Evaluation recall at 90% specificity (calibration-selected MSPC score)",
    "primary_scientific": "Primary: balanced error",
    "primary_operational": "Primary: workload limited",
    "challenger_scientific": "Comparison: balanced error",
    "challenger_operational": "Comparison: workload limited",
    "all_pass_baseline": "Always predict pass",
    "all_flag_baseline": "Flag every sample",
    "modal_k": "Modal selected inputs",
    "modal_scaler": "Modal scaler",
    "confirmatory_claims_allowed": "Confirmatory claims allowed",
    "abs_prevalence_shift": "Absolute prevalence shift",
    "ks_pvalue_scores": "Score distribution p-value",
    "max_PSI": "Maximum measurement PSI",
    "median_PSI": "Median measurement PSI",
}


def table_header(column: str) -> str:
    """Translate display headings without changing stored artifact fields."""
    readable = column.replace("_", " ")
    return TABLE_HEADERS.get(column, readable[:1].upper() + readable[1:])


def table_value(column: str, value: object) -> object:
    """Translate known categorical fields; retain numeric values and unknown identifiers."""
    if column in {"mode", "replication_mode", "procedure"}:
        return procedure_label(value)
    if column == "classifier":
        return CLASSIFIERS.get(str(value), value)
    if column in {"role", "model_scope"}:
        return {"primary": "Primary model", "challenger": "Comparison model"}.get(str(value), value)
    if column == "threshold_policy":
        return {"scientific": "Balanced error", "operational": "Workload limited"}.get(str(value), value)
    if column in {
        "fold",
        "fold_index",
        "eval_scope",
        "evaluation_region",
        "source_selection_region",
        "TNR90_semantics",
        "workload_semantics",
        "score_reference",
        "missingness_reference",
        "drift_gate_status",
    }:
        return {
            "LOCKBOX": "Final retrospective block",
            "lockbox": "Final retrospective block",
            "outer_fold": "Chronological test period",
            "held_out_DEV_calibration": "Held-out calibration",
            "held_out_calibration": "Held-out calibration",
            "retrospective_evaluation_ROC_diagnostic": "Retrospective ranking diagnostic",
            "illustrative_mean_weekly_policy_not_per_week_cap": "Illustrative mean weekly rule",
            "held_out_calibration_same_retained_model": "Held-out calibration scores from the same retained model",
            "FIT_original_column_masks": "Original-column missingness masks in earlier fitting samples",
            "HIGH_SHIFT": "Large descriptive shift",
            "PASS": "Within heuristic shift limits",
        }.get(str(value), value)
    return value


def fold_count_label(frame) -> str:
    """Describe recorded fold counts without treating CLI defaults as observations."""
    if frame is None or frame.empty:
        return "unavailable"
    if "n_folds" in frame:
        counts = sorted({int(value) for value in frame.n_folds.dropna()})
        return ", ".join(map(str, counts)) if counts else "unavailable"
    if "fold" in frame:
        return str(frame.fold.nunique())
    return "unavailable"


def later_sample_scope(frame) -> str:
    """Derive common later evaluation denominators from frozen confusion counts."""
    if frame is None or frame.empty or not {"TP", "FN", "FP", "TN"} <= set(frame):
        return "Later-block sample counts are unavailable."
    totals = frame[["TP", "FN", "FP", "TN"]].dropna()
    failures = (totals.TP + totals.FN).unique()
    passes = (totals.FP + totals.TN).unique()
    if len(failures) != 1 or len(passes) != 1:
        return "Evaluation totals vary across the recorded rules; read each denominator separately."
    return f"The later block contains {int(failures[0])} failures and {int(passes[0])} passes."


def procedure_label(value: object) -> str:
    """Describe a predefined procedure, including chronological window variants."""
    name = str(value)
    if name.startswith("krr_"):
        fraction = "30%" if "30" in name else "20%"
        mode = "joint" if "joint" in name else "values_only" if "values_only" in name else "values_and_indicators"
        return f"{PROCEDURES[mode]} ({fraction} calibration)"
    return PROCEDURES.get(name, name)


def family_label(row: object) -> str:
    """Describe a selector/model/input family without internal input-mode codes."""
    return f"{row['selector']} / {CLASSIFIERS.get(row['classifier'], row['classifier'])} / {procedure_label(row['replication_mode'])}"


def role_label(role: object, policy: object) -> str:
    """Translate existing LR role and threshold-policy names."""
    model = "Primary model" if role == "primary" else "Comparison model"
    threshold = "balanced-error threshold" if policy == "scientific" else "workload-limited threshold"
    return f"{model}: {threshold}"

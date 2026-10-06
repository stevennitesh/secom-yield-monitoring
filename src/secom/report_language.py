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

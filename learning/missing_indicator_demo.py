import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ber_tpr_tnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-12)  # True+ (fail recall)
    tnr = tn / (tn + fp + 1e-12)  # True- (pass specificity)
    ber = 1.0 - 0.5 * (tpr + tnr)
    return ber, tpr, tnr


def make_data(n=2500, fail_rate=0.12, seed=42):
    rng = np.random.default_rng(seed)

    # Labels: 1=fail, 0=pass
    y = (rng.random(n) < fail_rate).astype(int)

    # Feature A: weak value signal
    x_val = rng.normal(loc=0.0 + 0.4 * y, scale=1.0, size=n)

    # Feature B: mostly noise in value, but missingness itself is informative
    x_miss = rng.normal(0, 1, size=n)
    p_missing = np.where(y == 1, 0.65, 0.08)  # fail rows missing much more often
    miss_mask = rng.random(n) < p_missing
    x_miss[miss_mask] = np.nan

    # Extra noise features
    noise = rng.normal(size=(n, 6))

    X = np.column_stack([x_val, x_miss, noise])
    names = ["x_val_weak", "x_miss_informative"] + [f"noise_{i}" for i in range(1, 7)]
    return X, y, names, miss_mask


def evaluate_median_only(X_train, y_train, X_test, y_test):
    # No missing indicators
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ),
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    ber, tpr, tnr = ber_tpr_tnr(y_test, pred)
    return ber, tpr, tnr


def evaluate_median_plus_indicator(X_train, y_train, X_test, y_test):
    # Add missing indicators automatically
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ),
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    ber, tpr, tnr = ber_tpr_tnr(y_test, pred)
    return ber, tpr, tnr


if __name__ == "__main__":
    X, y, names, miss_mask = make_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=11
    )

    b1, tpr1, tnr1 = evaluate_median_only(X_train, y_train, X_test, y_test)
    b2, tpr2, tnr2 = evaluate_median_plus_indicator(X_train, y_train, X_test, y_test)

    print("Missingness rates in full data:")
    print(f"  fail class (y=1): {miss_mask[y == 1].mean():.3f}")
    print(f"  pass class (y=0): {miss_mask[y == 0].mean():.3f}\n")

    print("Median-only:")
    print(f"  BER={b1:.3f}, True+={tpr1:.3f}, True-={tnr1:.3f}")

    print("Median + indicator:")
    print(f"  BER={b2:.3f}, True+={tpr2:.3f}, True-={tnr2:.3f}")

    # Visual summary
    labels = ["BER (lower better)", "True+ (higher better)", "True- (higher better)"]
    median_only = [b1, tpr1, tnr1]
    with_ind = [b2, tpr2, tnr2]

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("When Missing Indicators Help", fontsize=13)

    # Left: missingness by class
    miss_by_class = [miss_mask[y == 0].mean(), miss_mask[y == 1].mean()]
    axes[0].bar(["pass (0)", "fail (1)"], miss_by_class)
    axes[0].set_title("Feature x_miss missing rate by class")
    axes[0].set_ylabel("Missing fraction")

    # Right: model metrics
    axes[1].bar(x - w / 2, median_only, width=w, label="Median only")
    axes[1].bar(x + w / 2, with_ind, width=w, label="Median + indicator")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_title("Evaluation on held-out test split")
    axes[1].legend()

    fig.text(
        0.5,
        0.01,
        "If missingness differs by class, an indicator can carry predictive signal that median-only imputation loses.",
        ha="center",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.show()
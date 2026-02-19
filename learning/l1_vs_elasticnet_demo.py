import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)

    # Two latent factors (true signal)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)

    # Correlated feature groups
    x1 = z1 + 0.10 * rng.normal(size=n)
    x2 = 0.95 * z1 + 0.20 * rng.normal(size=n)
    x3 = -0.90 * z1 + 0.20 * rng.normal(size=n)

    x4 = z2 + 0.10 * rng.normal(size=n)
    x5 = 1.05 * z2 + 0.20 * rng.normal(size=n)

    # Pure noise features
    noise = rng.normal(size=(n, 8))

    X = np.column_stack([x1, x2, x3, x4, x5, noise])
    names = ["x1_z1", "x2_z1_dup", "x3_z1_inv", "x4_z2", "x5_z2_dup"] + [f"noise_{i}" for i in range(1, 9)]

    # Imbalanced fail label (~15%)
    score = 1.2 * z1 + 1.0 * z2 + 0.7 * rng.normal(size=n)
    y = (score > np.quantile(score, 0.85)).astype(int)

    return X, y, names


def model_l1(seed=0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="saga",
            C=0.15,
            l1_ratio=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )


def model_en(seed=0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="saga",
            C=0.15,
            l1_ratio=0.4,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )


def fit_abs_coef(pipe, X, y):
    pipe.fit(X, y)
    return np.abs(pipe[-1].coef_[0])


def selection_frequency(builder, X, y, repeats=30, tol=1e-6):
    freq = np.zeros(X.shape[1], dtype=float)
    for seed in range(repeats):
        X_tr, _, y_tr, _ = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        pipe = builder(seed)
        pipe.fit(X_tr, y_tr)
        selected = np.abs(pipe[-1].coef_[0]) > tol
        freq += selected.astype(float)
    return freq / repeats


if __name__ == "__main__":
    X, y, names = make_data()

    coef_l1 = fit_abs_coef(model_l1(0), X, y)
    coef_en = fit_abs_coef(model_en(0), X, y)

    freq_l1 = selection_frequency(model_l1, X, y, repeats=40)
    freq_en = selection_frequency(model_en, X, y, repeats=40)

    print(f"Fail rate: {y.mean():.3f}")
    print("Top by |coef| (L1):")
    for i in np.argsort(coef_l1)[::-1][:8]:
        print(f"  {names[i]:12s}  {coef_l1[i]:.4f}")

    print("\nTop by |coef| (Elastic Net):")
    for i in np.argsort(coef_en)[::-1][:8]:
        print(f"  {names[i]:12s}  {coef_en[i]:.4f}")

    x = np.arange(len(names))
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("L1 vs Elastic Net (Correlated Features)", fontsize=13)

    axes[0].bar(x - w / 2, coef_l1, width=w, label="L1 |coef|")
    axes[0].bar(x + w / 2, coef_en, width=w, label="EN |coef|")
    axes[0].set_title("Full-data absolute coefficients")
    axes[0].set_xticks(x, names, rotation=30, ha="right")
    axes[0].legend()

    axes[1].bar(names, freq_l1)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Selection frequency (L1)")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(names, freq_en)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Selection frequency (Elastic Net)")
    axes[2].tick_params(axis="x", rotation=30)

    fig.text(
        0.5,
        0.01,
        "L1 often picks one feature from a correlated group; Elastic Net tends to share weight across correlated features.",
        ha="center",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.show()

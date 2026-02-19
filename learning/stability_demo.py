import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_data(n=1800, seed=42):
    rng = np.random.default_rng(seed)

    # Latent factors
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    z3 = rng.normal(size=n)

    # Correlated feature groups (redundant sensors)
    g1 = np.column_stack([z1 + 0.10 * rng.normal(size=n), z1 + 0.15 * rng.normal(size=n), z1 + 0.20 * rng.normal(size=n)])
    g2 = np.column_stack([z2 + 0.10 * rng.normal(size=n), z2 + 0.15 * rng.normal(size=n), z2 + 0.20 * rng.normal(size=n)])
    g3 = np.column_stack([z3 + 0.10 * rng.normal(size=n), z3 + 0.15 * rng.normal(size=n), z3 + 0.20 * rng.normal(size=n)])

    noise = rng.normal(size=(n, 15))
    X = np.hstack([g1, g2, g3, noise])

    names = (
        ["g1_a", "g1_b", "g1_c", "g2_a", "g2_b", "g2_c", "g3_a", "g3_b", "g3_c"]
        + [f"noise_{i}" for i in range(1, 16)]
    )

    # Imbalanced fail label (~12%)
    score = 1.1 * z1 - 1.0 * z2 + 0.9 * z3 + 0.8 * rng.normal(size=n)
    y = (score > np.quantile(score, 0.88)).astype(int)  # 1=fail, 0=pass

    return X, y, names


def make_model(kind, seed):
    if kind == "l1":
        l1_ratio = 1.0
    elif kind == "en":
        l1_ratio = 0.4
    else:
        raise ValueError("kind must be 'l1' or 'en'")

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="saga",
            C=0.15,
            l1_ratio=l1_ratio,
            class_weight="balanced",
            max_iter=6000,
            random_state=seed,
        ),
    )


def ber(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-12)  # True+ (fail recall)
    tnr = tn / (tn + fp + 1e-12)  # True- (pass specificity)
    return 1.0 - 0.5 * (tpr + tnr)


def topk_set_from_coef(pipe, k):
    coef = np.abs(pipe[-1].coef_[0])
    idx = np.argsort(coef)[::-1][:k]
    return set(idx), coef


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def run_method(kind, X, y, repeats=40, top_k=8):
    n_features = X.shape[1]
    freq = np.zeros(n_features, dtype=float)
    selected_sets = []
    bers = []

    for seed in range(repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        model = make_model(kind, seed)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_te)
        bers.append(ber(y_te, pred))

        s, _ = topk_set_from_coef(model, top_k)
        selected_sets.append(s)
        for j in s:
            freq[j] += 1

    freq /= repeats

    pairwise_j = [
        jaccard(a, b)
        for a, b in combinations(selected_sets, 2)
    ]

    return {
        "freq": freq,
        "bers": np.array(bers),
        "pairwise_j": np.array(pairwise_j),
    }


if __name__ == "__main__":
    X, y, names = make_data()
    print(f"n={len(y)}, fail_rate={y.mean():.3f}, n_features={X.shape[1]}")

    res_l1 = run_method("l1", X, y, repeats=40, top_k=8)
    res_en = run_method("en", X, y, repeats=40, top_k=8)

    print("\nStability summary")
    print(f"L1 mean BER: {res_l1['bers'].mean():.4f} +/- {res_l1['bers'].std():.4f}")
    print(f"EN mean BER: {res_en['bers'].mean():.4f} +/- {res_en['bers'].std():.4f}")
    print(f"L1 mean pairwise Jaccard: {res_l1['pairwise_j'].mean():.4f}")
    print(f"EN mean pairwise Jaccard: {res_en['pairwise_j'].mean():.4f}")

    # Plot: top-15 most frequently selected by EN
    top_idx = np.argsort(res_en["freq"])[::-1][:15]
    top_names = [names[i] for i in top_idx]

    x = np.arange(len(top_idx))
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Feature Stability Demo: L1 vs Elastic Net", fontsize=13)

    axes[0].bar(x - w / 2, res_l1["freq"][top_idx], width=w, label="L1")
    axes[0].bar(x + w / 2, res_en["freq"][top_idx], width=w, label="Elastic Net")
    axes[0].set_title("Selection Frequency (Top 15 by EN)")
    axes[0].set_xticks(x, top_names, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].hist(res_l1["pairwise_j"], bins=20, alpha=0.65, label="L1")
    axes[1].hist(res_en["pairwise_j"], bins=20, alpha=0.65, label="Elastic Net")
    axes[1].set_title("Pairwise Jaccard Across Repeats")
    axes[1].set_xlabel("Jaccard")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    axes[2].hist(res_l1["bers"], bins=12, alpha=0.65, label="L1")
    axes[2].hist(res_en["bers"], bins=12, alpha=0.65, label="Elastic Net")
    axes[2].set_title("BER Distribution Across Repeats")
    axes[2].set_xlabel("BER (lower is better)")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    fig.text(
        0.5,
        0.01,
        "Higher Jaccard means more stable feature sets across seeds/splits.",
        ha="center",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.show()
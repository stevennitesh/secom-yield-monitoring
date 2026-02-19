import numpy as np
import matplotlib.pyplot as plt


def make_xor_data(n_per_cluster=120, noise=0.25, seed=42):
    rng = np.random.default_rng(seed)

    # Class 0 clusters: (-1,-1), (+1,+1)
    c00 = rng.normal(loc=[-1.0, -1.0], scale=noise, size=(n_per_cluster, 2))
    c11 = rng.normal(loc=[+1.0, +1.0], scale=noise, size=(n_per_cluster, 2))

    # Class 1 clusters: (-1,+1), (+1,-1)
    c01 = rng.normal(loc=[-1.0, +1.0], scale=noise, size=(n_per_cluster, 2))
    c10 = rng.normal(loc=[+1.0, -1.0], scale=noise, size=(n_per_cluster, 2))

    X2 = np.vstack([c00, c11, c01, c10])
    y = np.array([0] * (2 * n_per_cluster) + [1] * (2 * n_per_cluster))

    # Add one pure noise feature
    noise_col = rng.normal(0.0, 1.0, size=(X2.shape[0], 1))
    X = np.hstack([X2, noise_col])

    feature_names = ["x1", "x2", "noise"]
    return X, y, feature_names


def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def relieff_binary(X, y, k=10, n_samples=None, seed=0):
    """
    Simplified ReliefF for binary classes.
    Higher score => feature better separates nearby opposite-class points.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    indices = np.arange(n)
    if n_samples is not None and n_samples < n:
        indices = rng.choice(indices, size=n_samples, replace=False)

    w = np.zeros(p, dtype=float)

    for i in indices:
        xi = X[i]
        d = np.linalg.norm(X - xi, axis=1)

        same = np.where(y == y[i])[0]
        same = same[same != i]
        diff = np.where(y != y[i])[0]

        k_hit = min(k, len(same))
        k_miss = min(k, len(diff))
        if k_hit == 0 or k_miss == 0:
            continue

        hit_idx = same[np.argpartition(d[same], k_hit - 1)[:k_hit]]
        miss_idx = diff[np.argpartition(d[diff], k_miss - 1)[:k_miss]]

        w += np.mean(np.abs(xi - X[miss_idx]), axis=0)
        w -= np.mean(np.abs(xi - X[hit_idx]), axis=0)

    return w / len(indices)


def pearson_abs_scores(X, y):
    scores = []
    for j in range(X.shape[1]):
        r = np.corrcoef(X[:, j], y)[0, 1]
        if np.isnan(r):
            r = 0.0
        scores.append(abs(r))
    return np.array(scores)


if __name__ == "__main__":
    X, y, names = make_xor_data()
    Xz = zscore(X)

    p_scores = pearson_abs_scores(X, y)
    r_scores = relieff_binary(Xz, y, k=15, n_samples=None, seed=1)

    # Print rankings
    p_order = np.argsort(p_scores)[::-1]
    r_order = np.argsort(r_scores)[::-1]

    print("Pearson |r| ranking:")
    for i in p_order:
        print(f"  {names[i]:>5s}: {p_scores[i]:.4f}")

    print("\nReliefF ranking:")
    for i in r_order:
        print(f"  {names[i]:>5s}: {r_scores[i]:.4f}")

    # Visual popup
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Pearson vs ReliefF on XOR-like Data", fontsize=13)

    sc = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.65, s=18)
    axes[0].set_title("Data layout (x1, x2)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    legend1 = axes[0].legend(*sc.legend_elements(), title="Class")
    axes[0].add_artist(legend1)

    axes[1].bar(names, p_scores)
    axes[1].set_title("Pearson |r|")
    axes[1].set_ylabel("Higher = stronger linear relation")

    axes[2].bar(names, r_scores)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title("ReliefF score")
    axes[2].set_ylabel("Higher = better local class separation")

    fig.text(
        0.5,
        0.01,
        "XOR pattern: single-feature linear correlation is weak, but ReliefF can still score x1/x2 as useful.",
        ha="center",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.show()
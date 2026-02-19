import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(12)
n = 700

# Two latent factors drive the label
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)

# Imbalanced binary label (~15% fail)
signal = 1.2 * z1 + 1.0 * z2 + 0.5 * rng.normal(size=n)
y = (signal > np.quantile(signal, 0.85)).astype(float)  # 0=pass, 1=fail

# Features:
# f1 and f2 are redundant (both from z1), f3 is complementary (z2), f4 is noise
X = np.column_stack(
    [
        z1 + 0.15 * rng.normal(size=n),  # f1
        z1 + 0.15 * rng.normal(size=n),  # f2 (duplicate-ish)
        z2 + 0.15 * rng.normal(size=n),  # f3 (new info)
        rng.normal(size=n),  # f4 noise
    ]
)
names = ["f1_z1", "f2_z1_dup", "f3_z2", "f4_noise"]

# Standardize
X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)


def abs_corr_scores(X_, y_):
    yc = y_ - y_.mean()
    yn = np.linalg.norm(yc)
    scores = []
    for j in range(X_.shape[1]):
        xj = X_[:, j] - X_[:, j].mean()
        s = abs(np.dot(xj, yc)) / (np.linalg.norm(xj) * yn + 1e-12)
        scores.append(s)
    return np.array(scores)


def gram_schmidt_rank(X_, y_, k=3):
    Xw = X_.copy()
    r = y_ - y_.mean()  # residual target direction
    remaining = list(range(Xw.shape[1]))
    selected = []
    step_scores = []

    for _ in range(min(k, Xw.shape[1])):
        # Score remaining features against current residual.
        scores = []
        for j in remaining:
            xj = Xw[:, j]
            s = abs(np.dot(xj, r)) / (np.linalg.norm(xj) * np.linalg.norm(r) + 1e-12)
            scores.append(s)

        best_local = int(np.argmax(scores))
        best_j = remaining[best_local]
        selected.append(best_j)
        step_scores.append(scores[best_local])

        # Orthogonal direction of selected feature.
        q = Xw[:, best_j]
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            break
        q = q / qn

        # Remove selected direction from remaining features and target residual.
        for j in remaining:
            if j == best_j:
                continue
            Xw[:, j] = Xw[:, j] - np.dot(Xw[:, j], q) * q
        r = r - np.dot(r, q) * q

        remaining.remove(best_j)

    return selected, step_scores


uni_scores = abs_corr_scores(X, y)
uni_order = np.argsort(uni_scores)[::-1]
gs_sel, gs_step_scores = gram_schmidt_rank(X, y, k=3)

print("Univariate |corr| ranking:")
print("  " + " > ".join(names[i] for i in uni_order))
print("Gram-Schmidt selection order:")
print("  " + " > ".join(names[i] for i in gs_sel))

# For plotting GS contribution by selected order
gs_contrib = np.zeros(len(names))
for j, s in zip(gs_sel, gs_step_scores):
    gs_contrib[j] = s

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Gram-Schmidt Feature Selection Demo", fontsize=13)

# Redundancy view: f1 vs f2
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.65, s=14)
axes[0].set_title("f1 vs f2 (redundant pair)")
axes[0].set_xlabel("f1_z1")
axes[0].set_ylabel("f2_z1_dup")

# Univariate ranking
axes[1].bar(names, uni_scores)
axes[1].set_title("Univariate |corr| scores")
axes[1].tick_params(axis="x", rotation=20)

# Gram-Schmidt step contributions
axes[2].bar(names, gs_contrib)
axes[2].set_title("Gram-Schmidt incremental scores")
axes[2].tick_params(axis="x", rotation=20)

fig.text(
    0.5,
    0.01,
    "Univariate often ranks both duplicates high. Gram-Schmidt tends to pick one duplicate, then a complementary feature.",
    ha="center",
)
plt.tight_layout(rect=[0, 0.06, 1, 0.93])
plt.show()

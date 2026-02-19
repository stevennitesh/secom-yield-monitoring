import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# Imbalanced labels like SECOM: 0=pass, 1=fail
n_pass, n_fail = 800, 60
y = np.r_[np.zeros(n_pass), np.ones(n_fail)]
n = y.size

# 1) Strong linear signal (Pearson should be high)
x_linear = np.r_[rng.normal(0.0, 1.0, n_pass), rng.normal(1.0, 1.0, n_fail)]

# 2) Inverse linear signal (negative Pearson)
x_inverse = np.r_[rng.normal(1.0, 1.0, n_pass), rng.normal(0.0, 1.0, n_fail)]

# 3) Nonlinear but class-separating pattern (Pearson can be near 0)
# pass around -2 and +2, fail around 0
x_nonlinear = np.r_[
    rng.choice([-2.0, 2.0], size=n_pass) + rng.normal(0.0, 0.35, n_pass),
    rng.normal(0.0, 0.35, n_fail),
]

# 4) Mostly noise + one extreme fail outlier (can distort Pearson)
x_outlier = rng.normal(0.0, 1.0, n)
x_outlier[n_pass] = 10.0  # one fail point made extreme

features = {
    "linear": x_linear,
    "inverse": x_inverse,
    "nonlinear": x_nonlinear,
    "outlier": x_outlier,
}


def pearson_r(x: np.ndarray, y_: np.ndarray) -> float:
    return float(np.corrcoef(x, y_)[0, 1])


def t_from_r(r: float, n_samples: int) -> float:
    # Binary-label Pearson is point-biserial; linked to t-stat under standard assumptions
    denom = max(1e-12, 1.0 - r * r)
    return r * np.sqrt((n_samples - 2) / denom)


scores = []
for name, x in features.items():
    r = pearson_r(x, y)
    t = t_from_r(r, n)
    scores.append((name, r, t))

scores_sorted = sorted(scores, key=lambda z: abs(z[1]), reverse=True)

print("Feature scores (ranked by |r|)")
for name, r, t in scores_sorted:
    print(f"{name:10s}  r={r:+.3f}   |r|={abs(r):.3f}   implied_t={t:+.3f}")

# For plotting convenience
r_map = {name: r for name, r, _ in scores}
names = [name for name, _, _ in scores]
rvals = [r_map[name] for name in names]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Pearson Correlation Demo (Binary fail/pass labels)", fontsize=13)

# Panel 1: Pearson scores
axes[0].bar(names, rvals)
axes[0].axhline(0, color="black", lw=1)
axes[0].set_title("Pearson r by feature")
axes[0].set_ylabel("r")
axes[0].tick_params(axis="x", rotation=20)
for i, v in enumerate(rvals):
    axes[0].text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top")

# Panel 2: linear feature distributions by class
axes[1].hist(x_linear[:n_pass], bins=30, alpha=0.6, label="pass (0)")
axes[1].hist(x_linear[n_pass:], bins=30, alpha=0.6, label="fail (1)")
axes[1].set_title(f"Linear feature (r={r_map['linear']:+.2f})")
axes[1].legend()

# Panel 3: nonlinear feature distributions by class
axes[2].hist(x_nonlinear[:n_pass], bins=30, alpha=0.6, label="pass (0)")
axes[2].hist(x_nonlinear[n_pass:], bins=30, alpha=0.6, label="fail (1)")
axes[2].set_title(f"Nonlinear feature (r={r_map['nonlinear']:+.2f})")
axes[2].legend()

fig.text(
    0.5,
    0.01,
    "Pearson is great for linear class shifts. It can miss nonlinear separation and can be sensitive to outliers.",
    ha="center",
)
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.show()
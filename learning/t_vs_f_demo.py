import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway

# Reproducible toy data (binary classes: pass vs fail)
rng = np.random.default_rng(42)
n_pass, n_fail = 800, 60

X_pass = {
    "f1_clean_shift": rng.normal(0.0, 1.0, n_pass),
    "f2_var_imbalance": rng.normal(0.0, 1.0, n_pass),
    "f3_small_shift": rng.normal(0.0, 1.0, n_pass),
}
X_fail = {
    "f1_clean_shift": rng.normal(0.8, 1.0, n_fail),
    "f2_var_imbalance": rng.normal(1.2, 5.0, n_fail),  # huge fail variance
    "f3_small_shift": rng.normal(0.5, 1.0, n_fail),
}

names = list(X_pass.keys())
t_equal_vals, t_welch_vals, f_vals = [], [], []

print("Feature-wise stats")
print("-" * 86)
print(f"{'Feature':20s} {'|t_equal|':>10s} {'F':>10s} {'|t_equal|^2':>12s} {'|t_welch|':>10s}")
print("-" * 86)

for name in names:
    a = X_fail[name]
    b = X_pass[name]

    t_eq = ttest_ind(a, b, equal_var=True).statistic
    t_w = ttest_ind(a, b, equal_var=False).statistic
    f = f_oneway(a, b).statistic

    t_equal_vals.append(abs(t_eq))
    t_welch_vals.append(abs(t_w))
    f_vals.append(f)

    print(f"{name:20s} {abs(t_eq):10.3f} {f:10.3f} {abs(t_eq)**2:12.3f} {abs(t_w):10.3f}")

t_equal_vals = np.array(t_equal_vals)
t_welch_vals = np.array(t_welch_vals)
f_vals = np.array(f_vals)

print("-" * 86)
print(f"max(|F - |t_equal|^2|) = {np.max(np.abs(f_vals - t_equal_vals**2)):.6e}")

def rank_desc(feature_names, scores):
    idx = np.argsort(scores)[::-1]
    return " > ".join(feature_names[i] for i in idx)

print("\nRanking by |t_equal| :", rank_desc(names, t_equal_vals))
print("Ranking by F         :", rank_desc(names, f_vals))
print("Ranking by |t_welch| :", rank_desc(names, t_welch_vals))

# Visual popup
x = np.arange(len(names))
w = 0.36

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("t-test vs F-test (Binary Classes)", fontsize=13)

axes[0].bar(x - w / 2, f_vals, width=w, label="F (ANOVA)")
axes[0].bar(x + w / 2, t_equal_vals**2, width=w, label="|t_equal|^2")
axes[0].set_title("For 2 classes: F == t_equal^2")
axes[0].set_xticks(x, names, rotation=20, ha="right")
axes[0].legend()

axes[1].bar(x - w / 2, t_equal_vals, width=w, label="|t_equal|")
axes[1].bar(x + w / 2, t_welch_vals, width=w, label="|t_welch|")
axes[1].set_title("Welch can diverge under variance imbalance")
axes[1].set_xticks(x, names, rotation=20, ha="right")
axes[1].legend()

fig.text(
    0.5,
    0.01,
    "Equal-variance t-test and ANOVA F give the same ranking for binary labels. "
    "Welch t-test can reorder features when class variances differ.",
    ha="center",
)
plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.show()
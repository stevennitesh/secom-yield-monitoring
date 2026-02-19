import math
import matplotlib.pyplot as plt

# Toy data from our example
# delta_mu = |mean_fail - mean_pass|
data = {
    "A": {"n_fail": 10, "n_pass": 100, "delta_mu": 0.8, "sigma_fail": 1.0, "sigma_pass": 1.0},
    "B": {"n_fail": 2,  "n_pass": 100, "delta_mu": 1.2, "sigma_fail": 1.0, "sigma_pass": 1.0},
    "C": {"n_fail": 10, "n_pass": 100, "delta_mu": 0.6, "sigma_fail": 0.2, "sigma_pass": 0.2},
}

EPS = 1e-12

def s2n(d):
    return abs(d["delta_mu"]) / (d["sigma_fail"] + d["sigma_pass"] + EPS)

def t_abs(d):
    se = math.sqrt((d["sigma_fail"] ** 2) / d["n_fail"] + (d["sigma_pass"] ** 2) / d["n_pass"] + EPS)
    return abs(d["delta_mu"]) / se

# Compute scores
features = list(data.keys())
s2n_vals = [s2n(data[f]) for f in features]
t_vals = [t_abs(data[f]) for f in features]

# Rankings
rank_s2n = sorted(features, key=lambda f: s2n(data[f]), reverse=True)
rank_t = sorted(features, key=lambda f: t_abs(data[f]), reverse=True)

# Console output (optional)
print("Feature scores")
for f in features:
    print(f"{f}: S2N={s2n(data[f]):.3f}, |t|={t_abs(data[f]):.3f}")
print("S2N rank:", " > ".join(rank_s2n))
print("|t| rank:", " > ".join(rank_t))

# Plot popup
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("S2N vs t-test (|t|) on Toy SECOM-like Data", fontsize=13)

axes[0].bar(features, s2n_vals)
axes[0].set_title("S2N score")
axes[0].set_ylabel("Higher = stronger separation")
for i, v in enumerate(s2n_vals):
    axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

axes[1].bar(features, t_vals)
axes[1].set_title("|t| score (Welch-style denominator)")
axes[1].set_ylabel("Higher = stronger + more certain")
for i, v in enumerate(t_vals):
    axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

explain = (
    f"S2N rank: {' > '.join(rank_s2n)}\n"
    f"|t| rank: {' > '.join(rank_t)}\n\n"
    "Why A/B can flip:\n"
    "B has a bigger mean gap, so S2N likes it.\n"
    "But B has only n_fail=2, so uncertainty is high; |t| penalizes that."
)
fig.text(0.5, 0.01, explain, ha="center", va="bottom", fontsize=10)

plt.tight_layout(rect=[0, 0.15, 1, 0.92])
plt.show()
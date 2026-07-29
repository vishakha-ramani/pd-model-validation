"""Paper figure: Mode-B teacher-forced per-iteration law validation.

Reads the real captured trajectory and frozen coeffs vendored in this repository,
feeds the REAL per-step composition into the latency law, and plots
predicted vs measured iteration time per regime on log-log axes.

Output: figures/titer_modeb.png
"""
import gzip, json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repository root
COEFFS = json.load(open(os.path.join(ROOT, "coeffs.json")))
TRAJ = os.path.join(ROOT, "calibration", "results", "modeb", "trajectory.jsonl.gz")
PROBE_ROWS = 250
GAP = 0.002  # back-to-back threshold (s)


def is_prefill(r):
    return r["computed"] < r["prompt_len"]


def predict(reqs, c):
    B = 0; res = 0.0; pf = 0.0
    for r in reqs:
        if is_prefill(r):
            k = r["kappa"]; pk = r["computed"]
            pf += c["c_pf"] * k + c["c_attn"] * k * (pk + k / 2.0)
        else:
            B += 1; res += r["computed"]
    return c["c_base"] + c["c_dec"] * B + c["c_kv"] * res + pf


def regime(reqs):
    hp = any(is_prefill(r) for r in reqs)
    hd = any(not is_prefill(r) for r in reqs)
    if hp and hd: return "mixed"
    if hp: return "pure_prefill"
    return "pure_decode"


rows = []
with gzip.open(TRAJ, "rt") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
rows.sort(key=lambda s: s["step"])

recs = []
for i in range(len(rows) - 1):
    s = rows[i]; nxt = rows[i + 1]
    if s["step"] < PROBE_ROWS or not s["reqs"]:
        continue
    gap = nxt["t_start"] - s["t_end"]
    if gap > GAP:
        continue
    recs.append((s["reqs"], nxt["t_start"] - s["t_start"], regime(s["reqs"])))

# per-regime MAPE on the full set
def mape(sub):
    e = [abs(predict(rq, COEFFS) - m) / m for rq, m, _ in sub]
    return 100.0 * sum(e) / len(e)

labels = {"pure_decode": "decode", "pure_prefill": "prefill", "mixed": "mixed"}
colors = {"pure_decode": "#1f77b4", "pure_prefill": "#2ca02c", "mixed": "#d62728"}
stats = {name: (sum(1 for r in recs if r[2] == name),
                mape([r for r in recs if r[2] == name]))
         for name in labels}
all_mape = mape(recs)
print("n", len(recs), "all MAPE", round(all_mape, 2))
for k, v in stats.items():
    print(k, v)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# deterministic subsample per regime for rendering (stats already use all points)
def sample(sub, n):
    if len(sub) <= n:
        return sub
    step = len(sub) / n
    return [sub[int(j * step)] for j in range(n)]

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(3.4, 3.1))
for name in ("pure_decode", "mixed", "pure_prefill"):
    sub = sample([r for r in recs if r[2] == name], 4000)
    xs = [m * 1000 for _, m, _ in sub]
    ys = [predict(rq, COEFFS) * 1000 for rq, _, _ in sub]
    n, mp = stats[name]
    ax.scatter(xs, ys, s=3, alpha=0.25, color=colors[name], edgecolors="none",
               label=f"{labels[name]}  MAPE {mp:.1f}\\%")
lo, hi = 10, 800
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("measured iteration time (ms)")
ax.set_ylabel("predicted iteration time (ms)")
leg = ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderpad=0.1,
                fontsize=8, markerscale=2)
ax.set_aspect("equal")
fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "titer_modeb.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
print("wrote", out)

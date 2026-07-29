"""Paper figure: real-engine parity for the full composed first-token estimate.

Reconstructs, per request, the end-to-end first-token time the router scores,

    TTFT = Tadm + n_c * Titer(batch) + Wp ,

and compares it against the realized first-token time measured from the
admission load-sweep capture (~31k requests, Llama-3.3-70B on H100 TP4,
vLLM 0.11.0, collocated single instance). Realized TTFT = t_first - t_enq,
where t_enq is enqueue and t_first is the first step at which the request's
computed count reaches its prompt length.

Two admission variants share the same prefill composition:
  oracle   - substitutes each request's realized admission delay Tadm.
             Isolates the composition (n_c*Titer + Wp): points sit on y=x.
  deploy   - the roll-forward admission estimator the router runs online
             (censored output-length mean, no queue snapshot). Tracks the
             un-queued bulk, under-predicts the queued tail by ~1 order.

Rows are reconstructed by figures/ttft_rows.json (produced in-cluster from
the admission capture PVC; see calibration/admission/RESULTS.md). Times in s.

Output: figures/ttft_parity.png
"""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = json.load(open(os.path.join(HERE, "ttft_rows.json")))


def stats(pairs):
    ratios = [p / r for r, p in pairs if r > 0 and p > 0]
    errs = [abs(p - r) / r for r, p in pairs if r > 0]
    mr = statistics.median(ratios)
    return len(pairs), (mr - 1) * 100, 100 * sum(errs) / len(errs)


oracle = [(row["r_ttft"], row["p_oracle"]) for row in ROWS]
deploy = [(row["r_ttft"], row["p_deploy"]) for row in ROWS]
# queued split by realized admission delay
qd = [(row["r_ttft"], row["p_deploy"]) for row in ROWS if row["r_tadm"] > 0.5]
nq = [(row["r_ttft"], row["p_deploy"]) for row in ROWS if row["r_tadm"] <= 0.5]

print("n", len(ROWS))
print("oracle       ", stats(oracle))
print("deploy all   ", stats(deploy))
print("deploy nqueue", stats(nq))
print("deploy queued", stats(qd))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample(sub, n):
    if len(sub) <= n:
        return sub
    step = len(sub) / n
    return [sub[int(j * step)] for j in range(n)]


# seconds -> ms
def ms(pairs):
    return [(r * 1000.0, p * 1000.0) for r, p in pairs]


plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(3.4, 3.1))

_, o_bias, o_mape = stats(oracle)
_, d_bias, d_mape = stats(deploy)

for pairs, color, label in [
    (nq, "#d62728", None),
    (qd, "#d62728", f"deployable  MAPE {d_mape:.0f}%"),
]:
    draw = sample(ms(pairs), 3500)
    ax.scatter([r for r, _ in draw], [p for _, p in draw],
               s=3, alpha=0.20, color=color, edgecolors="none", label=label)
draw = sample(ms(oracle), 3500)
ax.scatter([r for r, _ in draw], [p for _, p in draw],
           s=3, alpha=0.25, color="#1f77b4", edgecolors="none",
           label=f"oracle admission  MAPE {o_mape:.0f}%")

lo, hi = 8, 60000
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("measured TTFT (ms)")
ax.set_ylabel("predicted TTFT (ms)")
ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderpad=0.1,
          fontsize=7.5, markerscale=2.2)
ax.set_aspect("equal")
fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "ttft_parity.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
print("wrote", out)

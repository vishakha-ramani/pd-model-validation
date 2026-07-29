"""Byte-faithful Python ports of inference-sim/sim/admission_estimator.go.

Unit-agnostic. The admission harness runs them in seconds (coeffs.json units),
so t_iter and the returned T_adm are seconds.
"""
import math


def floored_t_adm(est, ctx):
    t_iter = ctx["t_iter"]
    if t_iter > est:
        return t_iter
    return est


def _slot_and_kv_fit(ctx):
    return ctx["batch_size"] < ctx["max_batch_size"] and ctx["free_kv_blocks"] >= ctx["req_kv_need"]


def estimate_waiting(ctx):
    mu = ctx.get("mu", 0.0)
    if mu <= 0:
        return 0.0
    return ctx.get("qwork", 0.0) / mu


def estimate_fluid(ctx):
    if _slot_and_kv_fit(ctx):
        return floored_t_adm(0.0, ctx)
    if ctx["batch_size"] <= 0 or ctx["remaining_steps_est"] <= 0 or ctx["t_iter"] <= 0:
        return floored_t_adm(0.0, ctx)
    waves = math.ceil((ctx["queue_depth"] + 1) / ctx["batch_size"])
    return floored_t_adm(waves * ctx["remaining_steps_est"] * ctx["t_iter"], ctx)


def estimate_rollforward(ctx):
    if _slot_and_kv_fit(ctx):
        return floored_t_adm(0.0, ctx)
    deps = []
    for r in ctx["running"]:
        rem = r["true_remaining"]
        if rem < 0:
            rem = int(ctx["remaining_steps_est"])
            if rem < 1:
                rem = 1
        deps.append((rem, r["kv_blocks"]))
    deps.sort(key=lambda d: d[0])  # stable ascending, matches sort.SliceStable
    need_slots = ctx["queue_depth"] + 1
    free_slots = ctx["max_batch_size"] - ctx["batch_size"]
    free_kv = ctx["free_kv_blocks"]
    for rem, kv in deps:
        free_slots += 1
        free_kv += kv
        if free_slots >= need_slots and free_kv >= ctx["req_kv_need"]:
            return floored_t_adm(rem * ctx["t_iter"], ctx)
    if ctx["batch_size"] > 0:
        waves = math.ceil((ctx["queue_depth"] + 1) / ctx["batch_size"])
        return floored_t_adm(waves * ctx["remaining_steps_est"] * ctx["t_iter"], ctx)
    if deps:
        return floored_t_adm(deps[-1][0] * ctx["t_iter"], ctx)
    return floored_t_adm(0.0, ctx)


ESTIMATORS = {
    "waiting": estimate_waiting,
    "fluid": estimate_fluid,
    "rollforward": estimate_rollforward,
}

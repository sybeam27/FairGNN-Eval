"""T9 -- direct uncertainty of the attribution shift Delta_attr = tau_alt - tau_primary.

The paper reports how often a change of configuration or protocol flips the sign or the resolved
status of tau_{-I->+I}. Those are comparisons of two separately summarised quantities. This script
instead treats the shift itself as the estimand and puts an interval on it.

    configuration pairs   primary vs robustness configuration, same data and splits   (26 pairs)
    selector pairs        sigma_c^BCE vs sigma_c^AUC, the same trained models         (36 pairs)
    native pairs          controlled vs native horizon/selector                       (21 pairs)
    procedure pairs       controlled vs the published-procedure bundle                (4 pairs)

Pairing: both sides are read from results/per_unit_metrics.csv.gz and joined on
(split_id, run_id), so Delta_attr is a matched per-unit difference wherever both sides come from
the same execution. Selector pairs are the same run evaluated twice. Configuration pairs are
different executions on the same splits; native and procedure pairs are dedicated runs. That is
recorded per pair in the `pairing` column: `same_run`, `same_split_different_run`.

Interval: the paper's paired hierarchical bootstrap -- resample the 6 splits, then the 5 runs
inside each resampled split, keeping the unit's two sides together. 10,000 replicates, seed below.

Reads only frozen files; writes results/phase0_audit/T9_delta_attr.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
SEED, REPS = 20260924, 10_000
COORDS = {"dAUC": ("auc", 1), "negDP": ("dp", -1), "negEO": ("eo", -1)}

u = pd.read_csv(os.path.join(RESULTS, "per_unit_metrics.csv.gz"))


def tau_I(method, dataset, protocol, selector):
    """per-unit tau_{-I->+I} for one cell, indexed by (split_id, run_id)."""
    s = u[(u.method == method) & (u.dataset == dataset)
          & (u.protocol == protocol) & (u.selector == selector)]
    if s.empty:
        return None
    p = s.pivot_table(index=["split_id", "run_id"], columns="state", values=["auc", "dp", "eo"])
    out = {}
    for c, (k, sg) in COORDS.items():
        if (k, "M_plus_I") not in p or (k, "M_minus_I") not in p:
            return None
        out[c] = sg * (p[(k, "M_plus_I")] - p[(k, "M_minus_I")])
    return pd.DataFrame(out)


def hier_ci(x: pd.Series, rng):
    """paired hierarchical bootstrap: splits, then runs within a split."""
    g = [v.to_numpy() for _, v in x.groupby(level="split_id")]
    est = np.empty(REPS)
    for r in range(REPS):
        draw = [g[i][rng.integers(0, len(g[i]), len(g[i]))] for i in rng.integers(0, len(g), len(g))]
        est[r] = np.concatenate(draw).mean()
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi), float((np.sign(est) == np.sign(x.mean())).mean())


def store_name(method, configuration):
    cfg = str(configuration)
    return method if cfg in ("default", "nan", "") else f"{method}-{cfg}"


def rows_for(pairs, family):
    rng = np.random.default_rng(SEED)
    out = []
    for p in pairs:
        a = tau_I(p["a_method"], p["dataset"], p["a_protocol"], p["a_selector"])
        b = tau_I(p["b_method"], p["dataset"], p["b_protocol"], p["b_selector"])
        if a is None or b is None:
            out.append(dict(family=family, **{k: p[k] for k in p}, coordinate="", n_units=0,
                            delta_mean=np.nan, lo=np.nan, hi=np.nan, sign_stability=np.nan,
                            excludes_zero=False, status="per-unit values unavailable"))
            continue
        idx = a.index.intersection(b.index)
        for c in COORDS:
            d = (b.loc[idx, c] - a.loc[idx, c]).dropna()
            m, lo, hi, ss = hier_ci(d, rng)
            out.append(dict(family=family, **{k: p[k] for k in p}, coordinate=c, n_units=len(d),
                            delta_mean=m, lo=lo, hi=hi, sign_stability=ss,
                            excludes_zero=bool(lo * hi > 0), status="ok"))
    return out


# --- configuration: primary vs robustness configuration of the same method and dataset
cfg = pd.read_csv(os.path.join(RESULTS, "2_configuration_variation.csv"))
cfg_pairs = [dict(dataset=r.dataset, comparison=f"{r.method}: {r.primary_configuration} -> {r.robustness_configuration}",
                  pairing="same_split_different_run",
                  a_method=store_name(r.method, r.primary_configuration), a_protocol="controlled", a_selector="common_bce",
                  b_method=store_name(r.method, r.robustness_configuration), b_protocol="controlled", b_selector="common_bce")
             for r in cfg.drop_duplicates(["method", "dataset", "primary_configuration",
                                           "robustness_configuration"]).itertuples()]

# --- selector: the same trained models evaluated under sigma_c^BCE and sigma_c^AUC
main = pd.read_csv(os.path.join(RESULTS, "1_main_package_vs_intervention.csv"))
sel_pairs = [dict(dataset=r.dataset, comparison=f"{r.method} ({r.configuration}): BCE -> AUC selector",
                  pairing="same_run",
                  a_method=store_name(r.method, r.configuration), a_protocol="controlled", a_selector="common_bce",
                  b_method=store_name(r.method, r.configuration), b_protocol="controlled", b_selector="common_auc")
             for r in main.itertuples()]

# --- native: controlled vs the method's own horizon and selector (dedicated runs)
nat = pd.read_csv(os.path.join(RESULTS, "3b_protocol_native_horizon_selector.csv"))
nat_pairs = [dict(dataset=r.dataset, comparison=f"{r.method} ({r.configuration}): controlled -> native",
                  pairing="same_split_different_run",
                  a_method=store_name(r.method, r.configuration), a_protocol="controlled", a_selector="common_bce",
                  b_method=store_name(r.method, r.configuration), b_protocol="native", b_selector="common_bce")
             for r in nat.drop_duplicates(["method", "dataset", "configuration"]).itertuples()]

pub = pd.read_csv(os.path.join(RESULTS, "3c_protocol_native_published_procedure.csv"))
pub_pairs = [dict(dataset=r.dataset, comparison=f"{r.method} ({r.configuration}): controlled -> published procedure",
                  pairing="same_split_different_run",
                  a_method=store_name(r.method, r.configuration), a_protocol="controlled", a_selector="common_bce",
                  b_method=store_name(r.method, r.configuration), b_protocol="native", b_selector="common_bce")
             for r in pub.drop_duplicates(["method", "dataset", "configuration"]).itertuples()]

rows = (rows_for(cfg_pairs, "configuration") + rows_for(sel_pairs, "selector")
        + rows_for(nat_pairs, "native_horizon_selector") + rows_for(pub_pairs, "published_procedure"))
d = pd.DataFrame(rows)
d.insert(0, "seed", SEED)
d.to_csv(os.path.join(HERE, "T9_delta_attr.csv"), index=False)

ok = d[d.status == "ok"]
print(f"seed={SEED}, reps={REPS}")
print("pairs per family (comparisons x coordinates):")
print(ok.groupby("family").agg(rows=("coordinate", "size"),
                               comparisons=("comparison", "nunique"),
                               excludes_zero=("excludes_zero", "sum"),
                               median_abs_delta=("delta_mean", lambda s: float(np.median(s.abs())))).to_string())
print()
print("by family and coordinate:")
print(ok.groupby(["family", "coordinate"]).agg(n=("coordinate", "size"),
                                               excludes_zero=("excludes_zero", "sum"),
                                               median_abs=("delta_mean", lambda s: round(float(np.median(s.abs())), 4))).to_string())
miss = d[d.status != "ok"]
if len(miss):
    print("\nunavailable:", miss.comparison.tolist())

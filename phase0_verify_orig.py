"""Phase 0 integrity check for the results bundle.

Usage:  python phase0_verify.py /path/to/results
Recomputes the primary contrasts from per_unit_metrics.csv.gz and checks them
against the paper-facing CSVs. Prints a short report; nothing is modified.
"""
import sys
import numpy as np
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"
u = pd.read_csv(f"{R}/per_unit_metrics.csv.gz")
m = pd.read_csv(f"{R}/1_main_package_vs_intervention.csv")
NAME = {"BIND": "BIND-1pct", "FairSIN": "FairSIN-GCN"}
COORDS = [("dAUC", "auc", 1), ("negDP", "dp", -1), ("negEO", "eo", -1)]
rng = np.random.default_rng(0)


def unit_contrasts(method, dataset, protocol="controlled", selector="common_bce"):
    s = u[(u.method == method) & (u.dataset == dataset)
          & (u.protocol == protocol) & (u.selector == selector)]
    p = s.pivot_table(index=["split_id", "run_id"], columns="state", values=["auc", "dp", "eo"])
    out = {}
    for c, k, sg in COORDS:
        b, mi, pl = p[(k, "B")], p[(k, "M_minus_I")], p[(k, "M_plus_I")]
        out[("I", c)] = sg * (pl - mi)
        out[("nonint", c)] = sg * (mi - b)
        out[("pkg", c)] = sg * (pl - b)
    return pd.DataFrame(out)


def hier_boot(x, reps=4000):
    groups = [g.values for _, g in x.groupby(level="split_id")]
    est = np.empty(reps)
    for r in range(reps):
        draw = [groups[i][rng.integers(0, len(groups[i]), len(groups[i]))]
                for i in rng.integers(0, len(groups), len(groups))]
        est[r] = np.concatenate(draw).mean()
    return est


rows = []
for _, r in m.iterrows():
    C = unit_contrasts(NAME.get(r.method, r.method), r.dataset)
    for c, _, _ in COORDS:
        x = C[("I", c)]
        mu = x.mean()
        est = hier_boot(x)
        lo, hi = np.percentile(est, [2.5, 97.5])
        nz = x[x != 0]
        rows.append(dict(
            method=r.method, dataset=r.dataset, coord=c, mean=mu,
            mean_err=abs(mu - r[f"tau_I_{c}_mean"]),
            lo=lo, hi=hi, lo_csv=r[f"tau_I_{c}_lo"], hi_csv=r[f"tau_I_{c}_hi"],
            ss_csv=r[f"tau_I_{c}_sign_stability"],
            ss_unit_agree=(np.sign(nz) == np.sign(mu)).mean() if len(nz) else 0.0,
            ss_boot=(np.sign(est) == np.sign(mu)).mean(),
            res_csv=bool(r[f"tau_I_{c}_resolved"]),
        ))
T = pd.DataFrame(rows)
ci_csv = (T.lo_csv > 0) | (T.hi_csv < 0)
big = T["mean"].abs() >= 0.01
T["res_unit_rule"] = ci_csv & big & (T.ss_csv >= 0.75)
T["res_boot_rule"] = ci_csv & big & (T.ss_boot >= 0.75)

print("== 1. Point estimates reproduce from per-unit export")
print("   max |mean diff| =", T.mean_err.max())
print("   max |CI diff| (MC error, 4k reps) =",
      max((T.lo - T.lo_csv).abs().max(), (T.hi - T.hi_csv).abs().max()))
print("== 2. Resolved counts under two sign-stability definitions")
print(T.groupby("coord")[["res_csv", "res_unit_rule", "res_boot_rule"]].sum())
print("   cells that differ (CI and |tau| pass, unit-level stability < 0.75):")
d = T[T.res_unit_rule != T.res_boot_rule]
print(d[["method", "dataset", "coord", "mean", "lo_csv", "hi_csv", "ss_csv"]].round(4).to_string())
print("   CSV stability != agreement-with-estimate:",
      int((~np.isclose(T.ss_csv, T.ss_unit_agree)).sum()), "rows")

print("== 3. Common baseline B")
b = u[(u.protocol == "controlled") & (u.selector == "common_bce") & (u.state == "B")]
print(b.groupby("dataset")[["auc", "dp", "eo"]].mean().round(3))
spread = b.groupby(["dataset", "split_id", "run_id"]).auc.agg(lambda s: s.max() - s.min())
print("   max AUC spread of B across methods, per dataset:")
print(spread.groupby("dataset").max().round(4).to_string())

print("== 4. Headline counts")
for c, _, _ in COORDS:
    ni, ti, tp = m[f"tau_nonint_{c}_mean"], m[f"tau_I_{c}_mean"], m[f"tau_pkg_{c}_mean"]
    print(f"   {c}: nonint larger {int((ni.abs() > ti.abs()).sum())}/36, "
          f"pkg<0 {int((tp < 0).sum())}/36, |tau_I|<0.01 {int((ti.abs() < 0.01).sum())}/36")

print("== 5. Configuration sign flips, substantive only")
cv = pd.read_csv(f"{R}/2_configuration_variation.csv")
cv["subst"] = cv.sign_changed & (cv.primary_tau_I_mean.abs() >= .01) & (cv.robustness_tau_I_mean.abs() >= .01)
cv["res_opp"] = cv.sign_changed & cv.primary_resolved & cv.robustness_resolved
print(cv.groupby("coordinate")[["sign_changed", "subst", "res_opp", "resolution_changed"]].sum())

print("== 6. NIFTY/German 2x2 on negDP")
mc = pd.read_csv(f"{R}/4_mechanistic_case_study.csv")
x = mc[(mc.method == "NIFTY") & (mc.coordinate == "negDP")].set_index("term")["mean"]
for lab, h1 in [("dedicated rerun", "bce.c1000"), ("frozen runs", "bce.frz")]:
    t00, t01, t10, t11 = x["D0.bce.h200"], x["D1.bce.h200"], x[f"D0.{h1}"], x[f"D1.{h1}"]
    print(f"   {lab}: theta00={t00:.4f} theta01={t01:.4f} theta10={t10:.4f} theta11={t11:.4f} | "
          f"dH_avg={((t10 - t00) + (t11 - t01)) / 2:.4f} dH|D1={t11 - t01:.4f} "
          f"dD={((t01 - t00) + (t11 - t10)) / 2:.4f} Gamma={t11 - t10 - t01 + t00:.4f}")

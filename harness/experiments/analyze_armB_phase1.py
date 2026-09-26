"""Arm B Phase 1: native-configuration cells against corrected Arm A (X14).

For each (method, dataset) present in the native CSVs, the matching corrected
Arm A cells are paired by (split, run).

    Arm A          tau_base^ArmA, tau_int^ArmA, tau_pkg^ArmA (H = 200)
    Arm B native   tau_int^native     = Y(M1^native) - Y(M0^native)
                   tau_base^fixed-ref = Y(M0^native) - Y(B_200)
                   tau_pkg^fixed-ref  = Y(M1^native) - Y(B_200)

B has no native protocol, so `tau_base^native` and `tau_pkg^native` are never
printed. B_200 is the same reference configuration in both arms. The native run
retrains it in-process at the same seed, however, so it is a fresh draw; its
drift from Arm A's B is reported as a diagnostic, not hidden.

What is compared is qualitative structure, not whether the numbers agree:

    * whether |tau_base| > |tau_int| holds;
    * the fairness-axis direction of tau_int;
    * resolved vs unresolved, using the frozen instruments: descriptive rule
      (sign stability >= 0.75 and |mean| >= 0.010) together with the paired
      hierarchical bootstrap's 95% interval excluding 0;
    * selector sensitivity: share of cells with |D| > |tau_int^BCE|.

The FairGB/credit trigger is evaluated exactly as fixed in X12, before any
result.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
from analyze_armA import build, sign_stability                       # noqa: E402
from bootstrap_armA import boot                                      # noqa: E402

ARMA = ["harness/results/armA_german.csv", "harness/results/armA_bail.csv",
        "harness/results/armA_bail_s23_25.csv", "harness/results/armA_credit.csv",
        "harness/results/armA_credit_s23_25.csv",
        "harness/results/armA_fairvgnn_german.csv",
        "harness/results/armA_fairvgnn_bail.csv",
        "harness/results/armA_fairvgnn_credit.csv"]
SEL = "common_bce"
SIGN_MIN, NEAR_ZERO = 0.75, 0.010
FULL_DESIGN = 30    # 6 splits x 5 runs, the frozen design; below it nothing is decided
B_REPS = 10_000


def cell_table3(d, method, dataset):
    """One row per (split, run), paired: BCE-slot effects on all three
    coordinates, and D = tau_int^BCE - tau_int^AUC for each. The Arm A
    bootstrap's own table carries AUC and DP only; EO is secondary here and
    gets the same treatment."""
    g = d[(d.method == method) & (d.dataset == dataset)]
    vals = [f"{q}_{c}" for q in ("abase", "int", "apkg") for c in ("auc", "ndp", "neo")]
    w = g.pivot_table(index=["split_id", "run_id"], columns="selector", values=vals)
    out = pd.DataFrame(index=w.index)
    for v in vals:
        out[v] = w[(v, "common_bce")]
    for c in ("auc", "ndp", "neo"):
        out[f"D_{c}"] = w[(f"int_{c}", "common_bce")] - w[(f"int_{c}", "common_auc")]
    return out.reset_index()


def summarize(d, method, dataset, rng, coord):
    """Means, sign stability, bootstrap interval and resolved state on one coord."""
    g = d[(d.method == method) & (d.dataset == dataset) & (d.selector == SEL)]
    cells = cell_table3(d, method, dataset)
    have = [f"abase_{coord}", f"int_{coord}", f"apkg_{coord}", f"D_{coord}"]
    reps = boot(cells, have, rng, reps=B_REPS) if len(cells) else None
    out = {}
    for q in ("abase", "int", "apkg"):
        col = f"{q}_{coord}"
        mu = float(g[col].mean())
        lo, hi = (np.percentile(reps[:, have.index(col)], [2.5, 97.5])
                  if reps is not None and col in have else (np.nan, np.nan))
        out[q] = dict(mean=mu, sign=sign_stability(g[col]), lo=lo, hi=hi)
    i = out["int"]
    out["complete"] = int(len(g)) == FULL_DESIGN
    # With fewer cells sign stability is trivially high and the bootstrap
    # interval collapses to a point, so "resolved" would be manufactured by the
    # sample size. The state is only defined on the full frozen design.
    i["resolved"] = (bool(i["sign"] >= SIGN_MIN and abs(i["mean"]) >= NEAR_ZERO
                          and np.isfinite(i["lo"]) and i["lo"] * i["hi"] > 0)
                     if out["complete"] else None)
    out["dominates"] = abs(out["abase"]["mean"]) > abs(i["mean"])
    w = d[(d.method == method) & (d.dataset == dataset)].pivot_table(
        index=["split_id", "run_id"], columns="selector", values=f"int_{coord}")
    if {"common_bce", "common_auc"} <= set(w.columns):
        D = (w["common_bce"] - w["common_auc"]).to_numpy()
        I = w["common_bce"].to_numpy()
        out["sel_share"] = float(np.mean(np.abs(D) > np.abs(I)))
    else:
        out["sel_share"] = np.nan
    out["n"] = int(len(g))
    return out


def transfer_class(sa, sn):
    """Controlled -> native transfer of the intervention conclusion, fixed in X22
    before the FairVGNN/credit result existed. Built only from frozen instruments:
    the resolved state (descriptive rule + bootstrap interval), the direction of
    tau_int, and the |tau_base| > |tau_int| relation. Precedence, first match:

        stable reversal     resolved in both arms, opposite signs
        resolution gained   unresolved -> resolved
        resolution lost     resolved -> unresolved
        relation changed    resolved state unchanged, relation flipped
        maintained          resolved state, direction if resolved, and relation
                            all unchanged
    """
    ra, rn = sa["int"]["resolved"], sn["int"]["resolved"]
    if ra is None or rn is None:
        return "n/a (incomplete design)"
    if ra and rn and np.sign(sa["int"]["mean"]) != np.sign(sn["int"]["mean"]):
        return "stable reversal"
    if (not ra) and rn:
        return "resolution gained"
    if ra and (not rn):
        return "resolution lost"
    if sa["dominates"] != sn["dominates"]:
        return "relation changed"
    return "maintained"


def fmt_q(x):
    return (f"{x['mean']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}] s{x['sign']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", nargs="*",
                    default=sorted(glob.glob("harness/results/armB_native_*.csv")))
    a = ap.parse_args()
    if not a.native:
        raise SystemExit("no native CSVs yet")

    A = build(ARMA)
    N = build(a.native)
    assert (N.protocol == "native").all(), "native CSVs must carry protocol=native"
    rng = np.random.default_rng(20260915)

    cells = sorted(N[["method", "dataset"]].drop_duplicates().itertuples(index=False))
    print(f"Arm B Phase 1 vs corrected Arm A -- {len(cells)} native cell(s), "
          f"sigma_c = {SEL}, bootstrap {B_REPS:,} paired hierarchical replicates")
    print("native quantities: tau_int^native, tau_base^fixed-ref, tau_pkg^fixed-ref "
          "(B_200 reference)\n")

    trigger = {}
    summary = []
    for m, ds in cells:
        g_n = N[(N.method == m) & (N.dataset == ds) & (N.selector == SEL)]
        g_a = A[(A.method == m) & (A.dataset == ds) & (A.selector == SEL)]
        print("=" * 108)
        print(f"{m} / {ds}   native H = {sorted(g_n.method_epochs.unique())}   "
              f"Arm A cells {len(g_a)}  native cells {len(g_n)}")
        print("=" * 108)

        # contract on the native side
        ra = (g_n.m1_auc - g_n.bc_auc - ((g_n.m0_auc - g_n.bc_auc) + (g_n.m1_auc - g_n.m0_auc))).abs().max()
        nf = int((~np.isfinite(g_n[["m1_auc", "m0_auc", "bc_auc", "m1_dp", "m0_dp", "bc_dp"]]
                               .to_numpy())).sum())
        print(f"  contract: alignment residual {ra:.1e}, non-finite {nf}, "
              f"EO undefined {int((g_n.eo_defined == 0).sum())}, "
              f"rng_contract {sorted(g_n.rng_contract.unique())}")
        j = g_n.merge(g_a, on=["split_id", "run_id"], suffixes=("_n", "_a"))
        print(f"  B_200 drift (native in-process redraw vs Arm A B, paired): "
              f"mean |dAUC| {np.abs(j.bc_auc_n - j.bc_auc_a).mean():.4f}  "
              f"mean |dDP| {np.abs(j.bc_dp_n - j.bc_dp_a).mean():.4f}  "
              f"bit-identical cells {int(((j.bc_auc_n == j.bc_auc_a) & (j.bc_dp_n == j.bc_dp_a)).sum())}/{len(j)}")

        for coord, lab in (("ndp", "-dDP  (primary)"), ("neo", "-dEO  (secondary)"),
                           ("auc", "dAUC")):
            sa = summarize(A, m, ds, rng, coord)
            sn = summarize(N, m, ds, rng, coord)
            print(f"\n  {lab}")
            print(f"    {'':<24}{'Arm A':>42}{'native':>42}")
            print(f"    {'tau_base':<24}{fmt_q(sa['abase']):>42}{fmt_q(sn['abase']):>42}"
                  f"   (native = fixed-ref)")
            print(f"    {'tau_int':<24}{fmt_q(sa['int']):>42}{fmt_q(sn['int']):>42}")
            print(f"    {'|tau_base| > |tau_int|':<24}{str(sa['dominates']):>42}{str(sn['dominates']):>42}")
            print(f"    {'tau_int direction':<24}{('+' if sa['int']['mean'] > 0 else '-'):>42}"
                  f"{('+' if sn['int']['mean'] > 0 else '-'):>42}")
            print(f"    {'tau_int resolved':<24}{str(sa['int']['resolved']):>42}"
                  f"{('n/a: ' + str(sn['n']) + '/' + str(FULL_DESIGN) + ' cells') if sn['int']['resolved'] is None else str(sn['int']['resolved']):>42}")
            print(f"    {'selector |D|>|tau_int|':<24}{sa['sel_share']:>42.2f}{sn['sel_share']:>42.2f}")
            tc = transfer_class(sa, sn)
            print(f"    {'transfer class (X22)':<24}{tc:>84}")
            summary.append(dict(method=m, dataset=ds, coord=lab.split()[0], sa=sa, sn=sn, tc=tc))
            if coord == "ndp" and not (sa["complete"] and sn["complete"]):
                print(f"    qualitative change on -dDP: not evaluated "
                      f"(Arm A {sa['n']}, native {sn['n']} of {FULL_DESIGN} cells)")
            elif coord == "ndp":
                rev_rel = sa["dominates"] != sn["dominates"]
                stable_rev = (sn["int"]["resolved"]
                              and np.sign(sn["int"]["mean"]) != np.sign(sa["int"]["mean"]))
                flip = sa["int"]["resolved"] != sn["int"]["resolved"]
                trigger[(m, ds)] = dict(relation_reversed=rev_rel,
                                        stable_direction_reversal=bool(stable_rev),
                                        resolved_state_flip=flip)
                print(f"    qualitative change on -dDP: relation reversed {rev_rel}, "
                      f"stable direction reversal {bool(stable_rev)}, "
                      f"resolved/unresolved flip {flip}")

    print("\n" + "=" * 108)
    print("Method x dataset summary (sigma_c^BCE; interval = 95% paired hierarchical bootstrap, "
          "s = sign stability)")
    print("=" * 108)
    for coord in ("-dDP", "-dEO", "dAUC"):
        print(f"\n  {coord}")
        print(f"  {'method':<9}{'dataset':<8}{'tau_base A -> fixed-ref':>34}"
              f"{'tau_int A -> native':>60}{'|b|>|i|':>9}{'resolved':>10}{'sel share':>13}  transfer")
        for r_ in sorted((x for x in summary if x["coord"] == coord),
                         key=lambda x: (x["method"], x["dataset"])):
            sa, sn = r_["sa"], r_["sn"]
            ib = f"{sa['abase']['mean']:+.3f} -> {sn['abase']['mean']:+.3f}"
            ii = (f"{sa['int']['mean']:+.3f} [{sa['int']['lo']:+.3f},{sa['int']['hi']:+.3f}] s{sa['int']['sign']:.2f}"
                  f" -> {sn['int']['mean']:+.3f} [{sn['int']['lo']:+.3f},{sn['int']['hi']:+.3f}] s{sn['int']['sign']:.2f}")
            rel = f"{'T' if sa['dominates'] else 'F'}->{'T' if sn['dominates'] else 'F'}"
            res = f"{str(sa['int']['resolved'])[0]}->{str(sn['int']['resolved'])[0]}"
            sel = f"{sa['sel_share']:.2f}->{sn['sel_share']:.2f}"
            print(f"  {r_['method']:<9}{r_['dataset']:<8}{ib:>34}   {ii:>60}{rel:>9}{res:>10}{sel:>13}  {r_['tc']}")

    print("\n" + "=" * 108)
    print("FairGB/credit trigger (X12): runs iff any FairGB german/bail cell shows a "
          "relation reversal,\na stable fairness-direction reversal, or a "
          "resolved/unresolved flip on -dDP")
    print("=" * 108)
    gb = {k: v for k, v in trigger.items() if k[0] == "FairGB"}
    for k, v in sorted(gb.items()):
        print(f"  {k[0]}/{k[1]}: {v}")
    need = {("FairGB", "german"), ("FairGB", "bail")}
    if not need <= set(gb):
        print(f"  pending: {sorted(need - set(gb))} not yet complete -- no decision")
    else:
        fire = any(any(v.values()) for v in gb.values())
        print(f"  -> FairGB/credit {'RUNS' if fire else 'stays held as a confirmatory cell'}")


if __name__ == "__main__":
    main()

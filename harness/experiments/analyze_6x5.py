"""6x5 audit analysis.

Two views are kept apart and never mixed.

    Published view
        tau_pkg^pub = Y(M1; H_p, sigma_p) - Y(B; H_B, sigma_B)
        The published system against the baseline, each at its own selector.
        Reported on its own. Not used in the intervention decomposition.

    Audit attribution view -- everything at one (sigma_c, delta_c, G_c, H)
        tau_base^audit = Y(M0) - Y(B)
        tau_int        = Y(M1) - Y(M0)
        tau_pkg^audit  = Y(M1) - Y(B)

The residual of tau_pkg^audit - (tau_base^audit + tau_int) is a join/alignment
regression check, not a scientific result: it holds for any common subtrahend,
so it passes even on a contaminated join. It is here to catch a mis-join.

Data contract: B, M0 and M1 of one (split, run) come from one execution
context. The baseline is read from the bc_* columns that pilot_tau.py writes
inside the process that trains B. No baseline is retrained and joined.

Coordinates [dAUC, -dDP] primary, [dAUC, -dEO] secondary; larger is better in
both. Per method, then per split, then per run. Overall means last.

Attribution class is descriptive, assigned per coordinate, never compressed to
one label per method, and never phrased as statistical significance.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

SIGN_MIN = 0.75      # modal-sign share below this -> unresolved
NEAR_ZERO = 0.010    # |mean| below this -> unresolved
SHARE_HI = 0.50
SHARE_MIN_MAG = 0.010   # both terms must clear this before a share is quoted

TERMS = (("tau_base^audit", "abase"), ("tau_int", "int"),
         ("tau_pkg^audit", "apkg"))


def fmt(v, w=4):
    return "[" + ", ".join(f"{x:+.{w}f}" for x in np.atleast_1d(v)) + "]"


def sign_stability(x):
    x = np.asarray(x, float)
    nz = x[x != 0.0]
    return 0.0 if nz.size == 0 else max((nz > 0).mean(), (nz < 0).mean())


def build(csvs):
    d = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    d = d.drop_duplicates(["method", "split_id", "run_id", "selector"])
    missing = {"bc_auc", "bc_dp", "bc_eo"} - set(d.columns)
    if missing:
        raise SystemExit(
            f"pilot CSV lacks the baseline under sigma_c ({sorted(missing)}). "
            "A retrained baseline cannot supply it -- GPU nondeterminism "
            "diverged the baseline in 16 of 30 cells. Re-run pilot_tau.py, "
            "which records bc_* in the process that trains B.")
    d["pubpkg_auc"] = d.m1pub_auc - d.b_auc
    d["pubpkg_ndp"] = -(d.m1pub_dp - d.b_dp)
    d["pubpkg_neo"] = -(d.m1pub_eo - d.b_eo)
    for tag, m in (("apkg", "m1"), ("abase", "m0")):
        d[f"{tag}_auc"] = d[f"{m}_auc"] - d.bc_auc
        d[f"{tag}_ndp"] = -(d[f"{m}_dp"] - d.bc_dp)
        d[f"{tag}_neo"] = -(d[f"{m}_eo"] - d.bc_eo)
    d["int_auc"] = d.m1_auc - d.m0_auc
    d["int_ndp"] = -(d.m1_dp - d.m0_dp)
    d["int_neo"] = -(d.m1_eo - d.m0_eo)
    return d


def contract_checks(d, splits, runs, methods):
    print("\n" + "=" * 100)
    print("[0] data contract")
    print("=" * 100)
    ok = True
    for fair in ("ndp", "neo"):
        r = (d[f"apkg_{fair}"] - (d[f"abase_{fair}"] + d[f"int_{fair}"])).abs().max()
        ra = (d.apkg_auc - (d.abase_auc + d.int_auc)).abs().max()
        print(f"  alignment residual  dAUC {ra:.2e}   -d{fair[1:].upper()} {r:.2e}"
              f"   {'OK' if max(r, ra) < 1e-9 else 'BROKEN -- join is wrong'}")
        ok &= max(r, ra) < 1e-9
    print("  (holds for any common subtrahend; this catches a mis-join, it is "
          "not a finding)")

    # B, M1^pub and the published package effect must not depend on sigma_c
    g = d.groupby(["method", "split_id", "run_id"])[["b_auc", "b_dp", "m1pub_auc",
                                                     "pubpkg_auc", "pubpkg_ndp"]]
    spread = (g.max() - g.min()).max().max()
    print(f"  published view invariant across sigma_c slots: max spread "
          f"{spread:.2e}  {'OK' if spread < 1e-12 else 'BROKEN'}")
    ok &= spread < 1e-12

    n = d[d.selector == "common_bce"].groupby("method").size()
    want = len(splits) * len(runs)
    bal = (n.nunique() == 1 and int(n.iloc[0]) == want)
    print(f"  design balance: {want} cells expected per method -- "
          f"{'OK' if bal else 'UNBALANCED'}")
    if not bal:
        print(n.to_string())
    bad = d[["m1_auc", "m0_auc", "bc_auc", "m1_dp", "m0_dp", "bc_dp"]]
    nn = int((~np.isfinite(bad.to_numpy())).sum())
    print(f"  non-finite outcomes: {nn}  {'OK' if nn == 0 else 'EFFECTS UNDEFINED'}")
    und = int((d.eo_defined == 0).sum())
    print(f"  EO undefined in {und} of {len(d)} rows -- reported, not dropped")
    return ok and bal and nn == 0


def published_view(d, methods, splits, runs):
    print("\n" + "=" * 100)
    print("[1] Published view   tau_pkg^pub = Y(M1; H_p, sigma_p) - Y(B; own "
          "selector)")
    print("    reported on its own; not an input to the decomposition below")
    print("=" * 100)
    p = d[d.selector == "common_bce"]      # selector-invariant, checked above
    for m in methods:
        g = p[p.method == m].sort_values(["split_id", "run_id"])
        print(f"\n  {m}   raw paired effects")
        print(f"{'split':>7}{'run':>5}   {'[dAUC, -dDP]':>22}{'[dAUC, -dEO]':>22}")
        for _, r in g.iterrows():
            print(f"{r.split_id:>7}{r.run_id:>5}   "
                  f"{fmt([r.pubpkg_auc, r.pubpkg_ndp]):>22}"
                  f"{fmt([r.pubpkg_auc, r.pubpkg_neo]):>22}")
        print(f"    per split   " + "".join(
            f"{s}:{fmt(g[g.split_id == s][['pubpkg_auc', 'pubpkg_ndp']].mean())}  "
            for s in splits))
        print(f"    per run     " + "".join(
            f"{r_}:{fmt(g[g.run_id == r_][['pubpkg_auc', 'pubpkg_ndp']].mean())}  "
            for r_ in runs))


def audit_view(d, methods, splits, runs, sel):
    ds = d[d.selector == sel]
    print("\n" + "=" * 100)
    print(f"[2] Audit attribution view   all terms at sigma_c = {sel}")
    print("=" * 100)
    for m in methods:
        g = ds[ds.method == m].sort_values(["split_id", "run_id"])
        print("\n" + "-" * 100)
        print(f"{m}   raw paired effects, primary [dAUC, -dDP]")
        print("-" * 100)
        print(f"{'split':>7}{'run':>5}{'epB':>5}{'epM0':>6}{'epM1':>6}   "
              f"{'tau_base^audit':>22}{'tau_int':>22}{'tau_pkg^audit':>22}")
        for _, r in g.iterrows():
            print(f"{r.split_id:>7}{r.run_id:>5}{r.bc_epoch:>5}{r.m0_epoch:>6}"
                  f"{r.m1_epoch:>6}   "
                  f"{fmt([r.abase_auc, r.abase_ndp]):>22}"
                  f"{fmt([r.int_auc, r.int_ndp]):>22}"
                  f"{fmt([r.apkg_auc, r.apkg_ndp]):>22}")
        for lab, key, idx in (("per split", "split_id", splits),
                              ("per run", "run_id", runs)):
            print(f"\n  {lab}")
            print(f"{key:>10}   {'tau_base^audit':>22}{'tau_int':>22}"
                  f"{'tau_pkg^audit':>22}")
            for v in idx:
                h = g[g[key] == v]
                print(f"{v:>10}   {fmt(h[['abase_auc','abase_ndp']].mean()):>22}"
                      f"{fmt(h[['int_auc','int_ndp']].mean()):>22}"
                      f"{fmt(h[['apkg_auc','apkg_ndp']].mean()):>22}")
        print(f"\n  secondary [dAUC, -dEO], per split")
        for v in splits:
            h = g[g.split_id == v]
            print(f"{v:>10}   {fmt(h[['abase_auc','abase_neo']].mean()):>22}"
                  f"{fmt(h[['int_auc','int_neo']].mean()):>22}"
                  f"{fmt(h[['apkg_auc','apkg_neo']].mean()):>22}")


def classify(d, methods, sel):
    ds = d[d.selector == sel]
    print("\n" + "=" * 100)
    print("[3] Attribution class -- descriptive, per coordinate, one row per "
          "(method, coordinate)")
    print(f"    unresolved if sign stability < {SIGN_MIN} or |mean| < "
          f"{NEAR_ZERO}; this is not a significance statement")
    print("=" * 100)
    print(f"{'method':<10}{'coord':<7}{'mean int':>10}{'sd':>8}{'sgn':>6}"
          f"{'mean base':>11}{'sgn':>6}{'mean pkg':>10}{'sgn':>6}   class")
    out = {}
    for m in methods:
        g = ds[ds.method == m]
        for coord, lab in (("auc", "dAUC"), ("ndp", "-dDP"), ("neo", "-dEO")):
            i = g[f"int_{coord}"].to_numpy()
            b_ = g[f"abase_{coord}"].to_numpy()
            k = g[f"apkg_{coord}"].to_numpy()
            mi, mb, mk = i.mean(), b_.mean(), k.mean()
            si, sb, sk = (sign_stability(x) for x in (i, b_, k))
            i_res = si >= SIGN_MIN and abs(mi) >= NEAR_ZERO
            k_res = sk >= SIGN_MIN and abs(mk) >= NEAR_ZERO
            if not k_res:
                cls = "unresolved (package effect unresolved)"
            elif not i_res:
                cls = ("unresolved / protocol-sensitive "
                       f"({'near zero' if abs(mi) < NEAR_ZERO else 'sign unstable'})")
            elif np.sign(mi) != np.sign(mk):
                cls = "ATTRIBUTION REVERSAL"
            elif abs(mi) >= abs(mb):
                cls = "intervention-supported"
            else:
                cls = "package-driven"
            out[(m, lab)] = (cls, mi, mb, mk, si, sk)
            print(f"{m:<10}{lab:<7}{mi:>+10.4f}{i.std(ddof=1):>8.4f}{si:>6.2f}"
                  f"{mb:>+11.4f}{sb:>6.2f}{mk:>+10.4f}{sk:>6.2f}   {cls}")
    return out


def shares(cls, methods):
    print("\n" + "=" * 100)
    print("[4] Attribution share  tau_int / tau_pkg^audit -- quoted only when "
          "both terms")
    print(f"    clear |{SHARE_MIN_MAG}| and point the same way; otherwise "
          "described as offset/amplification")
    print("=" * 100)
    for m in methods:
        for lab in ("dAUC", "-dDP", "-dEO"):
            c, mi, mb, mk = cls[(m, lab)][:4]
            if abs(mi) < SHARE_MIN_MAG or abs(mk) < SHARE_MIN_MAG:
                note = ("one term is too small to divide by; no share")
            elif np.sign(mi) != np.sign(mk):
                note = (f"opposite directions: base {mb:+.4f} and intervention "
                        f"{mi:+.4f} offset, package lands at {mk:+.4f}; "
                        "no share")
            elif np.sign(mb) != np.sign(mi):
                note = (f"base {mb:+.4f} opposes intervention {mi:+.4f}; the "
                        f"package {mk:+.4f} is what survives the offset; "
                        "no share")
            else:
                note = (f"share {mi / mk:.2f}  (base {mb:+.4f}, "
                        f"intervention {mi:+.4f}, package {mk:+.4f})")
            print(f"  {m:<10}{lab:<7}{note}")


def selector_sensitivity(d, methods, splits, runs):
    print("\n" + "=" * 100)
    print("[5] Selector sensitivity -- paired within (method, split, run)")
    print("=" * 100)
    w = d.pivot_table(index=["method", "split_id", "run_id"], columns="selector",
                      values=["int_auc", "int_ndp", "int_neo", "m1_auc", "m1_dp",
                              "m0_auc", "m0_dp", "bc_auc", "bc_dp", "bc_epoch",
                              "m0_epoch", "m1_epoch"])

    print("\n  D = tau_int^BCE - tau_int^AUC")
    print(f"{'method':<10}{'n':>4}{'mean D':>24}{'sd D':>24}{'sign stab':>20}"
          f"{'|D|>|tau_int^BCE|':>20}")
    for m in methods:
        h = w.loc[m]
        da = (h[("int_auc", "common_bce")] - h[("int_auc", "common_auc")]).to_numpy()
        dn = (h[("int_ndp", "common_bce")] - h[("int_ndp", "common_auc")]).to_numpy()
        ib = h[("int_ndp", "common_bce")].to_numpy()
        frac = float((np.abs(dn) > np.abs(ib)).mean())
        print(f"{m:<10}{len(da):>4}{fmt([da.mean(), dn.mean()]):>24}"
              f"{fmt([da.std(ddof=1), dn.std(ddof=1)]):>24}"
              f"{fmt([sign_stability(da), sign_stability(dn)]):>20}{frac:>20.2f}")
    print("  last column: share of cells where swapping the selector moves "
          "tau_int by more\n  than tau_int itself is.")

    print("\n  Does the selector move the components themselves? "
          "value^BCE - value^AUC")
    print("  B is the same baseline for every method -- if B moves as much as "
          "M1 does,\n  selector sensitivity is a property of the protocol, not "
          "of the fair method.")
    print(f"{'method':<10}{'component':<12}{'mean dAUC':>12}{'mean dDP':>12}"
          f"{'sd dAUC':>11}{'sd dDP':>11}{'mean |dEpoch|':>15}")
    for m in methods:
        h = w.loc[m]
        for comp, ep in (("B", "bc_epoch"), ("M0", "m0_epoch"), ("M1", "m1_epoch")):
            pre = {"B": "bc", "M0": "m0", "M1": "m1"}[comp]
            da = (h[(f"{pre}_auc", "common_bce")] - h[(f"{pre}_auc", "common_auc")]).to_numpy()
            dd = (h[(f"{pre}_dp", "common_bce")] - h[(f"{pre}_dp", "common_auc")]).to_numpy()
            de = (h[(ep, "common_bce")] - h[(ep, "common_auc")]).abs().to_numpy()
            print(f"{m:<10}{comp:<12}{da.mean():>+12.4f}{dd.mean():>+12.4f}"
                  f"{da.std(ddof=1):>11.4f}{dd.std(ddof=1):>11.4f}{de.mean():>15.1f}")


def uncertainty(d, methods, splits, runs, sel):
    ds = d[d.selector == sel]
    print("\n" + "=" * 100)
    print(f"[6] Split-level vs run-level variation, recomputed on "
          f"{len(splits)}x{len(runs)}")
    print("    repeats are run-level stochastic variation, not seed "
          "uncertainty: the same seed\n    can diverge through GPU "
          "nondeterminism (measured: 16 of 30 baseline cells)")
    print("    3x5 ratios stay a pilot estimate and are not carried into any "
          "headline")
    print("=" * 100)
    print(f"{'method':<10}{'effect':<16}{'split_sd':>22}{'run_sd':>22}"
          f"{'split/run':>20}")
    for m in methods:
        g = ds[ds.method == m]
        for name, tag in TERMS:
            cols = [f"{tag}_auc", f"{tag}_ndp"]
            ss = g.groupby("split_id")[cols].mean().std(ddof=1).to_numpy()
            rs = g.groupby("run_id")[cols].mean().std(ddof=1).to_numpy()
            rt = np.where(rs > 0, ss / np.maximum(rs, 1e-12), np.nan)
            print(f"{m:<10}{name:<16}{fmt(ss):>22}{fmt(rs):>22}{fmt(rt):>20}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", nargs="+", required=True)
    ap.add_argument("--selector", default="common_bce")
    a = ap.parse_args()

    d = build(a.pilot)
    sel = a.selector
    methods = sorted(d.method.unique())
    splits = sorted(d.split_id.unique())
    runs = sorted(d.run_id.unique())
    print(f"{len(d)} rows | {len(splits)} splits x {len(runs)} runs x "
          f"{len(methods)} methods | primary sigma_c = {sel}")
    print("B, M0, M1 of a cell come from one execution context (bc_* written "
          "in-process)")

    contract_checks(d, splits, runs, methods)
    published_view(d, methods, splits, runs)
    audit_view(d, methods, splits, runs, sel)
    cls = classify(d, methods, sel)
    shares(cls, methods)
    selector_sensitivity(d, methods, splits, runs)
    uncertainty(d, methods, splits, runs, sel)

    print("\n" + "=" * 100)
    print("Overall means -- computed last, read last")
    print("=" * 100)
    ds = d[d.selector == sel]
    p = d[d.selector == "common_bce"]
    print(f"{'method':<10}{'tau_pkg^pub':>22}{'tau_base^audit':>22}"
          f"{'tau_int':>22}{'tau_pkg^audit':>22}")
    for m in methods:
        g, gp = ds[ds.method == m], p[p.method == m]
        print(f"{m:<10}{fmt(gp[['pubpkg_auc','pubpkg_ndp']].mean()):>22}"
              f"{fmt(g[['abase_auc','abase_ndp']].mean()):>22}"
              f"{fmt(g[['int_auc','int_ndp']].mean()):>22}"
              f"{fmt(g[['apkg_auc','apkg_ndp']].mean()):>22}")
    print("\ngerman + GCN only. The question at this stage is whether the "
          "framework has an\nempirical signal, not what holds for fair GNNs in "
          "general.")


if __name__ == "__main__":
    main()

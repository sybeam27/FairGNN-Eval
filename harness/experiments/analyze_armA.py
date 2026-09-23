"""Arm A cross-dataset analysis.

One audit horizon, H = 200, for every (method, dataset). Nothing here is a
published protocol and nothing is labelled as one. Every tau_int is an effect
conditional on H = 200.

    tau_base^audit = Y(M0) - Y(B)
    tau_int        = Y(M1) - Y(M0)
    tau_pkg^audit  = Y(M1) - Y(B)

read at one sigma_c from one execution context per cell. The residual of
tau_pkg^audit - (tau_base^audit + tau_int) is a join/alignment check, not a
finding: it holds for any common subtrahend.

Order: contract -> raw split x run per (method, dataset) -> per split ->
per run -> selector sensitivity -> per dataset -> cross-dataset. Means last.
Coordinates [dAUC, -dDP] primary, [dAUC, -dEO] secondary; larger is better.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness"))

SIGN_MIN = 0.75
NEAR_ZERO = 0.010
DEGEN = 1e-4          # same epoch for both arms and an effect below this


def fmt(v, w=4):
    return "[" + ", ".join(f"{x:+.{w}f}" for x in np.atleast_1d(v)) + "]"


def sign_stability(x):
    x = np.asarray(x, float)
    nz = x[x != 0.0]
    return 0.0 if nz.size == 0 else max((nz > 0).mean(), (nz < 0).mean())


def build(paths):
    ds = []
    legacy = 0
    for p in paths:
        d = pd.read_csv(p)
        # FairVGNN rows written before the RNG contract (X9-X11) ran with seed 1
        # in every run and restored checkpoints without the CUDA generator. They
        # belong to the fixed-seed / incomplete-CUDA-RNG diagnostic arm and never
        # enter the primary analysis. A row predates the contract iff its file
        # has no rng_contract column.
        if "rng_contract" not in d.columns:
            k = d.method == "FairVGNN"
            legacy += int(k.sum())
            d = d[~k]
        ds.append(d)
    d = pd.concat(ds, ignore_index=True)
    key = ["method", "dataset", "split_id", "run_id", "selector"]
    dup = d.duplicated(key, keep=False)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} rows share a (method, dataset, split, "
                         "run, selector) key across inputs; refusing to pick one")
    if legacy:
        print(f"excluded {legacy} legacy FairVGNN rows "
              "(fixed-seed / incomplete-CUDA-RNG diagnostic arm)")
    for tag, m in (("apkg", "m1"), ("abase", "m0")):
        d[f"{tag}_auc"] = d[f"{m}_auc"] - d.bc_auc
        d[f"{tag}_ndp"] = -(d[f"{m}_dp"] - d.bc_dp)
        d[f"{tag}_neo"] = -(d[f"{m}_eo"] - d.bc_eo)
    d["int_auc"] = d.m1_auc - d.m0_auc
    d["int_ndp"] = -(d.m1_dp - d.m0_dp)
    d["int_neo"] = -(d.m1_eo - d.m0_eo)
    d["pub_auc"] = d.m1pub_auc - d.b_auc
    d["pub_ndp"] = -(d.m1pub_dp - d.b_dp)
    d["degenerate"] = ((d.m1_epoch == d.m0_epoch)
                       & (d.int_auc.abs() < DEGEN) & (d.int_ndp.abs() < DEGEN))
    return d


def contract(d):
    print("=" * 104)
    print("[0] contract")
    print("=" * 104)
    ra = (d.apkg_auc - (d.abase_auc + d.int_auc)).abs().max()
    rn = (d.apkg_ndp - (d.abase_ndp + d.int_ndp)).abs().max()
    re_ = (d.apkg_neo - (d.abase_neo + d.int_neo)).abs().max()
    print(f"  alignment residual  dAUC {ra:.1e}  -dDP {rn:.1e}  -dEO {re_:.1e}"
          f"   {'OK' if max(ra, rn, re_) < 1e-9 else 'BROKEN'}")
    print("  (a join check, not a finding -- it holds for any common subtrahend)")
    g = d.groupby(["method", "dataset", "split_id", "run_id"])[
        ["b_auc", "b_dp", "m1pub_auc", "pub_auc", "pub_ndp"]]
    sp = (g.max() - g.min()).max().max()
    print(f"  published view invariant across sigma_c slots: {sp:.1e} "
          f"{'OK' if sp < 1e-12 else 'BROKEN'}")
    nf = int((~np.isfinite(d[["m1_auc", "m0_auc", "bc_auc", "m1_dp", "m0_dp",
                              "bc_dp"]].to_numpy())).sum())
    print(f"  non-finite outcomes {nf}   EO undefined {int((d.eo_defined==0).sum())}"
          f" of {len(d)} rows")
    print("\n  cells per (dataset, method, selector), and degenerate contrasts")
    t = d.groupby(["dataset", "method", "selector"]).agg(
        n=("degenerate", "size"), degenerate=("degenerate", "sum"))
    print(t[t.degenerate > 0].to_string() if t.degenerate.sum()
          else "    no degenerate contrasts")
    print(f"  total degenerate: {int(d.degenerate.sum())} of {len(d)} rows -- "
          "counted, kept in the data, never read as 'no effect'")
    for ds_ in sorted(d.dataset.unique()):
        h = d[d.dataset == ds_]
        print(f"  {ds_:<8} splits {sorted(h.split_id.unique())} x "
              f"{h.run_id.nunique()} runs x {h.method.nunique()} methods "
              f"| provenance {sorted(h.provenance.unique())}")


def per_cell(d, sel, fh):
    """Raw split x run, written in full to the report file."""
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d.method.unique()):
            g = d[(d.dataset == ds_) & (d.method == m) & (d.selector == sel)]
            if g.empty:
                continue
            g = g.sort_values(["split_id", "run_id"])
            print(f"\n--- {m} / {ds_} / {sel}  raw paired effects, "
                  f"[dAUC, -dDP]", file=fh)
            print(f"{'split':>6}{'run':>4}{'epB':>5}{'epM0':>6}{'epM1':>6}   "
                  f"{'tau_base':>21}{'tau_int':>21}{'tau_pkg':>21}  deg", file=fh)
            for _, r in g.iterrows():
                print(f"{r.split_id:>6}{r.run_id:>4}{r.bc_epoch:>5}"
                      f"{r.m0_epoch:>6}{r.m1_epoch:>6}   "
                      f"{fmt([r.abase_auc, r.abase_ndp]):>21}"
                      f"{fmt([r.int_auc, r.int_ndp]):>21}"
                      f"{fmt([r.apkg_auc, r.apkg_ndp]):>21}"
                      f"{'  *' if r.degenerate else ''}", file=fh)


def per_split_run(d, sel):
    print("\n" + "=" * 104)
    print(f"[2] per split, then per run -- sigma_c = {sel}, [dAUC, -dDP]")
    print("=" * 104)
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d.method.unique()):
            g = d[(d.dataset == ds_) & (d.method == m) & (d.selector == sel)]
            if g.empty:
                continue
            print(f"\n  {m} / {ds_}")
            print(f"{'split':>8}   {'tau_base':>21}{'tau_int':>21}{'tau_pkg':>21}")
            for s in sorted(g.split_id.unique()):
                h = g[g.split_id == s]
                print(f"{s:>8}   {fmt(h[['abase_auc','abase_ndp']].mean()):>21}"
                      f"{fmt(h[['int_auc','int_ndp']].mean()):>21}"
                      f"{fmt(h[['apkg_auc','apkg_ndp']].mean()):>21}")
            print(f"{'run':>8}   {'tau_base':>21}{'tau_int':>21}{'tau_pkg':>21}")
            for r_ in sorted(g.run_id.unique()):
                h = g[g.run_id == r_]
                print(f"{r_:>8}   {fmt(h[['abase_auc','abase_ndp']].mean()):>21}"
                      f"{fmt(h[['int_auc','int_ndp']].mean()):>21}"
                      f"{fmt(h[['apkg_auc','apkg_ndp']].mean()):>21}")


def selector_sensitivity(d):
    print("\n" + "=" * 104)
    print("[3] selector sensitivity, paired within (method, dataset, split, run)")
    print("=" * 104)
    w = d.pivot_table(index=["dataset", "method", "split_id", "run_id"],
                      columns="selector",
                      values=["int_auc", "int_ndp", "m1_auc", "m1_dp", "m0_auc",
                              "m0_dp", "bc_auc", "bc_dp", "bc_epoch", "m1_epoch"])
    print("\n  D = tau_int^BCE - tau_int^AUC")
    print(f"{'dataset':<9}{'method':<10}{'n':>4}{'mean D':>23}{'sd D':>23}"
          f"{'sign':>18}{'|D|>|int^BCE|':>15}")
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d.method.unique()):
            try:
                h = w.loc[(ds_, m)]
            except KeyError:
                continue
            da = (h[("int_auc", "common_bce")] - h[("int_auc", "common_auc")]).to_numpy()
            dn = (h[("int_ndp", "common_bce")] - h[("int_ndp", "common_auc")]).to_numpy()
            ib = h[("int_ndp", "common_bce")].to_numpy()
            frac = float((np.abs(dn) > np.abs(ib)).mean())
            print(f"{ds_:<9}{m:<10}{len(da):>4}{fmt([da.mean(), dn.mean()]):>23}"
                  f"{fmt([da.std(ddof=1), dn.std(ddof=1)]):>23}"
                  f"{fmt([sign_stability(da), sign_stability(dn)]):>18}{frac:>15.2f}")
    print("\n  does the selector move the components themselves? "
          "value^BCE - value^AUC")
    print("  B is the same baseline for every method in a dataset")
    print(f"{'dataset':<9}{'method':<10}{'comp':<5}{'mean dAUC':>12}{'mean dDP':>12}"
          f"{'mean |dEpoch|':>15}")
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d.method.unique()):
            try:
                h = w.loc[(ds_, m)]
            except KeyError:
                continue
            for comp, pre in (("B", "bc"), ("M0", "m0"), ("M1", "m1")):
                da = (h[(f"{pre}_auc", "common_bce")] - h[(f"{pre}_auc", "common_auc")]).mean()
                dd = (h[(f"{pre}_dp", "common_bce")] - h[(f"{pre}_dp", "common_auc")]).mean()
                ep = "bc_epoch" if comp == "B" else "m1_epoch"
                de = (h[(ep, "common_bce")] - h[(ep, "common_auc")]).abs().mean()
                print(f"{ds_:<9}{m:<10}{comp:<5}{da:>+12.4f}{dd:>+12.4f}{de:>15.1f}")


def per_dataset(d, sel):
    print("\n" + "=" * 104)
    print(f"[4] per dataset -- sigma_c = {sel}")
    print("=" * 104)
    print(f"{'dataset':<9}{'method':<10}{'tau_base':>21}{'tau_int':>21}"
          f"{'tau_pkg^audit':>21}{'|base|>|int|':>13}{'sgn int':>9}")
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d.method.unique()):
            g = d[(d.dataset == ds_) & (d.method == m) & (d.selector == sel)]
            if g.empty:
                continue
            b = g[["abase_auc", "abase_ndp"]].mean().to_numpy()
            i = g[["int_auc", "int_ndp"]].mean().to_numpy()
            k = g[["apkg_auc", "apkg_ndp"]].mean().to_numpy()
            print(f"{ds_:<9}{m:<10}{fmt(b):>21}{fmt(i):>21}{fmt(k):>21}"
                  f"{('yes' if abs(b[1]) > abs(i[1]) else 'no'):>13}"
                  f"{sign_stability(g.int_ndp):>9.2f}")


def cross(d, sel):
    print("\n" + "=" * 104)
    print("[5] cross-dataset -- the three questions")
    print("=" * 104)
    rows = []
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d.method.unique()):
            g = d[(d.dataset == ds_) & (d.method == m) & (d.selector == sel)]
            if g.empty:
                continue
            rows.append(dict(dataset=ds_, method=m,
                             base=g.abase_ndp.mean(), int_=g.int_ndp.mean(),
                             pkg=g.apkg_ndp.mean(),
                             base_a=g.abase_auc.mean(), int_a=g.int_auc.mean(),
                             sgn=sign_stability(g.int_ndp),
                             sd=g.int_ndp.std(ddof=1)))
    t = pd.DataFrame(rows)
    n_big = int((t.base.abs() > t.int_.abs()).sum())
    print(f"\n  J1  |tau_base| > |tau_int| on the fairness axis in "
          f"{n_big} of {len(t)} (method, dataset) cells")
    for ds_ in sorted(t.dataset.unique()):
        h = t[t.dataset == ds_]
        print(f"      {ds_:<8} {int((h.base.abs() > h.int_.abs()).sum())}/{len(h)}"
              f"   median |base|/|int| = "
              f"{np.median(h.base.abs() / np.maximum(h.int_.abs(), 1e-9)):.1f}x")

    print("\n  J2  variance of (tau_pkg^audit - tau_int) explained by dataset "
          "vs by method")
    t["gap"] = t.pkg - t.int_
    vd = t.groupby("dataset").gap.mean().std(ddof=1)
    vm = t.groupby("method").gap.mean().std(ddof=1)
    print(f"      sd across dataset means = {vd:.4f}")
    print(f"      sd across method  means = {vm:.4f}")
    print(f"      -> {'dataset' if vd > vm else 'method'} dominates "
          f"(ratio {vd/max(vm,1e-12):.2f})")

    print("\n  J3  selector sensitivity vs the intervention effect it perturbs")
    w = d.pivot_table(index=["dataset", "method", "split_id", "run_id"],
                      columns="selector", values=["int_ndp"])
    for ds_ in sorted(d.dataset.unique()):
        fr, md = [], []
        for m in sorted(d.method.unique()):
            try:
                h = w.loc[(ds_, m)]
            except KeyError:
                continue
            dn = (h[("int_ndp", "common_bce")] - h[("int_ndp", "common_auc")]).to_numpy()
            ib = h[("int_ndp", "common_bce")].to_numpy()
            fr.append(float((np.abs(dn) > np.abs(ib)).mean()))
            md.append(np.median(np.abs(dn)) / max(np.median(np.abs(ib)), 1e-9))
        print(f"      {ds_:<8} |D|>|tau_int| in {np.mean(fr)*100:4.0f}% of cells "
              f"(per method {['%.2f' % f for f in fr]}), "
              f"median |D|/|tau_int| = {np.mean(md):.2f}x")

    print("\n" + "=" * 104)
    print("Overall means -- last")
    print("=" * 104)
    print(f"{'method':<10}{'tau_base':>21}{'tau_int':>21}{'tau_pkg^audit':>21}")
    for m in sorted(d.method.unique()):
        g = d[(d.method == m) & (d.selector == sel)]
        print(f"{m:<10}{fmt(g[['abase_auc','abase_ndp']].mean()):>21}"
              f"{fmt(g[['int_auc','int_ndp']].mean()):>21}"
              f"{fmt(g[['apkg_auc','apkg_ndp']].mean()):>21}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--selector", default="common_bce")
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    d = build(a.csv)
    print(f"Arm A -- controlled audit arm, H = 200 for every cell. "
          f"{len(d)} rows.\nNot a published protocol; every tau_int is "
          f"conditional on H = 200.\n")
    contract(d)
    if a.raw:
        with open(a.raw, "w") as fh:
            per_cell(d, a.selector, fh)
        print(f"\n  raw split x run tables written to {a.raw}")
    per_split_run(d, a.selector)
    selector_sensitivity(d)
    per_dataset(d, a.selector)
    cross(d, a.selector)


if __name__ == "__main__":
    main()

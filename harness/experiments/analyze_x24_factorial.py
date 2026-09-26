"""X24: NIFTY/German 2x2 protocol-factor decomposition, as pre-registered (c219c58).

Inputs are the four protocol cells:

    P00  H=200,  D0   NIFTY rows of harness/results/armA_german.csv   (frozen, reused)
    P10  H=1000, D0   new
    P01  H=200,  D1   new
    P11  H=1000, D1   harness/results/armB_native_NIFTY_german.csv    (frozen, reused)

D is the NIFTY augmentation / validation-view configuration: a bundled,
protocol-level contrast (X24 §2), never "dropout".

For each selector the script builds one table with one row per (split, run)
(30 rows). Its columns are the four cells' τ_int on each coordinate plus the
cell-level contrasts. The table goes through the frozen `bootstrap_armA.boot()`,
so every protocol shares the resampling indices. The frozen X22/X23 resolved
rule is applied unchanged.

Nothing frozen is read for anything but these rows, and nothing frozen is
written.

    python harness/experiments/analyze_x24_factorial.py --p00 ... --p10 ... --p01 ... --p11 ...
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_armA import sign_stability                               # noqa: E402
from bootstrap_armA import boot                                       # noqa: E402

SEED = 20260916
B_REPS = 10_000
SIGN_MIN, NEAR_ZERO = 0.75, 0.010                  # frozen X22/X23 rule
SELS = ("common_bce", "common_auc")
COORDS = ("ndp", "neo", "auc")                     # -dDP primary, -dEO secondary, dAUC
COORD_LABEL = {"ndp": "-dDP", "neo": "-dEO", "auc": "dAUC"}
INT_COL = {"ndp": "int_ndp", "neo": "int_neo", "auc": "int_dauc"}   # pipeline column names
CELLS = {"P00": (200, "D0"), "P10": (1000, "D0"), "P01": (200, "D1"), "P11": (1000, "D1")}
EXPECT = {"P00": (None, None), "P10": ("armA", 1000),
          "P01": ("native", 200), "P11": ("native", 1000)}
CONTRASTS = ("total", "H", "D", "I", "Hs0", "Hs1", "Ds0", "Ds1")
CONTRAST_LABEL = {"total": "Δ_total", "H": "Δ_H", "D": "Δ_D", "I": "I_HD",
                  "Hs0": "H | D0 (v10-v00)", "Hs1": "H | D1 (v11-v01)",
                  "Ds0": "D | H200 (v01-v00)", "Ds1": "D | H1000 (v11-v10)"}
TOL = 1e-12


def load_cell(name: str, path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[(d.method == "NIFTY") & (d.dataset == "german")].copy()
    H, _ = CELLS[name]
    proto, mep = EXPECT[name]
    errs = []
    if len(d) != 60:
        errs.append(f"{len(d)} rows, expected 60")
    if d.duplicated(["split_id", "run_id", "selector"]).any():
        errs.append("duplicate (split, run, selector)")
    if set(d.split_id) != set(range(20, 26)) or set(d.run_id) != set(range(5)):
        errs.append("design is not splits 20-25 x runs 0-4")
    if (d.groupby(["split_id", "run_id"]).selector.nunique() != 2).any():
        errs.append("a cell lacks one selector")
    if not (d.seed == 27 + d.run_id).all():
        errs.append("seed != 27 + run")
    if not np.isfinite(d[list(INT_COL.values())].to_numpy()).all():
        errs.append("non-finite tau_int")
    if "eo_defined" in d.columns and not d.eo_defined.astype(bool).all():
        errs.append("EO undefined")
    if proto is None:
        if "protocol" in d.columns:
            errs.append("P00 must be the Arm A file (no protocol column)")
    else:
        if "protocol" not in d.columns or set(d.protocol) != {proto}:
            errs.append(f"protocol != {proto}")
        if "method_epochs" not in d.columns or set(d.method_epochs) != {mep}:
            errs.append(f"method_epochs != {mep}")
    ep = d[["m1_epoch", "m0_epoch"]].to_numpy()
    if (ep < 0).any() or (ep > H).any():
        errs.append(f"selected epoch outside [0, {H}]")
    if errs:
        raise SystemExit(f"[contract] {name} ({path}): {errs}")
    return d


def stat(t: pd.DataFrame, cols: list, reps: np.ndarray, col: str) -> dict:
    x = t[col].to_numpy(dtype=float)
    mu = float(x.mean())
    lo, hi = np.percentile(reps[:, cols.index(col)], [2.5, 97.5])
    sg = sign_stability(pd.Series(x))
    sg = float(sg) if sg != "n/a" else float("nan")
    resolved = bool(np.isfinite(sg) and sg >= SIGN_MIN and abs(mu) >= NEAR_ZERO
                    and lo * hi > 0)
    return dict(mean=mu, lo=float(lo), hi=float(hi), sign=sg, resolved=resolved)


def label(S: dict, c: str) -> str:
    tot, H, D, I = (S[f"{q}_{c}"] for q in ("total", "H", "D", "I"))
    if (not tot["resolved"]) or not (H["resolved"] or D["resolved"] or I["resolved"]):
        return "H4 weak/no reproducible factor effect"
    if I["resolved"] and abs(I["mean"]) >= max(abs(H["mean"]), abs(D["mean"])):
        return "H3 interaction-dominant"

    def same(a, b):
        return np.sign(S[a]["mean"]) == np.sign(S[b]["mean"])
    if (H["resolved"] and not D["resolved"] and not I["resolved"]
            and same(f"Hs0_{c}", f"Hs1_{c}")):
        return "H1 horizon-dominant"
    if (D["resolved"] and not H["resolved"] and not I["resolved"]
            and same(f"Ds0_{c}", f"Ds1_{c}")):
        return "H2 configuration-dominant"
    if H["resolved"] and D["resolved"]:
        return "both factors"
    return "mixed/other"


def fmt(s: dict) -> str:
    return (f"{s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] "
            f"s{s['sign']:.2f} {'R' if s['resolved'] else 'u'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in CELLS:
        ap.add_argument(f"--{k.lower()}", required=True)
    ap.add_argument("--summary_csv", default=None,
                    help="optional long-format output (new file only)")
    a = ap.parse_args()

    frames = {k: load_cell(k, getattr(a, k.lower())) for k in CELLS}
    keys = {k: set(map(tuple, d[["split_id", "run_id"]].drop_duplicates().to_numpy()))
            for k, d in frames.items()}
    if len({frozenset(v) for v in keys.values()}) != 1:
        raise SystemExit("[contract] the four cells do not share the same (split, run) set")
    print("contract: 4 cells x 60 rows, same 30 (split, run) cells, seeds 27+run, "
          "finite, EO defined, epochs within [0, H], protocol/method_epochs as registered")

    rng = np.random.default_rng(SEED)
    stats, long = {}, []
    for sel in SELS:
        t = None
        for k, d in frames.items():
            g = (d[d.selector == sel].set_index(["split_id", "run_id"])
                 [[INT_COL[c] for c in COORDS]])
            g.columns = [f"{k}_{c}" for c in COORDS]
            t = g if t is None else t.join(g, how="inner")
        if len(t) != 30:
            raise SystemExit(f"[contract] {sel}: joined table has {len(t)} rows")
        for c in COORDS:
            v = {k: t[f"{k}_{c}"] for k in CELLS}
            t[f"total_{c}"] = v["P11"] - v["P00"]
            t[f"H_{c}"] = 0.5 * ((v["P10"] - v["P00"]) + (v["P11"] - v["P01"]))
            t[f"D_{c}"] = 0.5 * ((v["P01"] - v["P00"]) + (v["P11"] - v["P10"]))
            t[f"I_{c}"] = v["P11"] - v["P10"] - v["P01"] + v["P00"]
            t[f"Hs0_{c}"] = v["P10"] - v["P00"]
            t[f"Hs1_{c}"] = v["P11"] - v["P01"]
            t[f"Ds0_{c}"] = v["P01"] - v["P00"]
            t[f"Ds1_{c}"] = v["P11"] - v["P10"]
        t = t.reset_index()
        cols = [x for x in t.columns if x not in ("split_id", "run_id")]
        reps = boot(t, cols, rng, reps=B_REPS)

        # numerical contract (X24 §7)
        mu = t[cols].mean()
        for c in COORDS:
            ix = {q: cols.index(f"{q}_{c}") for q in CONTRASTS}
            iv = {k: cols.index(f"{k}_{c}") for k in CELLS}
            checks = {
                "total = H + D (means)": abs(mu[f"total_{c}"] - (mu[f"H_{c}"] + mu[f"D_{c}"])),
                "total = v11 - v00 (means)": abs(mu[f"total_{c}"] - (mu[f"P11_{c}"] - mu[f"P00_{c}"])),
                "I = v11-v10-v01+v00 (means)": abs(mu[f"I_{c}"] - (mu[f"P11_{c}"] - mu[f"P10_{c}"]
                                                                   - mu[f"P01_{c}"] + mu[f"P00_{c}"])),
                "total = H + D (every replicate)": float(np.max(np.abs(
                    reps[:, ix["total"]] - (reps[:, ix["H"]] + reps[:, ix["D"]])))),
                "cell-contrast mean = contrast of means (every replicate)": float(np.max(np.abs(
                    reps[:, ix["total"]] - (reps[:, iv["P11"]] - reps[:, iv["P00"]])))),
            }
            bad = {k_: v_ for k_, v_ in checks.items() if not v_ <= TOL}
            if bad:
                raise SystemExit(f"[contract] decomposition check failed ({sel}, {c}): {bad}")
        print(f"numerical decomposition contract ({sel}): all residuals <= {TOL:g}")

        S = {}
        for col in cols:
            S[col] = stat(t, cols, reps, col)
            q, c = col.rsplit("_", 1)
            long.append(dict(selector=sel, quantity=q, coord=c, **S[col]))
        stats[sel] = (t, S)

    # ---- selector / boundary diagnostic (X24 §11) ----
    diag = {}
    for k, d in frames.items():
        H, _ = CELLS[k]
        for sel in SELS:
            g = d[d.selector == sel]
            for arm in ("m1", "m0"):
                e = g[f"{arm}_epoch"].to_numpy()
                diag[(k, sel, arm)] = dict(median=float(np.median(e)), lo=int(e.min()),
                                           hi=int(e.max()), ratio=float(np.median(e / H)),
                                           boundary=float(np.mean(e == H)))

    # ---- Table A ----
    for sel in SELS:
        t, S = stats[sel]
        print("\n" + "=" * 118)
        print(f"Table A. Four protocol cells  (selector {sel}; 95% paired hierarchical bootstrap, "
              f"{B_REPS:,} reps; s = sign stability; R/u = resolved/unresolved)")
        print("=" * 118)
        print(f"{'cell':<5}{'H':>5} {'D':<3}{'tau_int dAUC':>36}{'tau_int -dDP':>38}{'tau_int -dEO':>38}")
        for k, (H, D) in CELLS.items():
            print(f"{k:<5}{H:>5} {D:<3}{fmt(S[f'{k}_auc']):>36}{fmt(S[f'{k}_ndp']):>38}"
                  f"{fmt(S[f'{k}_neo']):>38}")
        print(f"\n  selected epochs ({sel}):  median [min,max]  median epoch/H  boundary rate (epoch == H)")
        for k in CELLS:
            m1, m0 = diag[(k, sel, "m1")], diag[(k, sel, "m0")]
            print(f"  {k}  M1 {m1['median']:7.1f} [{m1['lo']:4d},{m1['hi']:4d}] {m1['ratio']:.2f} "
                  f"{m1['boundary']:.2f}   |   M0 {m0['median']:7.1f} [{m0['lo']:4d},{m0['hi']:4d}] "
                  f"{m0['ratio']:.2f} {m0['boundary']:.2f}")

    # ---- Table B ----
    labels = {}
    for sel in SELS:
        t, S = stats[sel]
        print("\n" + "=" * 118)
        print(f"Table B. Attribution-shift decomposition  (selector {sel})")
        print("=" * 118)
        for c in COORDS:
            labels[(sel, c)] = label(S, c)
            print(f"\n  {COORD_LABEL[c]}   label: {labels[(sel, c)]}")
            for q in CONTRASTS:
                print(f"    {CONTRAST_LABEL[q]:<22}{fmt(S[f'{q}_{c}']):>44}")

    # ---- Table C ----
    print("\n" + "=" * 118)
    print("Table C. Selector robustness: tau_int under sigma_c^BCE vs sigma_c^AUC "
          "(qualitative change = resolved state or direction differs)")
    print("=" * 118)
    for c in COORDS:
        print(f"\n  {COORD_LABEL[c]}")
        for k, (H, D) in CELLS.items():
            b, u = stats["common_bce"][1][f"{k}_{c}"], stats["common_auc"][1][f"{k}_{c}"]
            changed = (b["resolved"] != u["resolved"]) or (np.sign(b["mean"]) != np.sign(u["mean"]))
            print(f"    {k} H={H:<5}{D}  BCE {fmt(b):>40}   AUC {fmt(u):>40}   changed: {changed}")
        for q in ("total", "H", "D", "I"):
            b, u = stats["common_bce"][1][f"{q}_{c}"], stats["common_auc"][1][f"{q}_{c}"]
            changed = (b["resolved"] != u["resolved"]) or (np.sign(b["mean"]) != np.sign(u["mean"]))
            print(f"    {CONTRAST_LABEL[q]:<13}  BCE {fmt(b):>40}   AUC {fmt(u):>40}   changed: {changed}")

    # ---- H5 diagnostic flag (X24 §10) ----
    print("\n" + "=" * 118)
    print("H5 selector-associated diagnostic flag (never a primary label, never mediation)")
    print("=" * 118)
    for c in COORDS:
        sel_diff = (labels[("common_bce", c)] != labels[("common_auc", c)]
                    or stats["common_bce"][1][f"total_{c}"]["resolved"]
                    != stats["common_auc"][1][f"total_{c}"]["resolved"])
        ep_diff = any(len({diag[(k, sel, arm)][fld] for k in CELLS}) > 1
                      for sel in SELS for arm in ("m1", "m0") for fld in ("boundary", "ratio"))
        print(f"  {COORD_LABEL[c]:<5} label/resolved differs across selectors: {sel_diff}; "
              f"boundary rate or median epoch/H differs across cells: {ep_diff}; "
              f"H5 flagged: {bool(sel_diff and ep_diff)}")

    if a.summary_csv:
        if os.path.exists(a.summary_csv):
            raise SystemExit(f"refusing to overwrite {a.summary_csv}")
        pd.DataFrame(long).to_csv(a.summary_csv, index=False)
        print(f"\n[written] {a.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

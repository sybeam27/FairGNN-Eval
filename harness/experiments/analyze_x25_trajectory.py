"""X25 analysis: NIFTY/German trajectory–selection decomposition (pre-registered 2547bf3).

Inputs:
* the two trajectory reruns: R10 (H=1000, D0) and R11 (H=1000, D1), each a
  pilot CSV plus per-arm `.npz` files;
* the frozen H=200 cells P00 and P01 (bridge only);
* the frozen P10 and P11 plus the X24 summary (reproduction gate only).

Every decomposition quantity is computed within the rerun trajectories.
Selectors call the frozen `core.trajectory._argbest` on validation records
only; test scores are read at the selected epoch and nowhere else. One 30-row
table goes through the frozen `bootstrap_armA.boot()` once, and the frozen
resolved rule is applied unchanged.

    python harness/experiments/analyze_x25_trajectory.py --r10_csv ... --r10_traj ... \
        --r11_csv ... --r11_traj ... --outdir harness/results/x25
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, ROOT)
from analyze_armA import sign_stability                               # noqa: E402
from bootstrap_armA import boot                                       # noqa: E402
from core.evaluator import evaluate                                   # noqa: E402
from core.trajectory import _argbest                                  # noqa: E402

SEED = 20260917
B_REPS = 10_000
SIGN_MIN, NEAR_ZERO = 0.75, 0.010                  # frozen rule
GRID = (25, 50, 100, 150, 200, 300, 400, 600, 800, 1000)
LATE, EARLY = (600, 800, 1000), (100, 150, 200)
CAP, FULL = 200, 1000
SIGMAS = ("bce", "auc")
SEL = {"bce": "common_bce", "auc": "common_auc"}
COORDS = ("ndp", "neo", "auc")
LAB = {"ndp": "-dDP", "neo": "-dEO", "auc": "dAUC"}
INT_COL = {"ndp": "int_ndp", "neo": "int_neo", "auc": "int_dauc"}
TOL = 1e-12
PREFIX_TOL = 1e-9
MAX_FLIPS = 4          # amended G3 search depth (X25_G3_AMENDMENT 326f04f)
BOUNDARY_FLIPS = []    # recorded boundary-flip comparisons, reported with the results
SENSITIVITY = [False]  # --sensitivity_replayed: use the replayed value at boundary-flip slots


def noise_bound(noise_csv):
    """Frozen envelope: max(1e-5, 4 x max measured replay spread) -- X14 form.

    Measured independently in x25_replay_noise.py before the amendment was
    written, and never derived from any effect estimate.
    """
    r = pd.read_csv(noise_csv)
    return max(1e-5, 4 * float(r.replay_spread.max())), float(r.replay_spread.max()), len(r)


def explain_flips(stored, y, a, dp_ref, eo_ref, bound):
    """Smallest set of within-envelope sign flips reproducing dp_ref and eo_ref."""
    import itertools
    cand = np.nonzero(np.abs(stored) <= bound)[0]
    cand = cand[np.argsort(np.abs(stored[cand]))]
    for k in range(1, min(MAX_FLIPS, len(cand)) + 1):
        for sub in itertools.combinations(cand.tolist(), k):
            f = stored.copy()
            f[list(sub)] = -f[list(sub)]
            m = evaluate(y, a, raw_score=f, decision="score>0")
            if abs(m["dp"] - dp_ref) <= TOL and abs(m["eo"] - eo_ref) <= TOL:
                return list(sub)
    return None


class _Rec:
    __slots__ = ("epoch", "predictive_loss", "metrics")

    def __init__(self, e, b, u):
        self.epoch, self.predictive_loss, self.metrics = e, b, {"auc": u}


def select_epoch(epochs, val_bce, val_auc, sigma, cap):
    """Selected epoch over support {0..cap}, via the frozen `_argbest` (earliest tie).

    Receives validation arrays only: no test information can enter selection.
    """
    recs = [_Rec(int(e), float(b), float(u))
            for e, b, u in zip(epochs, val_bce, val_auc) if int(e) <= cap]
    if sigma == "bce":
        return _argbest(recs, lambda r: r.predictive_loss, maximise=False)
    if sigma == "auc":
        return _argbest(recs, lambda r: r.metrics["auc"], maximise=True)
    raise ValueError(sigma)


class Arm:
    """One arm's recorded trajectory. Outcomes are evaluated lazily by epoch."""

    def __init__(self, z):
        self.epochs = np.asarray(z["epochs"]).astype(int)
        self.val_bce = np.asarray(z["val_bce"], dtype=float)
        self.val_auc = np.asarray(z["val_auc"], dtype=float)
        self.test_raw = np.asarray(z["test_raw"], dtype=float)
        self.y = np.asarray(z["test_y"]).astype(int)
        self.a = np.asarray(z["test_a"]).astype(int)
        self.slot = {"bce": int(z["slot_bce"]) if "slot_bce" in z else None,
                     "auc": int(z["slot_auc"]) if "slot_auc" in z else None}
        self._cache = {}

    def sel(self, sigma, cap):
        return select_epoch(self.epochs, self.val_bce, self.val_auc, sigma, cap)

    def y_at(self, e):
        e = int(e)
        if e not in self._cache:
            idx = int(np.searchsorted(self.epochs, e))
            if idx >= len(self.epochs) or self.epochs[idx] != e:
                raise KeyError(f"epoch {e} not recorded")
            m = evaluate(self.y, self.a, raw_score=self.test_raw[idx], decision="score>0")
            self._cache[e] = dict(ndp=-m["dp"], neo=-m["eo"], auc=m["auc"],
                                  dp=m["dp"], eo=m["eo"])
        return self._cache[e]


def cell_row(arms, h200, frz_full):
    """All cell-level quantities for one (split, run).

    arms:     {D: (M1 Arm, M0 Arm)} on H=1000 trajectories
    h200:     {(D, sigma): {coord: frozen H=200 tau_int}}
    frz_full: {(D, sigma): {coord: frozen H=1000 tau_int}}
    """
    row, diag = {}, {}
    for D, (m1, m0) in arms.items():
        for sig in SIGMAS:
            sel = {}
            for cap in (CAP, FULL):
                e1, e0 = m1.sel(sig, cap), m0.sel(sig, cap)
                sel[cap] = (e1, e0)
                diag[(D, sig, cap)] = (e1, e0)
                y1, y0 = m1.y_at(e1), m0.y_at(e0)
                for c in COORDS:
                    row[f"{D}.{sig}.c{cap}.{c}"] = y1[c] - y0[c]
            for c in COORDS:
                full, capv, hh = (row[f"{D}.{sig}.c{FULL}.{c}"], row[f"{D}.{sig}.c{CAP}.{c}"],
                                  h200[(D, sig)][c])
                row[f"{D}.{sig}.S.{c}"] = full - capv
                row[f"{D}.{sig}.h200.{c}"] = hh
                row[f"{D}.{sig}.T.{c}"] = capv - hh
                row[f"{D}.{sig}.H.{c}"] = full - hh
                row[f"{D}.{sig}.frz.{c}"] = frz_full[(D, sig)][c]
            if sig == "bce":
                (f1, f0), (c1, c0) = sel[FULL], sel[CAP]
                for c in COORDS:
                    row[f"{D}.tele.common.{c}"] = m1.y_at(c1)[c] - m0.y_at(c0)[c]
                    row[f"{D}.tele.m1ext.{c}"] = m1.y_at(f1)[c] - m1.y_at(c1)[c]
                    row[f"{D}.tele.m0ext.{c}"] = m0.y_at(f0)[c] - m0.y_at(c0)[c]
                    row[f"{D}.tele.full.{c}"] = row[f"{D}.bce.c{FULL}.{c}"]
        for t in GRID:
            y1, y0 = m1.y_at(t), m0.y_at(t)
            for c in COORDS:
                row[f"{D}.fx{t}.{c}"] = y1[c] - y0[c]
        for c in COORDS:
            row[f"{D}.L.{c}"] = (np.mean([row[f"{D}.fx{t}.{c}"] for t in LATE])
                                 - np.mean([row[f"{D}.fx{t}.{c}"] for t in EARLY]))
    return row, diag


def identity_residuals(get, Ds=("D0", "D1")):
    """Max absolute residuals of every pre-registered identity; `get(col)` -> array."""
    res = {}
    for D in Ds:
        for sig in SIGMAS:
            for c in COORDS:
                g = lambda q: np.asarray(get(f"{D}.{sig}.{q}.{c}"), dtype=float)  # noqa: E731
                res[f"{D}.{sig}.{c}: H = S + T"] = float(np.max(np.abs(g("H") - (g("S") + g("T")))))
                res[f"{D}.{sig}.{c}: S = c1000 - c200"] = float(np.max(np.abs(
                    g("S") - (g(f"c{FULL}") - g(f"c{CAP}")))))
        for c in COORDS:
            t = lambda q: np.asarray(get(f"{D}.tele.{q}.{c}"), dtype=float)  # noqa: E731
            res[f"{D}.{c}: full = common + m1ext - m0ext"] = float(np.max(np.abs(
                t("full") - (t("common") + t("m1ext") - t("m0ext")))))
            res[f"{D}.{c}: common = bce.c200"] = float(np.max(np.abs(
                t("common") - np.asarray(get(f"{D}.bce.c{CAP}.{c}"), dtype=float))))
    return res


def stat(values, rep_col):
    x = np.asarray(values, dtype=float)
    mu = float(x.mean())
    lo, hi = np.percentile(rep_col, [2.5, 97.5])
    sg = sign_stability(pd.Series(x))
    sg = float(sg) if sg != "n/a" else float("nan")
    res = bool(np.isfinite(sg) and sg >= SIGN_MIN and abs(mu) >= NEAR_ZERO and lo * hi > 0)
    return dict(mean=mu, lo=float(lo), hi=float(hi), sign=sg, resolved=res)


def cases(S_, T_, H_, L_):
    out = []
    if S_["resolved"] and np.sign(S_["mean"]) == np.sign(H_["mean"]) and not T_["resolved"]:
        out.append("A")
    if T_["resolved"]:
        out.append("B")
    if L_["resolved"] and L_["mean"] < 0 and not S_["resolved"]:
        out.append("C")
    if (not L_["resolved"]) and S_["resolved"]:
        out.append("D")
    if L_["resolved"] and S_["resolved"]:
        out.append("E")
    return out or ["none"]


def fmt(s):
    return (f"{s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] s{s['sign']:.2f} "
            f"{'R' if s['resolved'] else 'u'}")


def _nifty(path):
    d = pd.read_csv(path)
    return d[(d.method == "NIFTY") & (d.dataset == "german")].copy()


def csv_contract(tag, d, proto, H):
    errs = []
    if len(d) != 60:
        errs.append(f"{len(d)} rows")
    if d.duplicated(["split_id", "run_id", "selector"]).any():
        errs.append("duplicate keys")
    if set(d.split_id) != set(range(20, 26)) or set(d.run_id) != set(range(5)):
        errs.append("design")
    if (d.groupby(["split_id", "run_id"]).selector.nunique() != 2).any():
        errs.append("selectors")
    if not (d.seed == 27 + d.run_id).all():
        errs.append("seed")
    if proto is not None:
        if "protocol" not in d or set(d.protocol) != {proto}:
            errs.append("protocol")
        if "method_epochs" not in d or set(d.method_epochs) != {H}:
            errs.append("method_epochs")
        if set(d.b_epochs) != {200}:
            errs.append("b_epochs")
    if set(d.feature_normalize) != {0}:
        errs.append("feature_normalize")
    if not np.isfinite(d[list(INT_COL.values())].to_numpy()).all():
        errs.append("non-finite")
    if errs:
        raise SystemExit(f"[contract] {tag}: {errs}")


def load_run(tag, csv_path, traj_dir, proto, bound):
    d = _nifty(csv_path)
    csv_contract(tag, d, proto, FULL)
    arms, ids = {}, {}
    for (sp, rn), g in d.groupby(["split_id", "run_id"]):
        sp, rn = int(sp), int(rn)
        seed, pair, raw = 27 + rn, {}, {}
        for arm in ("M1", "M0"):
            p = os.path.join(traj_dir, f"german_s{sp}_seed{seed}_{arm}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"[contract] {tag}: missing trajectory {p}")
            z = dict(np.load(p, allow_pickle=False))
            ep = np.asarray(z["epochs"]).astype(int)
            bad = []
            if not np.array_equal(ep, np.arange(FULL + 1)):
                bad.append("epochs not 0..1000 (trajectory incomplete)")
            if not np.array_equal(np.asarray(z["test_epochs"]).astype(int), ep):
                bad.append("test epochs != validation epochs")
            if int(z["horizon"]) != FULL or int(z["split"]) != sp or int(z["seed"]) != seed:
                bad.append("identity")
            for k in ("val_raw", "val_bce", "test_raw"):
                if not np.isfinite(np.asarray(z[k], dtype=float)).all():
                    bad.append(f"non-finite {k}")
            if bad:
                raise SystemExit(f"[contract] {tag} {p}: {bad}")
            pair[arm], raw[arm] = Arm(z), z
        # node alignment within the cell and across the split
        for k in ("test_node_id", "test_y", "test_a", "val_node_id", "val_y", "val_a"):
            if not np.array_equal(raw["M1"][k], raw["M0"][k]):
                raise SystemExit(f"[contract] {tag} s{sp} r{rn}: M1/M0 {k} mismatch (node alignment)")
            if (sp, k) in ids and not np.array_equal(ids[(sp, k)], raw["M1"][k]):
                raise SystemExit(f"[contract] {tag} s{sp}: {k} differs across runs (node alignment)")
            ids[(sp, k)] = raw["M1"][k]
        if set(np.asarray(raw["M1"]["test_node_id"]).tolist()) & set(np.asarray(raw["M1"]["val_node_id"]).tolist()):
            raise SystemExit(f"[contract] {tag} s{sp}: validation and test overlap (test isolation)")
        # G3: full-support replay equals the pipeline slot; stored test outcome equals the restored checkpoint
        for sig in SIGMAS:
            row = g[g.selector == SEL[sig]].iloc[0]
            for arm, col in (("M1", "m1"), ("M0", "m0")):
                A = pair[arm]
                e = A.sel(sig, FULL)
                if e != int(row[f"{col}_epoch"]) or e != A.slot[sig]:
                    raise SystemExit(f"[G3] {tag} s{sp} r{rn} {arm} {sig}: replayed epoch {e} != "
                                     f"CSV {int(row[f'{col}_epoch'])} / slot {A.slot[sig]}")
                v = A.y_at(e)
                npos, nneg = int((A.y == 1).sum()), int((A.y == 0).sum())
                ddp, deo = abs(v["dp"] - row[f"{col}_dp"]), abs(v["eo"] - row[f"{col}_eo"])
                if ddp > TOL or deo > TOL:
                    # amended G3 (326f04f): a DP/EO mismatch is admissible only as
                    # sign flips of test nodes inside the frozen replay-noise
                    # envelope that reproduce the restored DP and EO exactly
                    idx = int(np.searchsorted(A.epochs, e))
                    stored = A.test_raw[idx]
                    sub = explain_flips(stored, A.y, A.a, row[f"{col}_dp"], row[f"{col}_eo"], bound)
                    if sub is None:
                        raise SystemExit(f"[G3] {tag} s{sp} r{rn} {arm} {sig}: stored-score DP/EO != "
                                         f"restored checkpoint and not explained by <= {MAX_FLIPS} "
                                         f"sign flips within {bound:.3e} (hard stop)")
                    BOUNDARY_FLIPS.append(dict(run=tag, split=sp, cell_run=rn, arm=arm, selector=sig,
                                               epoch=int(e), n_flips=len(sub),
                                               max_abs_score=float(np.max(np.abs(stored[sub]))),
                                               dDP=ddp, dEO=deo))
                    if SENSITIVITY[0]:
                        # robustness only (X25_G3_AMENDMENT 7): use the restored
                        # checkpoint's own DP/EO/AUC at this slot instead of the
                        # stored in-training score
                        A._cache[int(e)] = dict(ndp=-row[f"{col}_dp"], neo=-row[f"{col}_eo"],
                                                auc=row[f"{col}_auc"], dp=row[f"{col}_dp"],
                                                eo=row[f"{col}_eo"])
                if abs(v["auc"] - row[f"{col}_auc"]) * npos * nneg > 2 + 1e-9:
                    raise SystemExit(f"[G3] {tag} s{sp} r{rn} {arm} {sig}: AUC beyond 2 pair swaps")
        arms[(sp, rn)] = pair
    return d, arms


def lookup(d):
    return {(int(r.split_id), int(r.run_id), r.selector): r for r in d.itertuples()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r10_csv", required=True)
    ap.add_argument("--r10_traj", required=True)
    ap.add_argument("--r11_csv", required=True)
    ap.add_argument("--r11_traj", required=True)
    ap.add_argument("--p00", default=os.path.join(ROOT, "harness/results/armA_german.csv"))
    ap.add_argument("--p01", default=os.path.join(ROOT, "harness/results/x24_nifty_german_P01.csv"))
    ap.add_argument("--p10", default=os.path.join(ROOT, "harness/results/x24_nifty_german_P10.csv"))
    ap.add_argument("--p11", default=os.path.join(ROOT, "harness/results/armB_native_NIFTY_german.csv"))
    ap.add_argument("--x24_summary", default=os.path.join(ROOT, "harness/results/x24_nifty_german_summary.csv"))
    ap.add_argument("--noise_csv", default=os.path.join(ROOT, "harness/results/x25/x25_replay_noise.csv"),
                    help="frozen replay-noise envelope for amended G3")
    ap.add_argument("--sensitivity_replayed", action="store_true",
                    help="robustness: substitute the replayed outcome at boundary-flip slots")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    SENSITIVITY[0] = bool(a.sensitivity_replayed)
    os.makedirs(a.outdir, exist_ok=True)
    outs = {k: os.path.join(a.outdir, f"x25_{k}") for k in
            ("cell_table.csv", "summary.csv", "prefix_cells.csv", "selected_epochs.csv", "gate.txt")}
    for p in outs.values():
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")

    bound, spread, n_slots = noise_bound(a.noise_csv)
    print(f"amended G3 envelope (X25_G3_AMENDMENT 326f04f): max(1e-5, 4 x {spread:.3e}) = {bound:.3e} "
          f"from {n_slots} independent slot evaluations")
    r10, A10 = load_run("R10", a.r10_csv, a.r10_traj, "armA", bound)
    r11, A11 = load_run("R11", a.r11_csv, a.r11_traj, "native", bound)
    fz = {k: _nifty(p) for k, p in (("P00", a.p00), ("P01", a.p01), ("P10", a.p10), ("P11", a.p11))}
    csv_contract("P00", fz["P00"], None, None)
    for k in ("P01", "P10", "P11"):
        csv_contract(k, fz[k], None, None)
    for tag, new, old in (("R10", r10, fz["P10"]), ("R11", r11, fz["P11"])):
        for col in ("protocol", "method_epochs", "b_epochs", "feature_normalize", "provenance"):
            if set(new[col]) != set(old[col]):
                raise SystemExit(f"[config] {tag} {col}: {set(new[col])} != frozen {set(old[col])}")
    print(f"contracts + amended G3 on all 240 comparisons: pass "
          f"({len(BOUNDARY_FLIPS)} boundary-flip, rest exact); original G3 remains FAILED (X25_STOP_REPORT). "
          "Configuration identical to frozen P10/P11")
    for b in BOUNDARY_FLIPS:
        print(f"  boundary-flip: {b['run']} s{b['split']} r{b['cell_run']} {b['arm']} {b['selector']} "
              f"epoch {b['epoch']}: {b['n_flips']} flip(s), max |score| {b['max_abs_score']:.2e}, "
              f"dDP {b['dDP']:.2e}, dEO {b['dEO']:.2e}")

    L = {k: lookup(v) for k, v in fz.items()}
    L["R10"], L["R11"] = lookup(r10), lookup(r11)
    rows, prefix, sel_rows = [], [], []
    for key in sorted(A10):
        sp, rn = key
        h200 = {(D, sig): {c: getattr(L[F][(sp, rn, SEL[sig])], INT_COL[c]) for c in COORDS}
                for D, F in (("D0", "P00"), ("D1", "P01")) for sig in SIGMAS}
        frz = {(D, sig): {c: getattr(L[F][(sp, rn, SEL[sig])], INT_COL[c]) for c in COORDS}
               for D, F in (("D0", "P10"), ("D1", "P11")) for sig in SIGMAS}
        arms = {"D0": (A10[key]["M1"], A10[key]["M0"]), "D1": (A11[key]["M1"], A11[key]["M0"])}
        row, diag = cell_row(arms, h200, frz)
        rows.append(dict(split_id=sp, run_id=rn, **row))
        for (D, sig, cap), (e1, e0) in diag.items():
            sel_rows.append(dict(split_id=sp, run_id=rn, D=D, sigma=sig, cap=cap, m1_epoch=e1, m0_epoch=e0))
        for D, F in (("D0", "P00"), ("D1", "P01")):
            m1, m0 = arms[D]
            for sig in SIGMAS:
                fr = L[F][(sp, rn, SEL[sig])]
                e1, e0 = diag[(D, sig, CAP)]
                v1, v0 = m1.y_at(e1), m0.y_at(e0)
                d_out = max(abs(v1["auc"] - fr.m1_auc), abs(v1["dp"] - fr.m1_dp), abs(v1["eo"] - fr.m1_eo),
                            abs(v0["auc"] - fr.m0_auc), abs(v0["dp"] - fr.m0_dp), abs(v0["eo"] - fr.m0_eo))
                d_dpeo = max(abs(v1["dp"] - fr.m1_dp), abs(v1["eo"] - fr.m1_eo),
                             abs(v0["dp"] - fr.m0_dp), abs(v0["eo"] - fr.m0_eo))
                ep_same = (e1 == int(fr.m1_epoch)) and (e0 == int(fr.m0_epoch))
                prefix.append(dict(split_id=sp, run_id=rn, D=D, sigma=sig, cap_m1=e1, cap_m0=e0,
                                   h200_m1=int(fr.m1_epoch), h200_m0=int(fr.m0_epoch),
                                   epochs_equal=ep_same, max_abs_outcome_diff=d_out,
                                   max_abs_dp_eo_diff=d_dpeo,
                                   exact=bool(ep_same and d_out <= PREFIX_TOL)))
    t = pd.DataFrame(rows)
    cols = [c for c in t.columns if c not in ("split_id", "run_id")]
    rng = np.random.default_rng(SEED)
    reps = boot(t, cols, rng, reps=B_REPS)
    ci = {c: i for i, c in enumerate(cols)}

    res_cells = identity_residuals(lambda c: t[c])
    res_reps = identity_residuals(lambda c: reps[:, ci[c]])
    bad = {k: v for k, v in {**res_cells, **{f"rep {k}": v for k, v in res_reps.items()}}.items() if not v <= TOL}
    if bad:
        raise SystemExit(f"[contract] identity check failed: {bad}")
    print(f"identities (H=S+T, S def, telescoping) hold <= {TOL:g} per cell and in all {B_REPS:,} replicates")

    S = {c: stat(t[c], reps[:, ci[c]]) for c in cols}

    # ---- reproduction gate (X25 §8) ----
    xs = pd.read_csv(a.x24_summary)
    gate_lines, gate_ok = [], True
    for D, F in (("D0", "P10"), ("D1", "P11")):
        for sig in SIGMAS:
            for c in COORDS:
                r = xs[(xs.quantity == F) & (xs.coord == c) & (xs.selector == SEL[sig])].iloc[0]
                m = S[f"{D}.{sig}.c{FULL}.{c}"]["mean"]
                ok = bool(r["lo"] <= m <= r["hi"])
                gate_ok &= ok
                gate_lines.append(f"  {D} ({F}) {sig} {LAB[c]}: rerun {m:+.4f} in frozen "
                                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] (frozen mean {r['mean']:+.4f}): {ok}")
    h10, h11 = S[f"D0.bce.c{FULL}.ndp"], S[f"D1.bce.c{FULL}.ndp"]
    head = (not h10["resolved"]) and h11["resolved"] and h11["mean"] < 0
    gate_ok &= head
    gate_lines.append(f"  headline: R10 BCE -dDP {fmt(h10)} (frozen u); R11 BCE -dDP {fmt(h11)} "
                      f"(frozen R, negative): {head}")
    bit = []
    for tag, F in (("R10", "P10"), ("R11", "P11")):
        n = 0
        for key in sorted(A10):
            same = all(all(getattr(L[tag][(*key, SEL[sig])], k) == getattr(L[F][(*key, SEL[sig])], k)
                           for k in ("m1_epoch", "m0_epoch", "m1_auc", "m1_dp", "m1_eo",
                                     "m0_auc", "m0_dp", "m0_eo")) for sig in SIGMAS)
            n += same
        bit.append(f"  {tag} vs frozen {F}: {n}/30 cells identical in every selected epoch and outcome")
    gate_text = "\n".join(["Reproduction gate (X25 §8):"] + gate_lines + bit +
                          [f"  GATE {'PASS' if gate_ok else 'FAIL'}"])
    print("\n" + gate_text)
    open(outs["gate.txt"], "w").write(gate_text + "\n")
    if not gate_ok:
        print("\nSTOP: reproduction gate failed; no X25 analysis is reported (X25 §8).")
        return 2

    t.to_csv(outs["cell_table.csv"], index=False)
    pd.DataFrame(prefix).to_csv(outs["prefix_cells.csv"], index=False)
    pd.DataFrame(sel_rows).to_csv(outs["selected_epochs.csv"], index=False)
    pd.DataFrame([dict(quantity=c, **S[c]) for c in cols]).to_csv(outs["summary.csv"], index=False)

    # ---- Table 1 ----
    print("\n" + "=" * 120)
    print("Table 1. Selector-support decomposition on the H=1000 rerun trajectory "
          "(tau = mean M1-M0; S = full - cap200; 10,000 paired hierarchical bootstrap)")
    print("=" * 120)
    for c in COORDS:
        print(f"\n  {LAB[c]}")
        for D in ("D0", "D1"):
            for sig in SIGMAS:
                print(f"    {D} {sig.upper():<4} cap<=200 {fmt(S[f'{D}.{sig}.c{CAP}.{c}'])}   "
                      f"full<=1000 {fmt(S[f'{D}.{sig}.c{FULL}.{c}'])}   S {fmt(S[f'{D}.{sig}.S.{c}'])}")

    # ---- Table 2 ----
    pf = pd.DataFrame(prefix)
    print("\n" + "=" * 120)
    print("Table 2. Bridge: frozen H=200 vs H=1000-rerun capped at 200 (T = cap200 - H200; H = full - H200 = S + T)")
    print("=" * 120)
    for c in COORDS:
        print(f"\n  {LAB[c]}")
        for D in ("D0", "D1"):
            for sig in SIGMAS:
                g = pf[(pf.D == D) & (pf.sigma == sig)]
                x24 = (S[f"{D}.{sig}.frz.{c}"]["mean"] - S[f"{D}.{sig}.h200.{c}"]["mean"])
                print(f"    {D} {sig.upper():<4} H200 {S[f'{D}.{sig}.h200.{c}']['mean']:+.4f}  "
                      f"cap200 {S[f'{D}.{sig}.c{CAP}.{c}']['mean']:+.4f}  T {fmt(S[f'{D}.{sig}.T.{c}'])}  "
                      f"H {fmt(S[f'{D}.{sig}.H.{c}'])}  [X24 frozen simple effect {x24:+.4f}]")
                if c == "ndp":
                    print(f"         prefix: epochs equal {int(g.epochs_equal.sum())}/30, "
                          f"epochs+DP/EO equal {int(((g.epochs_equal) & (g.max_abs_dp_eo_diff <= PREFIX_TOL)).sum())}/30, "
                          f"exact (all outcomes <= 1e-9) {int(g.exact.sum())}/30, "
                          f"median max|diff| {g.max_abs_outcome_diff.median():.2e}")
    exact_all = bool(pf.exact.all())
    print(f"\n  exact prefix match (X25 §6, all 30 cells x both D x both sigma): {exact_all}")

    # ---- Table 3 ----
    print("\n" + "=" * 120)
    print("Table 3. Fixed-epoch intervention trajectory (descriptive; no per-epoch labels)")
    print("=" * 120)
    for D in ("D0", "D1"):
        print(f"\n  {D}   {'epoch':>5}  {'tau_int dAUC':>34}  {'tau_int -dDP':>34}  {'tau_int -dEO':>34}")
        for tt in GRID:
            q = [S[f"{D}.fx{tt}.{c}"] for c in ("auc", "ndp", "neo")]
            print(f"        {tt:>5}  " + "  ".join(
                f"{s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] s{s['sign']:.2f}" for s in q))
        print("        L_D (late 600-1000 minus early 100-200): " +
              "  ".join(f"{LAB[c]} {fmt(S[f'{D}.L.{c}'])}" for c in COORDS))

    # ---- Table 4 ----
    print("\n" + "=" * 120)
    print("Table 4. Telescoping selection decomposition, sigma_c^BCE, common cap 200 (algebraic only)")
    print("=" * 120)
    for c in COORDS:
        print(f"\n  {LAB[c]}")
        for D in ("D0", "D1"):
            print(f"    {D} common-cap {fmt(S[f'{D}.tele.common.{c}'])}  M1 ext {fmt(S[f'{D}.tele.m1ext.{c}'])}  "
                  f"M0 ext {fmt(S[f'{D}.tele.m0ext.{c}'])}  full {fmt(S[f'{D}.tele.full.{c}'])}")

    # ---- selector diagnostics ----
    se = pd.DataFrame(sel_rows)
    print("\n" + "=" * 120)
    print("Selector diagnostics (descriptive, correlational only)")
    print("=" * 120)
    for D, F in (("D0", "P00"), ("D1", "P01")):
        for sig in SIGMAS:
            for cap in (CAP, FULL):
                g = se[(se.D == D) & (se.sigma == sig) & (se.cap == cap)]
                gap = (g.m1_epoch - g.m0_epoch)
                print(f"  {D} {sig.upper():<4} cap {cap:>4}: M1 med {g.m1_epoch.median():6.1f} "
                      f"[{g.m1_epoch.min()},{g.m1_epoch.max()}] ratio {np.median(g.m1_epoch / cap):.2f} "
                      f"boundary {np.mean(g.m1_epoch == cap):.2f} | M0 med {g.m0_epoch.median():6.1f} "
                      f"[{g.m0_epoch.min()},{g.m0_epoch.max()}] ratio {np.median(g.m0_epoch / cap):.2f} "
                      f"boundary {np.mean(g.m0_epoch == cap):.2f} | M1-M0 gap med {gap.median():+.1f}")
            fr = fz[F][fz[F].selector == SEL[sig]]
            print(f"  {D} {sig.upper():<4} frozen {F} (H=200): M1 med {fr.m1_epoch.median():.1f} boundary "
                  f"{np.mean(fr.m1_epoch == 200):.2f} | M0 med {fr.m0_epoch.median():.1f} boundary "
                  f"{np.mean(fr.m0_epoch == 200):.2f}")
        for cap in (CAP, FULL):
            b = se[(se.D == D) & (se.sigma == "bce") & (se.cap == cap)].set_index(["split_id", "run_id"])
            u = se[(se.D == D) & (se.sigma == "auc") & (se.cap == cap)].set_index(["split_id", "run_id"])
            print(f"  {D} cap {cap:>4}: median BCE-AUC selected-epoch difference M1 "
                  f"{(b.m1_epoch - u.m1_epoch).median():+.1f}, M0 {(b.m0_epoch - u.m0_epoch).median():+.1f}")

    # ---- pre-registered cases (X25 §9) ----
    print("\n" + "=" * 120)
    print("Pre-registered interpretation cases (X25 §9); A/B bridge family, C/D/E trajectory-vs-selection family")
    print("=" * 120)
    for c in COORDS:
        for sig in SIGMAS:
            parts = []
            for D in ("D0", "D1"):
                k = cases(S[f"{D}.{sig}.S.{c}"], S[f"{D}.{sig}.T.{c}"], S[f"{D}.{sig}.H.{c}"], S[f"{D}.L.{c}"])
                parts.append(f"{D}: {','.join(k)}")
            print(f"  {LAB[c]:<5} {sig.upper():<4} " + "   ".join(parts))
    print("\n[written] " + ", ".join(outs.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

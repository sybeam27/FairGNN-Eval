"""Amended numerical-replay G3 check for X25, applied to every stored comparison.

The original G3 (X25 §7) required exact DP/EO agreement between the stored
in-training test scores and the pipeline's restored-checkpoint outcome. It
FAILED, and that failure stands. This script applies the amended rule, whose
envelope comes from an independent replay-noise diagnostic
(`x25_replay_noise.py`) and never from any effect estimate.

Per comparison (cell x arm x selector), all of these must hold:

    epoch      replayed selector epoch == CSV epoch == ValidationHistory slot
    alignment  test node ids, labels and sensitive attributes identical across
               arms of a cell and across runs of a split; validation and test
               disjoint
    finite     no non-finite stored score
    AUC        within the pre-existing 2 pair swaps
    DP/EO      exact (<= 1e-12)  OR  explained by boundary flips:
               A. the differing hard predictions are identified explicitly
               B. every difference is a sign flip of the score
               C. every flipped node lies within the measured noise envelope
                  bound = max(1e-5, 4 x max replay spread)   [X14 form]
               E. flipping exactly those nodes reproduces the CSV DP and EO
                  to <= 1e-12, with nothing unexplained
               F. the flip count and metric impact are recorded

Anything else is a hard stop. No tau, decomposition, bootstrap, table or figure
is computed here.

    python harness/experiments/x25_amended_g3.py --noise_csv ... --r10_csv ... --r10_traj ... \
        --r11_csv ... --r11_traj ... --out /tmp/.../x25_amended_g3.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, ROOT)

import analyze_x25_trajectory as AN                                   # noqa: E402
import x25_trajectory_run as X                                        # noqa: E402
from core.evaluator import evaluate                                   # noqa: E402

EXACT = 1e-12
AUC_SWAPS = 2.0
MAX_FLIPS = 4            # search depth; more unexplained nodes than this is a stop


def envelope(noise_csv):
    r = pd.read_csv(noise_csv)
    spread = float(r.replay_spread.max())
    return max(1e-5, 4 * spread), spread, r


def explain(stored, y, a, dp_ref, eo_ref, bound):
    """Smallest set of within-envelope sign flips reproducing dp_ref and eo_ref."""
    cand = np.nonzero(np.abs(stored) <= bound)[0]
    cand = cand[np.argsort(np.abs(stored[cand]))]
    for k in range(1, min(MAX_FLIPS, len(cand)) + 1):
        for sub in itertools.combinations(cand.tolist(), k):
            f = stored.copy()
            f[list(sub)] = -f[list(sub)]
            m = evaluate(y, a, raw_score=f, decision="score>0")
            if abs(m["dp"] - dp_ref) <= EXACT and abs(m["eo"] - eo_ref) <= EXACT:
                return list(sub), m
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise_csv", required=True)
    ap.add_argument("--r10_csv", required=True)
    ap.add_argument("--r10_traj", required=True)
    ap.add_argument("--r11_csv", required=True)
    ap.add_argument("--r11_traj", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")

    bound, spread, nz = envelope(a.noise_csv)
    print(f"noise envelope from {len(nz)} independent slot evaluations "
          f"({a.noise_csv}):\n  max replay spread {spread:.3e} -> bound = max(1e-5, 4 x spread) "
          f"= {bound:.3e}\n  max stored-vs-replayed {nz.stored_vs_replayed.max():.3e}, "
          f"sign flips observed {int(nz.n_sign_flips.sum())}")

    rows, stops = [], []
    for tag, csv_path, traj_dir in (("R10", a.r10_csv, a.r10_traj), ("R11", a.r11_csv, a.r11_traj)):
        d = pd.read_csv(csv_path)
        d = d[(d.method == "NIFTY") & (d.dataset == "german")]
        ids = {}
        for (sp, rn), g in d.groupby(["split_id", "run_id"]):
            sp, rn, seed = int(sp), int(rn), 27 + int(rn)
            raw = {}
            for arm in ("M1", "M0"):
                p = X.traj_path(traj_dir, "german", sp, seed, arm)
                if not X.valid_traj(p, 1000):
                    stops.append(f"{tag} s{sp} r{rn} {arm}: invalid or incomplete trajectory")
                    continue
                raw[arm] = dict(np.load(p, allow_pickle=False))
            if len(raw) != 2:
                continue
            for k in ("test_node_id", "test_y", "test_a", "val_node_id"):
                if not np.array_equal(raw["M1"][k], raw["M0"][k]):
                    stops.append(f"{tag} s{sp} r{rn}: {k} differs between arms (alignment)")
                if (sp, k) in ids and not np.array_equal(ids[(sp, k)], raw["M1"][k]):
                    stops.append(f"{tag} s{sp}: {k} differs across runs (alignment)")
                ids[(sp, k)] = raw["M1"][k]
            if set(raw["M1"]["test_node_id"].tolist()) & set(raw["M1"]["val_node_id"].tolist()):
                stops.append(f"{tag} s{sp}: validation and test overlap (test isolation)")
            for arm, col in (("M1", "m1"), ("M0", "m0")):
                A = AN.Arm(raw[arm])
                for sig in AN.SIGMAS:
                    row = g[g.selector == AN.SEL[sig]].iloc[0]
                    e = A.sel(sig, 1000)
                    ep_ok = (e == int(row[f"{col}_epoch"]) == A.slot[sig])
                    idx = int(np.searchsorted(A.epochs, e))
                    stored = A.test_raw[idx]
                    finite = bool(np.isfinite(stored).all())
                    v = A.y_at(e)
                    npos, nneg = int((A.y == 1).sum()), int((A.y == 0).sum())
                    swaps = abs(v["auc"] - row[f"{col}_auc"]) * npos * nneg
                    ddp, deo = abs(v["dp"] - row[f"{col}_dp"]), abs(v["eo"] - row[f"{col}_eo"])
                    rec = dict(run=tag, split=sp, cell_run=rn, arm=arm, selector=sig, epoch=e,
                               epoch_ok=ep_ok, finite=finite, auc_swaps=swaps, dDP=ddp, dEO=deo,
                               n_flips=0, max_abs_flip_score=np.nan, verdict="")
                    if not (ep_ok and finite) or swaps > AUC_SWAPS + 1e-9:
                        rec["verdict"] = "STOP"
                        stops.append(f"{tag} s{sp} r{rn} {arm} {sig}: epoch_ok {ep_ok}, finite {finite}, "
                                     f"AUC swaps {swaps:.2f}")
                    elif ddp <= EXACT and deo <= EXACT:
                        rec["verdict"] = "exact"
                    else:
                        sub, m = explain(stored, A.y, A.a, row[f"{col}_dp"], row[f"{col}_eo"], bound)
                        if sub is None:
                            rec["verdict"] = "STOP"
                            stops.append(f"{tag} s{sp} r{rn} {arm} {sig}: DP/EO mismatch "
                                         f"(dDP {ddp:.2e}, dEO {deo:.2e}) not explained by <= {MAX_FLIPS} "
                                         f"sign flips within the envelope {bound:.2e}")
                        else:
                            rec.update(verdict="boundary-flip", n_flips=len(sub),
                                       max_abs_flip_score=float(np.max(np.abs(stored[sub]))))
                            for j in sub:
                                rows.append(dict(rec, flip_node=int(A_node(raw[arm], j)),
                                                 flip_stored=float(stored[j]),
                                                 flip_dist_threshold=float(abs(stored[j]))))
                    rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    checks = out[out.flip_node.isna()] if "flip_node" in out else out
    n = len(checks)
    print(f"\namended G3 over {n} comparisons: exact {int((checks.verdict == 'exact').sum())}, "
          f"boundary-flip {int((checks.verdict == 'boundary-flip').sum())}, "
          f"STOP {int((checks.verdict == 'STOP').sum())}")
    bf = checks[checks.verdict == "boundary-flip"]
    for r in bf.itertuples():
        print(f"  boundary-flip: {r.run} s{r.split} r{r.cell_run} {r.arm} {r.selector} epoch {r.epoch}: "
              f"{r.n_flips} flip(s), max |score| {r.max_abs_flip_score:.2e} <= bound {bound:.2e}, "
              f"dDP {r.dDP:.2e}, dEO {r.dEO:.2e}, AUC swaps {r.auc_swaps:.2f}")
    if stops:
        print(f"\nHARD STOP: {len(stops)} unexplained condition(s):")
        for s in stops[:20]:
            print("  " + s)
        return 2
    print(f"\namended numerical-replay G3: PASS ({n} comparisons; original G3 remains FAILED and recorded)")
    print(f"[written] {a.out}")
    return 0


def A_node(raw, j):
    return int(np.asarray(raw["test_node_id"])[j])


if __name__ == "__main__":
    raise SystemExit(main())

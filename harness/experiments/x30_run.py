"""X30 method-coverage extension runner: B / M-I / M+I per (split, run).

Non-core methods only. The frozen `pilot_tau.py` is imported for its loader,
evaluator, CellStore and B construction contract and is not edited.

Per unit (split, run):

    B      algorithms.GNN at published("GNN", dataset), H epochs, trained here
    M-I    the method with its claimed intervention switched off   (adapter)
    M+I    the method at its repository configuration             (adapter)

Every arm returns the full-graph logits after each training epoch. The
validation slice goes into a `ValidationHistory` (which holds no test data);
σ_c^BCE and σ_c^AUC choose an epoch from it, and only then is the test slice
of that epoch's logits read. Both method arms start from
`seed_all(seed * 1000 + split)`.

Rows follow the pilot_tau schema, so `analyze_armA.build` and the frozen
bootstrap consume them unchanged. `m1pub_*` is M+I at its own code-native
selector replayed on the same trajectory (NaN when that selector chose
nothing); `b_*` is B at its own selector.

    --smoke   admission gates only, outcome-blind: prints no AUC, DP or EO,
              writes no rows.

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/x30_run.py \
        --method FairSIN --encoder GCN --dataset german --splits 20 --runs 1 --smoke
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

from core.published_config import published                      # noqa: E402
from core.trajectory import SplitRef, ValidationHistory, seed_all  # noqa: E402
from pilot_tau import CellStore, STORE_SELECTORS, load, outcome, _split_ref  # noqa: E402

PROTOCOL = "x30"


def _np(t):
    return np.asarray(t.detach().cpu() if torch.is_tensor(t) else t)


def adapter(method):
    if method == "FairSIN":
        from adapters import x30_fairsin as a
        return a
    if method == "EDITS":
        from adapters import x30_edits as a
        return a
    if method == "FairEdit":
        from adapters import x30_fairedit as a
        return a
    if method == "GEAR":
        from adapters import x30_gear as a
        return a
    if method == "SFG":
        from adapters import x31_sfg as a
        return a
    if method == "FnRGNN":
        from adapters import x31_fnrgnn as a
        return a
    if method == "BeMap":
        from adapters import x30_bemap as a
        return a
    if method == "FnRGNN":
        from adapters import x31_fnrgnn as a
        return a
    if method == "BIND":
        from adapters import x30_bind as a
        return a
    raise SystemExit(f"x30 does not know {method!r}")


def method_tag(method, encoder):
    return f"{method}-{encoder}" if encoder else method


def train_B(data, seed, epochs, dev, dataset):
    """B exactly as pilot_tau.main builds it (same constructor, same seed call)."""
    import core.paths  # noqa: F401  (puts the repository root on sys.path)
    from models.algorithms.GNN import GNN
    adj, f, y, itr, iva, ite, s, si = data
    cfg = dict(published("GNN", dataset)["config"]); cfg.pop("feature_normalize", None)
    torch.manual_seed(seed); np.random.seed(seed)
    b = GNN(adj, f, y, itr, iva, ite, s, si, device=dev, **cfg)
    hb = ValidationHistory(_split_ref(y, iva, s), epochs + 1,
                           state_fn=lambda: {k: v.detach().clone()
                                             for k, v in b.state_dict().items()})
    b.fit(epochs=epochs, trajectory=hb)

    def score(state, idx):
        if state is not None:
            b.load_state_dict(state)
        b.eval()
        with torch.no_grad():
            out = b.forwarding_predict(b.forward(b.features.to(dev), b.edge_index.to(dev)))
        return out.squeeze().detach().cpu().numpy()[np.asarray(idx)]
    return b, hb, score


def select(scores, ref_val, iv):
    """σ_c on a stored trajectory: epoch per selector, from validation only."""
    h = ValidationHistory(ref_val, len(scores), state_fn=lambda: None)
    for e, v in enumerate(scores):
        h.add(e, np.asarray(v, dtype=np.float64)[iv])
    return {k: h.slots[k][0] for k in STORE_SELECTORS if k in h.slots}, h


def gate_report(tag, plus, minus, iv, contract_only=False):
    """Outcome-blind checks. Returns (ok, lines, diag).

    contract_only=True is the per-unit contract of a full run: the
    constant-decision check is recorded as a diagnostic and never stops a cell
    (stopping on it would condition on predictions); every other check is a
    hard stop."""
    lines, ok, diag = [], True, {}

    def chk(name, cond, detail="", hard=True):
        nonlocal ok
        if hard:
            ok &= bool(cond)
        lines.append(f"    [{'PASS' if cond else ('FAIL' if hard else 'note')}] {name} {detail}")

    for arm, r in (("M+I", plus), ("M-I", minus)):
        sc = np.stack(r["scores"])
        chk(f"{arm} finite logits", np.isfinite(sc).all(),
            f"(non-finite {int((~np.isfinite(sc)).sum())})")
        v = sc[:, iv]
        const = float(np.mean([(e > 0).all() or (e <= 0).all() for e in v]))
        diag["constsign_" + ("plus" if arm == "M+I" else "minus")] = const
        chk(f"{arm} validation decisions not constant at every epoch", const < 1.0,
            f"(constant-sign epochs {const:.2f})", hard=not contract_only)
        chk(f"{arm} trajectory length", sc.shape[0] == r["epochs"],
            f"({sc.shape[0]} of {r['epochs']})")
    for k, v in plus.get("gate_checks", {}).items():
        chk(k, v[0], v[1])
    for k, v in minus.get("gate_checks", {}).items():
        chk(k, v[0], v[1])
    for k, v in plus.get("pair_checks", lambda m: {})(minus).items():
        chk(k, v[0], v[1])
    return ok, lines, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", type=int, default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--native", action="store_true",
                    help="native protocol: method arms at the adapter's native horizon, "
                         "code-native selector replayed as m1pub; B stays at --epochs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--protocol-name", default=PROTOCOL,
                    help="protocol key stored in every row (x30 cells keep the default)")
    a = ap.parse_args()
    if not a.smoke and not a.out:
        raise SystemExit("--out is required unless --smoke")
    if a.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise SystemExit("X30 runs on GPU 2 only: set CUDA_VISIBLE_DEVICES=2")
    ad = adapter(a.method)
    tag = method_tag(a.method, a.encoder)
    protocol = a.protocol_name + ("native" if a.native else "")
    if a.native and not hasattr(ad, "native_horizon"):
        raise SystemExit(f"{a.method} has no valid native protocol in X30")
    m_epochs = ad.native_horizon(a.dataset, a.encoder) if a.native else a.epochs
    store = None if a.smoke else CellStore(a.out)
    dev = a.device
    all_ok = True

    for split in a.splits:
        for run in range(a.runs):
            seed = a.seed0 + run
            if store and store.is_done(protocol, tag, a.dataset, split, run):
                print(f"  s{split} r{run} {tag}: already persisted, skipped", flush=True)
                continue
            b_cfg = published("GNN", a.dataset)["config"]
            data = load(a.dataset, split, dev,
                        feature_normalize=bool(b_cfg.get("feature_normalize", False)))
            adj, f, y, itr, iva, ite, s, si = data
            iv, it_, itr_np = _np(iva), _np(ite), _np(itr)
            yv, sv = _np(y), _np(s)
            ref_val = SplitRef(node_id=iv, y=yv[iv].astype(int), a=sv[iv].astype(int))

            arms = {}
            for arm in ("plus", "minus"):
                seed_all(seed * 1000 + split)
                r = ad.train_arm(a.dataset, a.encoder, arm, split, seed, m_epochs, device=dev)
                r["epochs"] = m_epochs
                trm, vam, tem = r["masks"]
                for nm, mk, ref in (("train", trm, itr_np), ("val", vam, iv), ("test", tem, it_)):
                    if not np.array_equal(np.where(np.asarray(mk))[0], np.sort(ref)):
                        raise SystemExit(f"[x30] {tag} {arm} {nm} split differs from the "
                                         f"common split on {a.dataset} s{split} (hard stop)")
                arms[arm] = r

            if a.smoke:
                ok, lines, _ = gate_report(tag, arms["plus"], arms["minus"], iv)
                all_ok &= ok
                print(f"  s{split} r{run} {tag}/{a.dataset}: gates {'PASS' if ok else 'FAIL'}")
                print("\n".join(lines))
                print(f"    config {arms['plus']['config']}", flush=True)
                continue

            ok, lines, diag = gate_report(tag, arms["plus"], arms["minus"], iv,
                                          contract_only=True)
            if not ok:
                print("\n".join(lines))
                raise SystemExit(f"[x30] {tag}/{a.dataset} s{split} r{run}: per-unit "
                                 "contract failed (hard stop; nothing persisted)")
            b, hb, b_score = train_B(data, seed, a.epochs, dev, a.dataset)
            base_pub = outcome(y, s, it_, b_score(b.best_state, it_))
            base_c = {sel: (hb.slots[sel][0], outcome(y, s, it_, b_score(hb.slots[sel][1], it_)))
                      for sel in STORE_SELECTORS}
            sel1, _ = select(arms["plus"]["scores"], ref_val, iv)
            sel0, _ = select(arms["minus"]["scores"], ref_val, iv)
            ce = arms["plus"]["code_epoch"]
            nan = dict(auc=float("nan"), dp=float("nan"), eo=float("nan"))
            m1_pub = (outcome(y, s, it_, arms["plus"]["scores"][ce][it_]) if ce >= 0 else nan)
            a_test = sv[it_].astype(int)
            rows = []
            for sel in STORE_SELECTORS:
                e1, e0 = sel1[sel], sel0[sel]
                sc1 = np.asarray(arms["plus"]["scores"][e1])[it_]
                sc0 = np.asarray(arms["minus"]["scores"][e0])[it_]
                m1, m0 = outcome(y, s, it_, sc1), outcome(y, s, it_, sc0)
                flip = (sc1 > 0) != (sc0 > 0)
                n1, n0 = int((a_test == 1).sum()), int((a_test == 0).sum())
                rows.append(dict(
                    method=tag, dataset=a.dataset, backbone=(a.encoder or "native"),
                    split_id=split, run_id=run, seed=seed, selector=sel,
                    provenance=arms["plus"].get("provenance", "repository"),
                    n_flip=int(flip.sum()), n_flip_a1=int((flip & (a_test == 1)).sum()),
                    n_flip_a0=int((flip & (a_test == 0)).sum()), n_test_a1=n1, n_test_a0=n0,
                    dp_min_step=min(1.0 / max(n1, 1), 1.0 / max(n0, 1)),
                    protocol=protocol, method_epochs=m_epochs, b_epochs=a.epochs,
                    eval_rng_seed=-1, rng_contract="stored-trajectory/paired-seed",
                    feature_normalize=-1,
                    m1_epoch=e1, m0_epoch=e0, code_epoch=ce,
                    m1_auc=m1["auc"], m1_dp=m1["dp"], m1_eo=m1["eo"],
                    m0_auc=m0["auc"], m0_dp=m0["dp"], m0_eo=m0["eo"],
                    b_auc=base_pub["auc"], b_dp=base_pub["dp"], b_eo=base_pub["eo"],
                    bc_epoch=base_c[sel][0], bc_auc=base_c[sel][1]["auc"],
                    bc_dp=base_c[sel][1]["dp"], bc_eo=base_c[sel][1]["eo"],
                    m1pub_auc=m1_pub["auc"], m1pub_dp=m1_pub["dp"], m1pub_eo=m1_pub["eo"],
                    int_dauc=m1["auc"] - m0["auc"], int_ndp=-(m1["dp"] - m0["dp"]),
                    int_neo=-(m1["eo"] - m0["eo"]),
                    pkg_dauc=m1_pub["auc"] - base_pub["auc"],
                    pkg_ndp=-(m1_pub["dp"] - base_pub["dp"]),
                    pkg_neo=-(m1_pub["eo"] - base_pub["eo"]),
                    eo_defined=bool(m1["eo_defined"] and m0["eo_defined"]),
                    nonfinite_plus=int(sum((~np.isfinite(v)).sum() for v in arms["plus"]["scores"])),
                    nonfinite_minus=int(sum((~np.isfinite(v)).sum() for v in arms["minus"]["scores"])),
                    constsign_plus=diag["constsign_plus"],
                    constsign_minus=diag["constsign_minus"],
                    config=repr(arms["plus"]["config"])))
            store.append_cell(rows)
            print(f"  s{split} r{run} {tag}/{a.dataset}: persisted "
                  f"(ep M+I={sel1['common_bce']} M-I={sel0['common_bce']} B={base_c['common_bce'][0]})",
                  flush=True)
    if a.smoke:
        print(f"\nSMOKE {tag}/{a.dataset}: {'ALL GATES PASS' if all_ok else 'GATE FAILURE'}")
        return 0 if all_ok else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

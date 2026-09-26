"""Did the frozen runs actually read harness/provenance/, per method x dataset?

`published_config._read` (harness/core/published_config.py:108) returns [] when a provenance file
is absent instead of raising, so a checkout without harness/provenance/ resolves a *different*
configuration silently. Before that line is changed, this script establishes what the frozen bundle
in fact used, by resolving every cell's configuration twice -- once with harness/provenance/ present
and once against an empty directory -- and comparing both against whatever the frozen store recorded.

Three kinds of recorded evidence, in decreasing strength:

  snapshot         the x30/x31 runners store the resolved configuration verbatim in `config`;
                   compared field by field
  horizon          `method_epochs` (native rows carry the resolved native horizon)
  preprocessing    `feature_normalize`, which published()/native_config() also resolve

A cell is CONFIRMED read-provenance when the two resolutions differ and the recorded evidence
matches the with-provenance one; INDISTINGUISHABLE when the two resolutions agree (the cell does
not depend on provenance, so nothing is proven and nothing is at risk); and MISMATCH when the
evidence matches neither, or matches the empty-directory resolution. Any MISMATCH is a stop.

Reads only frozen artifacts. Writes results/phase0_audit/provenance_read_check.csv.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

import core.published_config as PC  # noqa: E402
import build_results as B           # noqa: E402

REAL_PROV = PC.PROV
EMPTY_PROV = tempfile.mkdtemp(prefix="no_provenance_")


def resolve(fn, *args, prov):
    """Call a published_config resolver with PROV pointed at `prov`."""
    old = PC.PROV
    PC.PROV = prov
    try:
        return fn(*args)
    except Exception as e:                      # a resolver that refuses is itself an answer
        return {"__error__": f"{type(e).__name__}: {e}"}
    finally:
        PC.PROV = old


def flat(cfg):
    """the comparable part of a resolved configuration"""
    if not isinstance(cfg, dict):
        return {}
    d = dict(cfg.get("config", {}))
    if cfg.get("horizon") is not None:
        d["__horizon__"] = cfg["horizon"]
    if "__error__" in cfg:
        d["__error__"] = cfg["__error__"]
    return d


def snapshot(s):
    """the stored config snapshot for a cell, if the runner recorded one"""
    v = s.config.dropna().unique()
    out = []
    for x in v:
        try:
            out.append(ast.literal_eval(x))
        except Exception:
            pass
    return out


def main():
    store = B.load_store()
    rows = []
    for (sm, ds, proto), s in store.groupby(["method", "dataset", "protocol"]):
        method, configuration = B.split_name(sm)
        fn = PC.published if proto == "controlled" else getattr(PC, "native_config", PC.published)
        args = (method, ds)
        with_p = flat(resolve(fn, *args, prov=REAL_PROV))
        without_p = flat(resolve(fn, *args, prov=EMPTY_PROV))
        differ = with_p != without_p
        differing_keys = sorted(k for k in set(with_p) | set(without_p)
                                if with_p.get(k) != without_p.get(k))

        snaps = snapshot(s)
        verdict, evidence, detail = "indistinguishable", "none", ""
        if not differ:
            detail = "the two resolutions agree: this cell does not depend on harness/provenance/"
        else:
            # strongest available evidence first
            hit_with = hit_without = 0
            checked = []
            for k in differing_keys:
                # `published(method, dataset)` resolves the cell's DEFAULT configuration, so it is
                # not the right reference for a deliberately varied one (SAGE / GIN / GCNspmm ...).
                if configuration not in ("default", "", "GCN"):
                    continue
                if k == "__horizon__":
                    # the controlled arm fixes H = 200 for every cell by design; only a native row
                    # carries the resolved published horizon
                    if proto != "native":
                        continue
                    rec = sorted(s.method_epochs.dropna().unique()) if "method_epochs" in s else []
                    if len(rec) == 1:
                        checked.append(("method_epochs", rec[0],
                                        with_p.get(k), without_p.get(k)))
                    continue
                if k == "feature_normalize":
                    rec = sorted(s.feature_normalize.dropna().unique())
                    if len(rec) == 1 and rec[0] in (0, 1):
                        checked.append(("feature_normalize", bool(rec[0]),
                                        with_p.get(k), without_p.get(k)))
                    continue
                for sn in snaps:
                    if k in sn:
                        checked.append((f"config[{k}]", sn[k], with_p.get(k), without_p.get(k)))
                        break
            for name, rec, w, wo in checked:
                if rec == w and w != wo:
                    hit_with += 1
                elif rec == wo and w != wo:
                    hit_without += 1
            if checked:
                evidence = "; ".join(f"{n}={r!r} (prov={w!r}, empty={wo!r})"
                                     for n, r, w, wo in checked[:4])
            if hit_without:
                verdict = "MISMATCH"
                detail = "recorded value matches the empty-directory resolution"
            elif hit_with:
                verdict = "confirmed"
                detail = f"{hit_with} recorded field(s) match the provenance resolution only"
            else:
                verdict = "no_recorded_evidence"
                detail = ("the resolutions differ but the runner recorded nothing that "
                          "distinguishes them")
        rows.append(dict(method=method, configuration=configuration, dataset=ds, protocol=proto,
                         store_method=sm, n_rows=len(s),
                         resolutions_differ=differ,
                         differing_keys=",".join(differing_keys),
                         verdict=verdict, evidence=evidence, detail=detail))

    out = pd.DataFrame(rows).sort_values(["verdict", "method", "dataset", "protocol"])
    path = os.path.join(HERE, "provenance_read_check.csv")
    out.to_csv(path, index=False)

    print(f"cells checked: {len(out)}   (real PROV = {REAL_PROV})")
    print(out.verdict.value_counts().to_string())
    print("\nsaved:", path)

    bad = out[out.verdict == "MISMATCH"]
    if len(bad):
        print("\n*** STOP: recorded configuration matches the no-provenance resolution ***")
        print(bad[["method", "configuration", "dataset", "protocol", "evidence"]].to_string(index=False))
        return 1

    print("\nCells that depend on harness/provenance/ and are confirmed to have read it:")
    c = out[out.verdict == "confirmed"]
    for r in c.itertuples():
        print(f"  {r.method}/{r.configuration}/{r.dataset} [{r.protocol}]  keys={r.differing_keys}")
        print(f"      {r.evidence}")

    ne = out[out.verdict == "no_recorded_evidence"]
    if len(ne):
        print("\nCells whose resolution depends on provenance but whose runner recorded nothing "
              "that distinguishes the two (not a mismatch; simply unverifiable from the store):")
        for r in ne.itertuples():
            print(f"  {r.method}/{r.configuration}/{r.dataset} [{r.protocol}]  keys={r.differing_keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

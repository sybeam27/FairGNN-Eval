# T7 — FairGB backbone label: GCN or SAGE?

Audit date 2026-09-24. CPU-only inspection of source files and stored CSVs.
**No GPU was used and no training was re-run.** No frozen file was modified.

## Verdict

**[확인됨] FairGB cells instantiate `SAGE_encoder` (PyG `SAGEConv`, mean
aggregation). `SAGE` is correct. `results/per_unit_metrics.csv.gz` — and every
raw run CSV under `harness/results/` — is mislabelled `GCN`.**

## Evidence chain: what is actually constructed

1. **The adapter never passes an encoder.**
   `harness/experiments/pilot_tau.py:202-209` builds the FairGB model:

   ```python
   m = FairGB()
   m.fit(d, device=device, runs=1, seed=seed, epochs=epochs,
         trajectory=hist,
         **{k: v for k, v in cfg.items()
            if k in ("hidden", "c_lr", "c_wd", "e_lr", "e_wd",
                     "alpha", "eta", "dropout", "encoder", "warmup")},
         use_cal=..., use_cnm=...)
   ```

   `"encoder"` is in the forwarding whitelist, but the config never contains it.
   Verified by executing the frozen interpreters (CPU only):

   ```
   $ python -c "import sys; sys.path.insert(0,'harness'); \
       from core.published_config import published, native_config; ..."
   german published cfg keys: ['alpha','c_lr','e_lr','eta','feature_normalize','hidden']   encoder in cfg: False
   bail   published cfg keys: ['alpha','c_lr','e_lr','feature_normalize','hidden']         encoder in cfg: False
   credit published cfg keys: ['alpha','c_lr','c_wd','e_lr','e_wd','feature_normalize','hidden']  encoder in cfg: False
   ```

   Consistent with the upstream artifact: `models/FairGB-main/run.sh` (all three
   dataset lines) never sets `--encoder`.

2. **So the method's own default applies.**
   `models/algorithms/FairGB_alg.py:47`

   ```python
   dropout=0.5, hidden=16, encoder='SAGE', alpha=1,
   ```

   and `models/algorithms/FairGB_alg.py:71` `args.encoder = encoder`,
   `:98` `encoder_m, classifier, optimizer_e, optimizer_c = get_enc_cls_opt(args)`.

3. **The factory maps that to a GraphSAGE encoder.**
   `models/algorithms/FairGB/utils.py:40-44`

   ```python
   elif(args.encoder == 'SAGE'):
       encoder = SAGE_encoder(args).to(args.device)
   ```

   `models/algorithms/FairGB/models.py:101-107`

   ```python
   class SAGE_encoder(nn.Module):
       ...
       self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
       self.conv1.aggr = 'mean'
   ```

   The `GCN_encoder` branch (`utils.py:31-35`) is never reached for these cells.

## Where the wrong label comes from

**[확인됨] A single hard-coded literal in the run pipeline.**

`harness/experiments/pilot_tau.py:547`, inside the per-cell row constructor:

```python
cell_rows.append(dict(
    method=meth, dataset=a.dataset, backbone="GCN",
    ...
```

`backbone` is a constant string, not derived from `cfg` or from the constructed
module, so every Arm A / Arm B row written by `pilot_tau.py` says `GCN`
regardless of method:

```
$ cut -d, -f3 harness/results/armB_native_FairGB_*.csv harness/results/armA_*.csv | sort | uniq -c
   1080 GCN
```

That column is then copied verbatim into the per-unit file:
`harness/experiments/build_manifests.py:158-163`

```python
keep = ["method", "dataset", "backbone", "protocol", "selector", "split_id", "run_id"]
...
p = store[keep + [a, dp, eo, "eo_defined"]].copy()
```

giving `results/per_unit_metrics.csv.gz` → `FairGB` → `backbone = GCN`
(1,080 rows, verified; controlled + native protocols).

Every other published result file instead goes through
`harness/experiments/build_results.py:132-137` `canonical_backbone()`, which
reads `CANONICAL_BACKBONE` at `build_results.py:117`:

```python
"FairGB": "SAGE",          # FairGB_alg.py:47, its own default encoder
```

Confirmed present as `SAGE` in: `results/1_main_package_vs_intervention.csv`,
`results/3a_…csv`, `results/3b_…csv`, `results/3c_…csv`,
`results/cell_results.csv`, `results/coverage.csv`,
`results/method_configurations.csv`. Also
`harness/experiments/x30_config_table.py:67`
(`CORE_BACKBONE = {..., "FairGB": "SAGE"}   # FairGB_alg.py:47`).

## Which file is mislabelled

| file | FairGB backbone | correct? |
|---|---|---|
| `results/per_unit_metrics.csv.gz` | `GCN` | **NO — mislabelled** |
| `harness/results/armA_*.csv`, `harness/results/armB_native_FairGB_*.csv`, `harness/results/x26/x26_bail.csv` (`backbone` column) | `GCN` | **NO — mislabelled at source** (frozen; not to be edited) |
| `results/method_configurations.csv` and all other `results/*.csv` | `SAGE` | yes |
| paper text | `SAGE` | yes |

## Blast radius

**[확인됨] Label-only. No estimate changes.**

* The `backbone` column is never used to select a model — it is written after
  training, purely as metadata (`pilot_tau.py:547`).
* The per-unit values themselves are correct. Recomputing the headline estimand
  from `per_unit_metrics.csv.gz` per `PAPER_ARTIFACT_MAP.md` ("Recomputing a
  reported number") for FairGB/bail, controlled, `common_bce`, n = 30 units
  gives `tau_I_negDP = -0.05967843`, matching
  `results/1_main_package_vs_intervention.csv` (`-0.059678`) to 5e-6.
* The real risk is a **join/filter failure**: a reader who follows
  `PAPER_ARTIFACT_MAP.md` and joins `per_unit_metrics.csv.gz` to any other
  `results/*.csv` on `(method, dataset, backbone)` gets **zero rows for every
  FairGB cell**, because one side says `SAGE` and the other says `GCN`.
* Scope: FairGB is the only method where the hard-coded `"GCN"` is factually
  wrong. `NIFTY`, `FairGNN`, `FairVGNN` are genuinely GCN; the methods run by
  the `x30_*`/`x31_*` runners record their own backbones correctly
  (`FairSIN-SAGE` → `SAGE`, `FairSIN-GIN` → `GIN`, `BeMap-GAT` → `GAT`), and the
  remaining methods record the placeholder `native`.

## Proposed fix (not applied)

`harness/experiments/pilot_tau.py:547` should record the encoder that was
actually constructed rather than a literal, e.g.

```python
backbone=canonical_backbone(meth, "default"),   # or: str(mcfg.get("encoder", METHOD_DEFAULT_ENCODER[meth]))
```

`harness/results/**` is frozen, so the correction belongs in
`harness/experiments/build_manifests.py:157-172`: map the stored `backbone`
through `build_results.canonical_backbone(method, configuration)` before writing
`results/per_unit_metrics.csv.gz`, which makes the per-unit file join cleanly
with every other published table without touching any frozen run CSV. A one-line
note in `results/README.md` should record that the frozen run CSVs carry the
stale literal.

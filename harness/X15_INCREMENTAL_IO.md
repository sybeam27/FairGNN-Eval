# Cell-level incremental persistence, after the /home disk filled

## What happened

FairGB/bail, the last Arm B Phase 1 cell, ran 06:26-07:49 and trained all 30
cells. The pilot then wrote every row in one call at the end, and that call
failed:

    OSError: [Errno 28] No space left on device

`/home` (shared, 3.0T) was at 100%. The CSV was left at 0 bytes. The log holds
only a one-line BCE summary per cell, which cannot rebuild the rows, so the
cell has to be rerun.

This project did not fill the disk.

* The account's whole footprint was 19G: miniconda 9.6G, pip cache 4.6G,
  vscode 3.1G, workspace about 1G.
* The only file over 200 MB written in the preceding 24 h was the Claude Code
  binary.
* The rest sits in directories this account cannot read.

The repository stayed intact: every Phase 1 commit through `9a71431` was in
place and `git fsck` was clean. No cache was deleted. By the time of the rerun,
`/home` had 374G free again.

## Decision

Rerun FairGB/bail only, writing to `/tmp`. The four other Phase 1 cells are not
rerun. Arm A and the Phase 1 numerical configuration are unchanged.

## The I/O contract, changed; training and evaluation, not

`pilot_tau.py` no longer collects rows and writes them at the end. A
`CellStore`:

* **validates any existing file before running** and refuses to resume or
  overwrite when:
  * the last row is incomplete (no trailing newline);
  * a row has the wrong field count;
  * the header lacks a key column;
  * a `(protocol, method, dataset, split_id, run_id, selector)` key appears
    twice;
  * a cell has only one of its two selector rows;
* **resumes**: a `(method, split, run)` already on disk is skipped, and B is
  skipped too when every method of that `(split, run)` is done;
* **appends each cell the moment both selector rows exist**, then flushes and
  fsyncs. It refuses a duplicate key or an incomplete cell rather than writing
  it. A cell missing a selector slot is not persisted, so it reruns;
* the final summary reads the persisted file instead of an in-memory list.

The diff against the pre-change file touches store and loop-control lines only.
`rows.append` becomes `cell_rows.append` and the bulk `to_csv` is removed. No
training, selection, restoration or evaluation line changes. Skipping a finished
cell does not shift any later cell's randomness: each `(split, run)` reseeds
from `seed` and `seed * 1000 + split` independently.

## Verification before the rerun

`harness/tests/test_incremental_io.py` runs the real pilot as a subprocess. All
pass:

1. a fresh run writes 4 complete rows with unique keys and a trailing newline;
2. an identical rerun trains 0 cells and leaves the file byte-identical;
3. extending `--runs` from 2 to 3 trains exactly 1 cell, giving 6 unique rows;
4. a truncated last line is refused, file untouched;
5. a duplicated key is refused, file untouched;
6. a half-written cell is refused, file untouched.

The first run of the test reported 2 failures in checks 2 and 3. They came from
the test's own counter, which also counted the new "skipped" lines. The file
checks in the same run had already passed. The counter now matches only the
per-cell training line.

`harness/tests/test_native_phase1.py` was rerun on the patched pilot and passed
127/127: native changes in effect, restore within the CUDA noise floor, hard
predictions / DP / EO exact, AUC within resolution, completeness, selector
replay, test isolation, and the Arm A path unchanged.

## Rerun

FairGB/bail, native H = 1500, upstream normalization, isolation-preserving M0,
sigma_c^BCE and sigma_c^AUC, G_c, B_200 reference, splits 20-25, runs 0-4:
`/tmp/.../scratchpad/armB_native_FairGB_bail.csv`, persisted cell by cell.

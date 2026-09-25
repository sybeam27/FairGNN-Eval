#!/usr/bin/env bash
# Fetch the third-party method repositories and the datasets this study runs on.
#
# They are not redistributed here. Six of them ship no licence text at all, so we have no right to
# copy them; the two that do (BeMap, FairGB) are included in this checkout under their MIT terms,
# minus their bundled data. Everything else is fetched from its own source, pinned where a pin
# could be established and left unpinned -- and marked as such -- where it could not.
#
#   bash fetch_upstream.sh            fetch everything into models/ and data/
#   bash fetch_upstream.sh --list     print what would be fetched, and change nothing
#
# After fetching, apply our local edits:
#   bash fetch_upstream.sh --patch
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$HERE/models"
DATA="$HERE/data"
PATCHES="$HERE/patches"

# repo dir | URL | pin | what we had to change
REPOS=(
  "BIND-main|https://github.com/yushundong/BIND|UNPINNED|none"
  "FairSIN-main|https://github.com/Landon5282/FairSIN|UNPINNED|none"
  "FMP-main|https://github.com/zhimengj0326/FMP|UNPINNED|none"
  "GEAR-main|https://github.com/jma712/gear|UNPINNED|none"
  "SFG-main|https://github.com/sh-qiangchen/SFG|UNPINNED|none"
  "FairGT-main|UNKNOWN|UNPINNED|excluded from the study; listed for completeness"
  "FnRGNN-master|ANONYMISED-MIRROR|UNPINNED|none"
)

list_repos() {
  printf '%-16s %-52s %-10s %s\n' REPO URL PIN NOTE
  for r in "${REPOS[@]}"; do
    IFS='|' read -r d u p n <<<"$r"
    printf '%-16s %-52s %-10s %s\n' "$d" "$u" "$p" "$n"
  done
  cat <<'NOTE'

UNPINNED  we did not record a commit hash when the code was vendored, so the pin cannot be
          reconstructed honestly. Fetch the default branch and check it against
          harness/external_repos.tsv and harness/METHOD_EXTENSION_INVENTORY.csv, which record what
          each adapter expects. If upstream has moved, the adapters load files by path and by line
          number and will fail loudly rather than silently run something else.
UNKNOWN   the vendored copy carries a README but no repository URL we could verify. See
          harness/METHOD_EXTENSION_INVENTORY.csv for the source recorded at the time.
          Only FairGT-main is still UNKNOWN; GEAR and SFG were supplied by the author, fetched,
          and checked file by file.

VERIFIED  GEAR and SFG: every file the adapters read is byte-identical to the repository's default
          branch, checked 2026-09-25 at gear 47cf4c198505e7cde24dcf413f1dc60401d7aa05 and SFG
          cff7c5ab023ca8b2d8fb6f1255617ac533e0d957. Those hashes record what the default branch
          held on that date; they are NOT the commits the experiments ran against, which were never
          recorded, so the pin stays UNPINNED.
ANONYMISED-MIRROR
          FnRGNN's upstream identifies an author of this submission, so it is withheld during
          review and will be pointed at the public repository in the camera-ready.

Datasets: German, Bail, Credit, Income, Pokec-z, Pokec-n. Checksums for the exact files we used are
in harness/data_manifest.tsv; every dataset is redistributed by its own source under its own terms.
NOTE
}

patch_repos() {
  [ -d "$PATCHES" ] || { echo "no patches/ directory; nothing to apply"; return 0; }
  for p in "$PATCHES"/*.patch; do
    [ -e "$p" ] || continue
    echo "applying $(basename "$p")"
    git apply --directory=models -p1 "$p" || echo "  FAILED: $(basename "$p")"
  done
}

case "${1:-}" in
  --list)  list_repos; exit 0 ;;
  --patch) patch_repos; exit 0 ;;
esac

mkdir -p "$MODELS" "$DATA"
for r in "${REPOS[@]}"; do
  IFS='|' read -r d u p n <<<"$r"
  case "$u" in
    UNKNOWN|ANONYMISED-MIRROR)
      echo "SKIP $d: $u -- see --list"; continue ;;
  esac
  if [ -d "$MODELS/$d" ]; then echo "have $d"; continue; fi
  echo "cloning $d from $u"
  git clone --depth 1 "$u" "$MODELS/$d" || echo "  FAILED: $d"
done
echo
echo "Datasets are not fetched automatically: each has its own terms. See harness/data_manifest.tsv"
echo "for the checksums of the files this study used, and results/README.md for what each one is."
echo
echo "Now run:  bash fetch_upstream.sh --patch"

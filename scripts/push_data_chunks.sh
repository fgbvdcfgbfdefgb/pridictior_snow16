#!/usr/bin/env bash
# Commits and pushes the Parquet dataset in small batches (~6 months per push), so no single
# upload gets too big. Re-running is safe: already-pushed files are skipped.
# Requires: GITHUB_TOKEN in the environment (never written into the repo or .git/config).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GITHUB_TOKEN:?set GITHUB_TOKEN}"
AUTH=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)
BATCH=${BATCH:-6}
mapfile -t FILES < <(ls data/parquet/BTCUSDT_1s_*.parquet | sort)
n=0; batch=()
for f in "${FILES[@]}"; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then continue; fi
  batch+=("$f"); n=$((n+1))
  if [ "${#batch[@]}" -ge "$BATCH" ]; then
    git add -- "${batch[@]}"
    git commit -q -m "data: add BTCUSDT 1s parquet (${batch[0]##*_} .. ${batch[-1]##*_})"
    git -c http.extraheader="Authorization: Basic $AUTH" push -q origin HEAD:main
    echo "pushed ${#batch[@]} files"; batch=()
  fi
done
if [ "${#batch[@]}" -gt 0 ]; then
  git add -- "${batch[@]}"
  git commit -q -m "data: add BTCUSDT 1s parquet (remaining)"
  git -c http.extraheader="Authorization: Basic $AUTH" push -q origin HEAD:main
  echo "pushed ${#batch[@]} files"
fi
echo "done ($n new files)"

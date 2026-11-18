#!/usr/bin/env bash
set -euo pipefail

destination=${1:-public}
base_url=${2:-https://jaehun.me/}
base_url=${base_url%/}/
build_log=$(mktemp)
trap 'rm -f "$build_log"' EXIT

hugo --gc --minify --environment production --printPathWarnings \
  --baseURL "$base_url" --destination "$destination" 2>&1 | tee "$build_log"

# Hugo warns about duplicate paths but still exits successfully.
if grep -F 'Duplicate target paths' "$build_log"; then
  echo '::error::URL이 겹치는 글이 있습니다.'
  exit 1
fi

python3 -m unittest discover -s scripts -p 'test_check*.py'
python3 scripts/check_seo.py "$destination" --base-url "$base_url"
python3 scripts/check_local_links.py "$destination" --base-url "$base_url"

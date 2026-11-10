#!/usr/bin/env bash
# Publish only a verified Hugo build; the caller requests the Pages build.
set -euo pipefail

source_dir=$(realpath "$1")
published_dir=$(realpath "$2")

if [[ ! -f "$source_dir/index.html" || ! -f "$source_dir/CNAME" ]]; then
  echo 'Expected a Hugo build containing index.html and CNAME.' >&2
  exit 1
fi
if [[ "$(cat "$source_dir/CNAME")" != 'jaehun.me' ]]; then
  echo 'The build must retain the jaehun.me custom domain.' >&2
  exit 1
fi
if [[ ! "${GITHUB_SHA:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'GITHUB_SHA must identify the source commit.' >&2
  exit 1
fi
if [[ "$(git -C "$published_dir" branch --show-current)" != 'gh-pages' ||
      "$(git -C "$published_dir" rev-parse --show-toplevel)" != "$published_dir" ]]; then
  echo 'The destination must be the root of a gh-pages checkout.' >&2
  exit 1
fi
if [[ "$source_dir/" == "$published_dir/"* || "$published_dir/" == "$source_dir/"* ]]; then
  echo 'The build and publication directories must be separate.' >&2
  exit 1
fi
if [[ -n "$(git -C "$published_dir" status --porcelain)" ]]; then
  echo 'The gh-pages checkout must be clean before publication.' >&2
  exit 1
fi

rsync -a --delete --exclude='/.git' "$source_dir/" "$published_dir/"
touch "$published_dir/.nojekyll"
printf '{"commit":"%s"}\n' "$GITHUB_SHA" > "$published_dir/build-info.json"
git -C "$published_dir" config user.name 'github-actions[bot]'
git -C "$published_dir" config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git -C "$published_dir" add --all
if ! git -C "$published_dir" diff --cached --quiet; then
  git -C "$published_dir" commit -m "Publish Hugo site from $GITHUB_SHA"
fi
# A normal push rejects a concurrent publication instead of overwriting it.
git -C "$published_dir" push origin HEAD:refs/heads/gh-pages

#!/usr/bin/env bash
# Commit whatever a workflow produced under Actions/out/ back into the branch.
# Retries with rebase so parallel matrix runners do not overwrite each other.
# Usage: commit_results.sh "<commit message>"
set -euo pipefail

message="${1:-ci: record action output}"
branch="${GITHUB_REF_NAME:?GITHUB_REF_NAME is not set}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git add -A Actions/out Actions/triggers
if git diff --cached --quiet; then
  echo "nothing to commit"
  exit 0
fi

git commit -m "$message"

for attempt in 1 2 3 4 5; do
  if git push origin "HEAD:${branch}"; then
    echo "pushed on attempt ${attempt}"
    exit 0
  fi
  echo "push rejected (attempt ${attempt}), rebasing on origin/${branch}"
  git pull --rebase --autostash origin "${branch}"
done

echo "could not push after 5 attempts" >&2
exit 1

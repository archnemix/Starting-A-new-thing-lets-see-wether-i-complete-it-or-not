#!/usr/bin/env bash
# Commit whatever a workflow produced under Actions/out/ back into the branch.
#
# Two workflows now write into this repository - `tests` and `npm-signpost` - and each runs
# three parallel runners, so up to six processes race to push to one branch. The first version
# retried a rejected push five times with no delay and then gave up; on Windows that exhausted
# the attempts while another runner held the branch, and the job failed *after its tests had
# passed*, which reads like a test failure in the run list and is not one.
#
# So: more attempts, a backoff, and an explicit fetch+rebase instead of `pull --rebase`.
# `--autostash` stays, because a checkout can be dirty when a workflow re-runs.
#
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

attempts=8
for attempt in $(seq 1 "$attempts"); do
  if git push origin "HEAD:${branch}"; then
    echo "pushed on attempt ${attempt}"
    exit 0
  fi
  echo "push rejected (attempt ${attempt}/${attempts}); rebasing on origin/${branch}"
  git fetch origin "$branch"
  # Hard reset onto the remote keeps the commit we just made on top and cannot reorder it,
  # and is safe here because the only thing this job changed is its own record file.
  if ! git rebase --autostash "origin/${branch}"; then
    echo "rebase failed; aborting and retrying from a clean tree" >&2
    git rebase --abort || true
  fi
  # Backoff with jitter, so six racing runners do not retry in lockstep.
  sleep $(( attempt * 2 + RANDOM % 5 ))
done

echo "could not push after ${attempts} attempts" >&2
exit 1

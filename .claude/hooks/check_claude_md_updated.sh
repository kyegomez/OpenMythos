#!/usr/bin/env bash
# Stop hook: enforce that any session touching source / test files also
# appends a new entry to CLAUDE.md's execution log.
#
# Exit 0: allow Claude to stop (no block).
# Exit 2: block and feed the stderr message back so Claude must continue
#         and update CLAUDE.md before returning control to the user.
#
# Source of truth for the rule: CLAUDE.md § "执行进度日志 · 维护规则"

set -uo pipefail

# CLAUDE_PROJECT_DIR is provided by the Claude Code harness; fall back to pwd.
project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$project_dir" || exit 0

# Only relevant inside a git repo.
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# Collect every path that differs from HEAD (staged + unstaged + untracked).
staged=$(git diff --cached --name-only 2>/dev/null || true)
unstaged=$(git diff --name-only 2>/dev/null || true)
untracked=$(git ls-files --others --exclude-standard 2>/dev/null || true)
all_changes=$(printf '%s\n%s\n%s\n' "$staged" "$unstaged" "$untracked" \
  | sort -u | grep -v '^$' || true)

# "Meaningful" = source, tests, training, or package metadata that would
# require a log entry. CLAUDE.md / ANALYSIS_REPORT.md / OPTIMIZATION_PLAN.md
# are excluded so pure-docs changes don't trigger a self-loop.
meaningful=$(printf '%s\n' "$all_changes" | grep -E '^(open_mythos/|training/|tests/|test_main\.py|pyproject\.toml|requirements.*\.txt)' || true)
meaningful=$(printf '%s\n' "$meaningful" | grep -v '^$' || true)

if [ -z "$meaningful" ]; then
  # Nothing substantive changed — pure doc tweaks, research, or conversation.
  exit 0
fi

# If CLAUDE.md is in the working-tree diff, the rule is satisfied.
if printf '%s\n' "$all_changes" | grep -qx 'CLAUDE\.md'; then
  exit 0
fi

# If the last commit already added/updated CLAUDE.md and the meaningful
# changes are a new round built on top of that commit, also pass — we
# don't want to force redundant log entries for a continuation.
last_commit_files=$(git log -1 --name-only --pretty=format: 2>/dev/null | grep -v '^$' || true)
if printf '%s\n' "$last_commit_files" | grep -qx 'CLAUDE\.md'; then
  # Was the last commit MADE in this session (recent) AND are there no new
  # meaningful changes since that commit? Then we're good.
  if [ -z "$meaningful" ]; then
    exit 0
  fi
fi

# Rule violation. Emit a structured reminder on stderr and exit 2 so the
# harness feeds the message back to Claude and blocks the stop.
{
  echo "[Stop hook] CLAUDE.md 执行日志未更新"
  echo ""
  echo "本会话修改了源码 / 测试 / 训练脚本,但 CLAUDE.md 的进度条目未追加新记录。"
  echo "按 CLAUDE.md § 维护规则,这视为工作未交付。"
  echo ""
  echo "变更文件清单:"
  printf '  - %s\n' $meaningful
  echo ""
  echo "请向 CLAUDE.md 顶部"进度条目"区追加一条记录,格式参照文件开头的 '追加条目格式' 模板。"
  echo "追加完成后 Claude 才可 stop。"
} >&2

exit 2

#!/bin/bash
# palinode-hook-version: 0.23.0
# palinode-session-end.sh — Auto-capture Claude Code sessions to Palinode.
#
# Fires on SessionEnd (including /clear, logout, exit). Reads the transcript
# from stdin JSON, extracts a minimal summary, and POSTs to palinode-api.
#
# Fail-silent by design — never block Claude Code exit. Automatic capture is
# gated by the persisted control service; a denial or unavailable service exits
# before reading a transcript and never writes a content-bearing fallback.
#
# Install:
#   1. Copy to .claude/hooks/palinode-session-end.sh (or ~/.claude/hooks/…)
#   2. chmod +x .claude/hooks/palinode-session-end.sh
#   3. Register in .claude/settings.json — see ./settings.json in this dir.
#
# Or just run: `palinode init` — it installs all of this for you.

set -euo pipefail

PALINODE_API="${PALINODE_API_URL:-http://localhost:6340}"
MIN_MESSAGES="${PALINODE_HOOK_MIN_MESSAGES:-3}"
# Max time (seconds) the curl POST is allowed to run.  Raise with
# PALINODE_HOOK_TIMEOUT if your host is slow (cold Ollama, WAN Tailscale, NFS).
# The Claude Code hook runner timeout in settings.json must be > this value.
HOOK_TIMEOUT="${PALINODE_HOOK_TIMEOUT:-30}"

# Reasons to capture on. Default broad: clear, logout, normal exit (other),
# and non-interactive EOF. Override with PALINODE_HOOK_REASONS to narrow
# (e.g. "clear") or extend (add "resume" / "bypass_permissions_disabled").
# See https://code.claude.com/docs/en/hooks.md for the full reason list.
ALLOWED_REASONS="${PALINODE_HOOK_REASONS:-clear logout prompt_input_exit other}"

# Optional bearer auth for token-protected deployments (PALINODE_API_TOKEN).
# The ${AUTH[@]+…} expansion is the bash-3.2-safe empty-array idiom (set -u).
AUTH=()
if [ -n "${PALINODE_API_TOKEN:-}" ]; then
  AUTH=(-H "Authorization: Bearer ${PALINODE_API_TOKEN}")
fi

INPUT=$(cat)
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
SOURCE_REASON=$(echo "$INPUT" | jq -r '.source // .reason // "other"')

# This preflight intentionally precedes every transcript operation. A malformed
# or unavailable response fails closed without emitting user content.
CONTROL_PAYLOAD=$(jq -n \
  --arg cwd "$CWD" --arg source_path "$TRANSCRIPT_PATH" \
  '{action: "capture", cwd: $cwd, source_path: $source_path, automatic: true}')
CONTROL=$(curl -sS -f \
  -X POST "${PALINODE_API}/controls/check" \
  ${AUTH[@]+"${AUTH[@]}"} \
  -H "Content-Type: application/json" \
  -d "$CONTROL_PAYLOAD" \
  --connect-timeout 2 \
  --max-time "${HOOK_TIMEOUT}" 2>/dev/null) || exit 0
[ "$(echo "$CONTROL" | jq -r 'if .allowed == true then "yes" else "no" end' 2>/dev/null)" = "yes" ] || exit 0
PROJECT=$(echo "$CONTROL" | jq -r '.project // empty' 2>/dev/null) || exit 0

# Drop reasons we're not capturing. Word-boundary match on a space-padded
# allowlist so substrings (e.g. "log" in "logout") don't false-positive.
case " $ALLOWED_REASONS " in
  *" $SOURCE_REASON "*) ;;
  *) exit 0 ;;
esac

# No transcript → nothing to capture.
if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
  exit 0
fi

# Skip-if-/wrap-ran (floor/ceiling): if the human already ran /wrap this
# session, the transcript holds a `palinode_session_end` tool call. That
# agent-authored capture (summary + decisions + blockers, each with a why) is
# strictly richer than this deterministic floor, so writing the floor too just
# duplicates. Skip. Override with PALINODE_HOOK_FORCE=1 to capture regardless.
if [ "${PALINODE_HOOK_FORCE:-0}" != "1" ] \
   && jq -e -s 'any(.[]; .type == "assistant"
       and any(.message.content[]?; .type == "tool_use"
           and .name == "palinode_session_end"))' \
       "$TRANSCRIPT_PATH" >/dev/null 2>&1; then
  exit 0
fi

# Claude Code transcript format (JSONL):
#   user:      {type: "user", message: {role: "user", content: "text"}}
#   assistant: {type: "assistant", message: {content: [{type: "text", text: "..."}]}}
#
# Both extractions use `jq -s` (slurp) so all reductions happen INSIDE jq.
# Earlier versions piped `jq | head -1` and `jq | grep -c '.'`, which was
# fragile under `set -o pipefail`: the downstream consumer exits early, the
# next jq write hits a closed pipe → SIGPIPE → pipefail aborts the script.
# Slurping reads JSONL lines into an array; map+filter+slice runs without an
# early-exit downstream consumer, eliminating the SIGPIPE class entirely.
MSG_COUNT=$(jq -r -s 'map(select(.type == "user") | .message.content // empty) | length' \
  "$TRANSCRIPT_PATH" 2>/dev/null || echo 0)
MSG_COUNT=${MSG_COUNT:-0}

# Skip trivial sessions (few messages = not worth a memory).
if [ "$MSG_COUNT" -lt "$MIN_MESSAGES" ]; then
  exit 0
fi

# The first human ask is a *topic hint*, not content — and in Claude Code it is
# routinely wrapped in harness markup (slash-command expansion, system
# reminders, bash/IDE blocks). Left in, that markup is embedded and indexed as
# though it were what the session was about. Strip it here, at the
# source: wrapper blocks whose body is machinery lose the whole block; the
# command-name tags lose only the tags, keeping the human-meaningful text.
#
# The topic is the first user turn that still has human text after stripping,
# not blindly the first user turn: after /clear the transcript opens with a
# harness-generated (`isMeta`) <local-command-caveat> turn, then the bare
# `/clear` invocation, and only then the real ask. So: skip `isMeta`
# turns, drop the caveat block whole, treat a command invocation with empty
# args as no text, read only `text` blocks from block-array content (a
# tool_result turn has no human text), and take the first non-empty result.
# `|| echo ""` keeps a jq failure from tripping `set -e`.
FIRST_PROMPT=$(jq -r -s '
    [ .[] | select(.type == "user" and .isMeta != true) | .message.content // empty
      | if type == "string" then .
        elif type == "array" then [ .[] | select(type == "object" and .type == "text") | .text // "" ] | join(" ")
        else tojson end
      | gsub("<local-command-caveat>.*?</local-command-caveat>"; ""; "s")
      | gsub("<system-reminder>.*?</system-reminder>"; ""; "s")
      | gsub("<local-command-std(out|err)>.*?</local-command-std(out|err)>"; ""; "s")
      | gsub("<bash-(input|stdout|stderr)>.*?</bash-(input|stdout|stderr)>"; ""; "s")
      | if test("<command-name>") and (gsub("<command-(name|message)>[^<]*</command-(name|message)>|<command-args>\\s*</command-args>"; "") | test("^\\s*$")) then "" else . end
      | gsub("</?(command-message|command-name|command-args|user-prompt-submit-hook|ide_selection|ide_opened_file)>"; " ")
      | gsub("\\s+"; " ") | sub("^ +"; "") | sub(" +$"; "")
      | select(length > 0) ] | .[0] // ""' \
  "$TRANSCRIPT_PATH" 2>/dev/null | cut -c1-200 || echo "")

SUMMARY="Auto-captured (${SOURCE_REASON}, ${MSG_COUNT} messages). Topic: ${FIRST_PROMPT}"

PAYLOAD=$(jq -n \
  --arg summary "$SUMMARY" \
  --arg project "$PROJECT" \
  --arg source "claude-code-hook" \
  --arg cwd "$CWD" \
  --arg source_path "$TRANSCRIPT_PATH" \
  --arg session_id "$SESSION_ID" \
  '{summary: $summary, project: $project, source: $source, cwd: $cwd,
    source_path: $source_path, session_id: $session_id,
    trigger: "session-end-hook", automatic: true,
    decisions: [], blockers: []}')

# Dry-run: print what would be POSTed and write nothing. Lets you verify the
# hook wiring (reasons, triviality gate, payload shape) without touching the
# API or persisting a memory. PALINODE_HOOK_DRYRUN=1 to enable.
if [ "${PALINODE_HOOK_DRYRUN:-0}" = "1" ]; then
  echo "[palinode-session-end DRYRUN] would POST ${PALINODE_API}/session-end"
  echo "$PAYLOAD"
  exit 0
fi

# The API rechecks the persisted policy immediately before its write. Do not
# persist a fallback on failure: a 403 may mean the user paused capture after
# this preflight.
curl -sS -o /dev/null -f \
    -X POST "${PALINODE_API}/session-end" \
    ${AUTH[@]+"${AUTH[@]}"} \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    --connect-timeout 5 \
    --max-time "${HOOK_TIMEOUT}" 2>/dev/null || true

exit 0

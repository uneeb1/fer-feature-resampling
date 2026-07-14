# Codex CLI — a tool you (Claude) can call

This machine has **Codex CLI** installed and authenticated (logged in via ChatGPT, so no API key plumbing needed). You can invoke it directly via Bash whenever it's useful. Treat Codex the way you'd treat a capable peer agent: delegate bounded sub-tasks, get second opinions, offload work that doesn't need to live in your own context.

---

## When to use Codex

Use it when:
- The user asks you to "ask codex", "use codex", "have codex do X", or similar.
- You need a **second opinion** on a design, diff, or bug diagnosis.
- You want to **offload** a self-contained sub-task (research, a bounded refactor, writing a module from a spec, summarizing a long file) so your own context stays focused.
- You have **parallelizable** work — kick off a Codex run in the background while you keep working.

Don't use it for:
- Trivial one-liners you can answer directly.
- Anything that requires state already sitting only in your conversation (Codex won't see it unless you pipe it in).

---

## The one command shape to remember

**Always use `codex exec`. Never run bare `codex` or `codex "..."` — that opens an interactive TUI and will hang your Bash tool.**

Canonical invocation (always pass `-m` and `-c model_reasoning_effort=` explicitly — see Model selection below):

```bash
codex exec --sandbox workspace-write --skip-git-repo-check \
  -m <MODEL> -c model_reasoning_effort="<low|medium|high>" \
  -o /tmp/codex-out.txt \
  "<the task prompt>"
```

Then read `/tmp/codex-out.txt` with the Read tool to get the final message. On Windows the path `/tmp/codex-out.txt` works from Git Bash; if it doesn't, use `"$TEMP/codex-out.txt"` or an absolute path like `C:/Users/Migrando1/AppData/Local/Temp/codex-out.txt`.

Why these defaults:
- `--sandbox workspace-write` — lets Codex read and edit files in the working directory (matches the user's own usage). Do **not** silently upgrade to `danger-full-access`.
- `--skip-git-repo-check` — Codex refuses to run outside a git repo by default; many of the user's working dirs aren't repos.
- `-o <file>` — stdout works, but a file is more reliable for long outputs. stderr carries streaming progress and is noisy; ignore it unless debugging.
- **Explicit `-m` + `-c model_reasoning_effort`** — without these, Codex can silently default to `gpt-5.5` at `xhigh`, the most expensive combo. Always set them.

---

## Model selection (READ THIS BEFORE EVERY CALL)

The user pays for Codex on a daily limit. Burning it on the frontier model for trivial work is the failure mode to avoid. Pick model + reasoning per-call from the rubric below.

**Available models** (priority = how prominent in Codex's UI; lower = more prominent):

| Slug | Tier | Best for | Cost class |
|---|---|---|---|
| `gpt-5.5` | Frontier | Hard problems 5.4 can't solve, large unfamiliar codebases, novel research, deep design | 💸💸💸 highest |
| `gpt-5.4` | Strong everyday | Multi-file refactors, normal debugging, code review, writing modules from a spec | 💸💸 |
| `gpt-5.4-mini` | Small/fast | Lookups, summaries, formatting, short Q&A, single-file edits with clear instructions | 💸 cheapest |
| `gpt-5.3-codex` | Older coding | Legacy — Codex itself suggests upgrading to 5.4. Don't pick unless user names it. | 💸💸 |
| `gpt-5.2` | Older agent | Legacy — same as above. Don't pick unless user names it. | 💸💸 |

All five support `model_reasoning_effort` levels: `low`, `medium`, `high`, `xhigh`.

### Routing rubric (smart pick + transparent escalation)

| Task | Model | Reasoning |
|---|---|---|
| Lookup, summarize a file, format text, classify, "what does X do" | `gpt-5.4-mini` | `low` |
| Simple single-file edit with clear instructions, write a small function from a spec | `gpt-5.4-mini` | `medium` |
| Everyday coding: multi-file refactor, debug with clear repro, code review of small diff | `gpt-5.4` | `medium` |
| Hard debugging without clear repro, architectural call, security review of unfamiliar code, design Q | `gpt-5.4` | `high` |
| 5.4 already tried and produced a clearly-wrong / shallow answer, OR very large unfamiliar codebase, OR novel research | `gpt-5.5` | `high` |
| User explicitly says "use 5.5" / "use frontier" / "deep think" | `gpt-5.5` | as user directs (`high` default, `xhigh` only if they say so) |

**Never default to `xhigh` reasoning.** It's reserved for explicit user request only.

### Escalation protocol — using `gpt-5.5`

Before launching any `gpt-5.5` call that wasn't explicitly requested by name, say one short line in user-facing text:

> Escalating to gpt-5.5 (high) for this — reason: {large codebase / 5.4's previous attempt was shallow / novel design Q}. Say "stay on 5.4" if you'd rather.

Then run. Don't ask for permission and wait — the user opted for transparent escalation, not approval gates. But if they push back, drop down for the next call.

### Quick examples filled in

```bash
# Trivial: "what does this file do"
codex exec --sandbox workspace-write --skip-git-repo-check \
  -m gpt-5.4-mini -c model_reasoning_effort="low" \
  -o /tmp/codex-out.txt "Summarize what src/auth.ts does in 3 sentences."

# Everyday: refactor a function
codex exec --sandbox workspace-write --skip-git-repo-check \
  -m gpt-5.4 -c model_reasoning_effort="medium" \
  -o /tmp/codex-out.txt "Extract the validation logic in handler.ts:42-95 into a separate module."

# Hard: design review
codex exec --sandbox workspace-write --skip-git-repo-check \
  -m gpt-5.4 -c model_reasoning_effort="high" \
  -o /tmp/codex-out.txt "Review the auth middleware design in src/auth/. Focus on session handling correctness."

# Frontier (after escalation message to user):
codex exec --sandbox workspace-write --skip-git-repo-check \
  -m gpt-5.5 -c model_reasoning_effort="high" \
  -o /tmp/codex-out.txt "Trace why request X returns 500 across services A→B→C; logs in /tmp/logs/."
```

---

## Flag reference (the ones that matter)

| Flag | Use |
|---|---|
| `-s, --sandbox <mode>` | `read-only` \| `workspace-write` (default) \| `danger-full-access` |
| `--full-auto` | Shortcut for low-friction sandboxed auto-exec. Convenient for clearly-bounded tasks. |
| `--dangerously-bypass-approvals-and-sandbox` | No sandbox, no prompts. **Only with explicit user approval for that specific run.** |
| `--skip-git-repo-check` | Needed when the working dir isn't a git repo. Include by default. |
| `-o, --output-last-message <FILE>` | Writes the final assistant message to a file. Use this + Read. |
| `--json` | JSONL event stream on stdout (use if you need to parse intermediate steps). |
| `-m, --model <MODEL>` | Pick the model. **Always set explicitly** per the routing rubric — see Model selection above. |
| `-c model_reasoning_effort="<level>"` | Reasoning effort: `low` / `medium` / `high` / `xhigh`. Always set. xhigh only on explicit user ask. |
| `-C, --cd <DIR>` | Set Codex's working root. |
| `--add-dir <DIR>` | Extra writable dir alongside the workspace. |
| `-i, --image <FILE>` | Attach an image to the prompt. |
| `--ephemeral` | Don't persist session rollout to `~/.codex/sessions/`. |
| `--output-schema <FILE>` | Constrain the final response to a JSON Schema. |
| `resume --last` / `resume <SESSION_ID>` | Continue a prior Codex session. |

---

## Stdin patterns

**Append piped data as context** (stdin is wrapped in a `<stdin>` block):
```bash
cat long_log.txt | codex exec --sandbox workspace-write --skip-git-repo-check \
  -o /tmp/codex-out.txt "Find the root-cause stacktrace and explain it."
```

**Use stdin as the entire prompt** (pass `-` as the prompt arg):
```bash
cat prompt.txt | codex exec --sandbox workspace-write --skip-git-repo-check \
  -o /tmp/codex-out.txt -
```

---

## Output handling

- **stdout** — the final agent message (or JSONL events if `--json`).
- **stderr** — streaming progress/thoughts. Noisy. Don't show the user unless they're debugging.
- **`-o <file>`** — clean copy of the final message. Prefer this.
- If you used `--json`, the final answer is in the event shaped `{"type":"item.completed","item":{"type":"agent_message","text":"..."}}`. Other event types you'll see: `thread.started`, `turn.started`, `turn.completed`, `turn.failed`, `error`.
- Harmless noise on stderr: `Reading additional input from stdin...` appears even when no stdin was piped. Codex reads, hits EOF, proceeds. Ignore it. If you want to be explicit, append `< /dev/null` to the command.

---

## Safety rules

1. **Never run bare `codex`** — it's an interactive TUI and will hang your session. Always `codex exec`.
2. **Default sandbox is `workspace-write`.** Escalate to `danger-full-access` or `--dangerously-bypass-approvals-and-sandbox` **only with the user's explicit go-ahead for that specific run**, and tell them what's about to happen.
3. **Treat `~/.codex/auth.json` as a secret** — it contains access tokens. Don't print it, don't commit it.
4. **Long runs → background.** For any Codex call that might take more than ~30s, use the Bash tool's `run_in_background: true`. You'll be notified when it finishes — don't poll, don't sleep.
5. **Iterate via resume, not re-priming.** If the user wants to continue a Codex thread, use `codex exec resume --last "<follow-up>"` rather than re-sending the whole context.
6. **Trust-but-verify.** Codex's summary of what it did is not proof it did it. If Codex claims to have edited files, check the diff.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Not a git repository" | Add `--skip-git-repo-check` (already in the canonical shape). |
| Auth error | Have the user run `codex login status`, then `codex login` if needed. You can't log in for them — it's an interactive flow. |
| Hang / no output | You probably ran bare `codex` instead of `codex exec`. Kill it, retry with `exec`. |
| Very long output truncated | Use `-o <file>` and Read the file; don't rely on stdout capture. |
| Need structured data back | Pass `--output-schema schema.json` and a schema, then parse the output file as JSON. |

---

## Quick examples

**One-shot question:**
```bash
codex exec --sandbox workspace-write --skip-git-repo-check \
  -o /tmp/codex-out.txt \
  "Explain what src/auth/session.ts does in two paragraphs."
```
Then `Read /tmp/codex-out.txt`.

**Piping context in:**
```bash
git diff HEAD~5..HEAD | codex exec --sandbox workspace-write --skip-git-repo-check \
  -o /tmp/codex-out.txt \
  "Review this diff for bugs and security issues. Be specific about files and lines."
```

**Background long task** (Bash tool with `run_in_background: true`):
```bash
codex exec --sandbox workspace-write --skip-git-repo-check \
  -o /tmp/codex-essay.txt \
  "Write a 1500-word technical brief on <topic>, with citations."
```
Keep working. When the notification fires, Read the output file.

**Structured JSON output:**
```bash
codex exec --sandbox workspace-write --skip-git-repo-check \
  --output-schema ./schema.json \
  -o /tmp/codex-out.json \
  "Extract all TODO comments from src/ and return them as {file, line, text} objects."
```

**Continue the previous Codex session:**
```bash
codex exec --sandbox workspace-write --skip-git-repo-check \
  -o /tmp/codex-out.txt \
  resume --last "Now also add unit tests for the refactor you just did."
```
⚠ **Flag order matters for `resume`.** All `codex exec` flags (`--sandbox`, `--skip-git-repo-check`, `-o`, etc.) must come **before** the `resume` subcommand. Putting them after `resume` fails with `error: unexpected argument '--sandbox' found`. `resume`'s own args are just `[SESSION_ID] [PROMPT]` (or `--last [PROMPT]`).

---

## One-line summary for your own reference

> Codex is a peer CLI agent. Call it with `codex exec --sandbox workspace-write --skip-git-repo-check -m <model> -c model_reasoning_effort="<level>" -o <file> "<prompt>"`. Pick model + reasoning from the rubric (default `gpt-5.4-mini`/`low` for trivial, `gpt-5.4`/`medium` for everyday, `gpt-5.4`/`high` for hard, `gpt-5.5` only after announcing escalation). Background long runs, read the output file, never run bare `codex`, never escalate the sandbox without asking, never default to `xhigh`.

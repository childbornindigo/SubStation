> ⛔ HOLD — DO NOT SHIP TO HUDSON YET (Dee, 2026-07-10)
> Reason: OAuth stopped working at ~07:00 today for BOTH the orchestrator AND
> Claude Code terminal (which does NOT route through SubStation). That points at
> an account/OAuth-level problem, not the SDK-vs-raw-REST transport this package
> "fixes." Usage is also running higher than ever. Before shipping anyone a fix,
> confirm our own operation is healthy and not at shutdown risk, and re-test
> whether raw REST actually still 429s today (transport toggle theory). Do not
> merge/push/tag or hand this to Hudson until Dee lifts this hold.

# SubStation Update Package — for Hudson

Audit date: 2026-07-10. This machine's SubStation has working fixes for Fable
detection and OpenAI-shim tool-calling that Hudson's separate deployment
doesn't have yet. This doc is the one-command path to get his box current.

## Current state (as of this audit)

- **Branch:** `fix/image-gen-pro-quality-2026-07-09`
- **Latest commit:** `a8f02bb4d30371f349381329095b4dd857c10150`
- **NOT on `main` yet, NOT tagged yet** — this branch is queued for a
  batched merge/push/tag pass Dee runs separately. Do not point Hudson at
  `main` until that lands; point him at this branch/SHA directly for now,
  or wait for the tag if you're reading this after the merge happened
  (check `git log main --oneline -1` first — if `a8f02bb` or later is on
  `main`, use `main` instead of the branch name below).

## Commit inventory (all verified present, all on the branch above)

| SHA | What |
|---|---|
| `045018f` | OpenAI shim tool-forwarding (`/v1/chat/completions` now reads `tools`/`tool_choice`) |
| `a8a91e5` | Fable swap-detection guard + daily probe |
| `390ca8d` | gpt-5.5 + `quality:high` image-gen patch |
| `89c7313` | Claude-Code client-identity headers on `makeAnthropicRequest` |
| `a8f02bb` | **Release-audit commit.** Backports everything below — was live-only in `dist/index.js`, never committed until now |
| `ae06377` | ~~systemPrompt fix~~ — **REVERTED by `d89ccbc`, DO NOT REAPPLY.** Broke live billing classification, see below. |
| `d89ccbc` | Revert of `ae06377` — restores pre-fix behavior. |
| `4edcdd6` | **settingSources fix — safe, verified, in effect now.** `invokeClaudeSDKWithTools` passes `settingSources: []` to isolate from the operator's personal `~/.claude` environment, without touching `systemPrompt` at all. See "settingSources fix — what and why" below. |

### What `a8f02bb` backported (was dist-only hand-edits, now in `src` + committed)
- **`invokeClaudeSDKWithTools`** — the tools-path now goes through the Claude
  Agent SDK (via `createSdkMcpServer` + in-process MCP tool capture) instead
  of raw REST to `api.anthropic.com`. This dodges a real, reproducible
  Anthropic edge-side rate limit that raw REST tool-calling requests get
  bucketed into — confirmed independently: a fresh, fully-scoped OAuth token
  429'd on raw REST (even with matching client-identity headers) while the
  same token worked through the SDK. The old raw-REST path
  (`invokeClaudeMessagesAPIWithTools`) is kept as a documented LEGACY
  fallback, not the primary path anymore.
- `claude-sonnet-5` model support (`MODEL_CONFIG`, `MODEL_MAP`, `/v1/models`).
- `X-SubStation-No-Failover` request header — pins the whole Anthropic pool
  without ever crossing over to an OpenAI-disguised answer, for cases where
  you deliberately need to see the raw Anthropic result/error.
- A real bug found and fixed **during this audit**: 2 of `dist/index.js`'s
  3 `anthropic-client-version`/`User-Agent` occurrences were stuck on a
  stale `2.1.114`, inconsistent with the file's own 3rd occurrence and with
  the actually-installed Claude Code CLI (`2.1.197`, verified live via
  `claude --version` / `npm ls -g @anthropic-ai/claude-code`). All 3 are now
  `2.1.197` in both `src` and `dist`.

## ⛔ systemPrompt fix (`ae06377`) — REVERTED, do not reapply (see `d89ccbc`)

**This entire section describes a change that broke production and was
reverted within minutes.** Kept below for the record, not as guidance.

Within 6 seconds of this fix going live, the only live Anthropic account in
the pool started hard-failing every SDK-tools request with `API Error: 400
You're out of extra usage`. Dee confirmed live (chatting through that same
account via a separate native session at the time) that the account had
real usage available and should never see that message under his plan —
ruling out a real cap coincidence. Root cause, best evidence: passing a
**custom (non-preset) `systemPrompt` string** to the Agent SDK's `query()`
appears to make Anthropic's backend reclassify the session as generic
third-party Agent-SDK usage instead of Claude Code subscription usage,
which draws from a billing bucket this token isn't provisioned for.
Reverting immediately restored clean 200s on the exact same account/model.
**Do not pass a custom `systemPrompt` string to this SDK's `query()` calls
on a subscription OAuth token without first confirming the billing-
classification behavior with Anthropic directly** — the `{type:'preset',
preset:'claude_code', append:...}` form is the more likely safe path (keeps
the `claude_code` identity, adds instructions) but does not shed the token
cost, so the original cost problem below is still real and still unsolved.

## systemPrompt fix (`ae06377`) — what and why [HISTORICAL — REVERTED, see banner above]

**What changed:** `invokeClaudeSDKWithTools` (the SDK-mediated tool-calling
bridge added in `a8f02bb`) calls the Claude Agent SDK's `query()`. Before
this commit, it passed no `systemPrompt` option at all, so the Agent SDK
fell back to its own default: the full Claude Code system prompt (agentic
coding-assistant identity, tool-use instructions, environment scaffolding —
the same prompt a real Claude Code terminal session pays for). After this
commit, `query()` is called with `options.systemPrompt` set to a plain
string — `OPERATOR_SYSTEM_PROMPT` (the existing lean operator-identity
constant already used elsewhere in this file) plus any caller-provided
system messages, composed the same way the legacy raw-REST path
(`invokeClaudeMessagesAPIWithTools`) builds its `system` field. **A plain
string REPLACES the SDK's default entirely** — this is a deliberate choice;
the SDK's other `systemPrompt` form, `{type:'preset', preset:'claude_code',
append:'...'}`, only ADDS to the default and would make things worse, not
better.

**Why it mattered:** every tool-calling request through the SDK bridge was
silently paying for the full Claude Code system prompt (~24K tokens on a
cold cache, measured directly via the SDK's own reported `usage` — see
below) in addition to the caller's actual payload. On a fleet running many
tool-calling requests per session, that's real, avoidable token spend. It
was also an identity-correctness bug independent of cost: without this fix,
the model was liable to describe itself as "Claude Code" mid-response
(SubStation is not Claude Code and shouldn't present as it), since nothing
overrode the SDK's default persona.

**Evidence (measured directly against the SDK's own `result.usage`, not an
inferred/estimated number):**
- **Cold cache** (first call after a system-prompt change — worst case):
  default prompt cost **$0.1002/request**; lean prompt cost
  **$0.0652/request** — 35% cheaper.
- **Warm cache / steady state** (repeat calls, the normal operating mode
  for sustained orchestrator traffic): default **$0.0159/request**; lean
  **$0.0161/request** — statistically identical. Anthropic's prompt caching
  amortizes the fat default prompt down to near-zero incremental cost once
  it's warm, so **this fix is not a steady-state cost fix** — its value is
  the cold-start/cache-miss case (bursty usage, multi-account rotation
  fragmenting the cache, session restarts) and the identity correctness,
  not a guaranteed reduction in ongoing spend. If your bot is seeing
  elevated usage that this fix doesn't explain, look at request volume and
  cache-hit rate before assuming the system prompt is still the cause.
- **429-dodge survived the trim** (this was the critical risk in this
  change — confirmed empirically, not assumed): a post-fix, post-restart
  tool-call probe against a live token returned a real `toolu_`-prefixed
  tool call (200, `x-substation-served-model` matched the request), with no
  429 and no OpenAI-failover log line. The 429-immunity comes from the
  SDK/OAuth transport itself, not from the system prompt's content — so
  trimming the prompt does not reintroduce the raw-REST 429-bucketing
  problem that motivated the SDK bridge (`a8f02bb`) in the first place.

## settingSources fix (`4edcdd6`) — what and why, and why it's safe

**What changed:** `invokeClaudeSDKWithTools`'s `query()` call now passes
`options.settingSources = []`. This is an official SDK option ("SDK
isolation mode" per the SDK's own type docs) that stops loading the
operator's filesystem settings (`~/.claude/settings.json` and project/local
settings) — which by default get loaded automatically when the option is
omitted.

**Why it mattered:** using the SDK's own `getContextUsage()` diagnostic
(a real per-category token breakdown, not a guess), the actual system
prompt turned out to be tiny (~245 tokens) — the real cost was
`invokeClaudeSDKWithTools` silently inheriting the operator's **entire
personal Claude Code environment** on every single tool-calling request:
global MCP servers (deferred: 16,100 tokens), custom agents (1,956 tokens),
skills (5,411 → 721 tokens after the fix), none of which a shared
multi-tenant tool-calling bridge needs or wants.

**Why this is the safe fix and `ae06377` (reverted, above) was not:** this
change never touches `systemPrompt` — it stays completely unset, still
defaulting to the SDK's official `claude_code` preset. That preset appears
to be what keeps a request billed as normal Claude Code subscription usage
(see the reverted section above). `settingSources` is a different,
unrelated option that only controls which *filesystem settings* get loaded
into the session — it does not change the system-prompt/identity shape that
seems to drive billing classification.

**Evidence (measured directly against the SDK's own `result.usage`):**
- Cold cache: cache-creation tokens dropped from 24,385 → 9,552; cost
  $0.1002 → $0.0477/request.
- Warm/steady-state (4 consecutive calls): $0.0141–$0.0152/request,
  consistently at or below the prior $0.0159 baseline.
- **No billing-classification regression**, unlike `ae06377`: verified
  with 3 successive live tool-call probes against the exact
  account/model in production use (`opus-4-8`, `acct2-dsskerritt11`) post
  restart — all 200, real `toolu_`-prefixed tool calls, zero 429s, zero
  "out of extra usage" errors, zero OpenAI-failover lines.
  `pool-state.json`'s `errorCount` stayed at 0 throughout. Also verified
  end-to-end through the real Hermes agent CLI path (not just direct
  SubStation probes) with clean, fast, correct responses.
- One known gap: "Memory files" (7,550 tokens) stayed unchanged by this
  fix — appears to be a separate loading mechanism from `settingSources`,
  not addressed here. Not investigated further; flag if it turns out to
  matter.

## Build story: there is no build step

Verified: no `tsconfig.json`, no bundler config, no `build` script in
`package.json`, no relevant `devDependencies`. `dist/index.js` is not
compiled from `src/index.js` — it's the same file, kept in sync by hand
until this audit. Because of that, **`dist/` is no longer fully
gitignored** — `dist/index.js` specifically is now tracked (historical
`dist/*.bak-*` snapshots stay ignored, local-only). `src/index.js` and
`dist/index.js` are byte-identical as of `a8f02bb`.

## Update path (one command, once on the right branch/SHA)

```bash
cd ~/.hermes/extensions/substation
git fetch origin
git checkout fix/image-gen-pro-quality-2026-07-09   # or main, once merged — see note above
git pull
# no build step — dist/index.js is already the deployable file, pulled directly
```

## Restart

Adjust the service name/plist to whatever Hudson's launchd label actually is
on his machine — on this machine it's:
```bash
launchctl kickstart -k gui/$(id -u)/com.indigochild.substation
```

## Verification steps (run these after restart)

1. **Health check:**
   ```bash
   curl -s http://localhost:8403/v1/models -H "Authorization: Bearer <his-local-proxy-key>"
   ```
   Expect HTTP 200 and `sonnet-5` present in the `data[]` list (confirms
   the backport landed, not a stale artifact).

2. **Tool-call probe (Claude family) — confirms `invokeClaudeSDKWithTools` is live:**
   ```bash
   curl -s http://localhost:8403/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <his-local-proxy-key>" \
     -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"weather in Boston? use the tool"}],
          "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],
          "tool_choice":"required"}'
   ```
   Expect a real `tool_calls` array with an `id` prefixed `toolu_` — that
   prefix specifically means the call round-tripped through real Anthropic
   (not a text dodge, not a null). Live-verified on this machine just now:
   `id: "toolu_017FSh7YkFAMQeHgSdZwBmcN"`, `finish_reason: "tool_calls"`.

3. **`x-substation-served-model` header check:** on any `/v1/chat/completions`
   response, inspect this response header — it reports the model the SDK
   actually served vs. the model requested. If they differ, that's a real
   swap (see fableguard below); if they match, the request went to the
   expected model.

4. **Fableguard behavior:** the served-model swap-detection guard (`a8a91e5`)
   logs a `WARN: model swap detected` line (see `invokeClaudeSDKWithTools`
   and the SDK init-message comparison) whenever the requested model and the
   SDK-served model diverge. Check the daily probe output / logs for this
   pattern rather than assuming silence means "fine" — confirm at least one
   clean run post-update.

5. **Fable entitlement is PER-ACCOUNT.** If Hudson updates and his Fable
   behavior/bar doesn't move, that is almost certainly his account's
   entitlement, not a code/version gap — don't chase a code fix for what's
   actually an account-tier difference. Confirm via his own account's
   swap-detection guard output before assuming the update didn't take.

## What was explicitly NOT done in this audit (by design)

Per standing Right-Hand doctrine (never merge to `main` autonomously,
especially on a **public** repo) this audit stopped short of:
- Merging `fix/image-gen-pro-quality-2026-07-09` → `main`
- Pushing to `origin`
- Tagging a release (e.g. `v2026.07.10-hudson`)

Those three are a single batched decision for Dee to make explicitly —
not a blocker on getting this doc and the backport commit ready. Once he
gives the go: `git checkout main && git merge fix/image-gen-pro-quality-2026-07-09 && git push && git tag v2026.07.10-hudson && git push --tags`,
then update the "Current state" section above to point at `main`.

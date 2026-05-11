# Infrastructure Reality Audit

## Goal
Comprehensive audit of every system in the Indigo stack — compare what was designed/intended vs what's actually working. Identify what's real, what's fluff, and what's a mansion with broken doors.

## Context
Dee flagged that the previous harness built tons of half-finished things. We need to separate load-bearing infrastructure from architectural vaporware.

## Audit Categories
1. **LaunchAgents** — all com.indigo.*, com.indigoos.*, ai.hermes.* plists
2. **ObsidianBrain** — wiki compiler, vault hygiene, vault-watcher, scope enforcement
3. **Indigo Core** — plugins, event bus, self-improve loop
4. **SubStation** — credentials, routing, gateway health
5. **MCP Bridge** — handoff DB, agent connectivity
6. **Skills** — 43 skills in vault, how many actually work
7. **Projects** — what graduated, what's stalled, what's vaporware
8. **Cron/Scheduled tasks** — what runs, what's dead

## Status
- [ ] LaunchAgents audit
- [ ] ObsidianBrain audit
- [ ] Indigo Core audit
- [ ] SubStation audit
- [ ] MCP Bridge audit
- [ ] Skills audit
- [ ] Projects audit
- [ ] Final scorecard

## Started
2026-04-29 15:20 EDT

#!/usr/bin/env node
/**
 * Right Hand Agent SDK Service
 *
 * HTTP microservice that wraps the Claude Agent SDK for Telegram dispatch.
 * Rotates through all available Anthropic OAuth tokens before falling back to Codex.
 *
 * Endpoints:
 *   POST /ask        — send a prompt, get a response
 *   POST /reset      — force-reset the persistent session
 *   POST /model      — manual model switch { provider: "anthropic" | "codex" }
 *   POST /uncap      — clear anthropic cap flag (shortcut)
 *   GET  /health     — session status
 *
 * Sessions auto-reboot every 6h or when context gets heavy.
 */

import { createServer, request as httpRequest } from 'http';
import { query } from '@anthropic-ai/claude-agent-sdk';
import { homedir } from 'os';
import { join } from 'path';
import { readFileSync, writeFileSync, existsSync } from 'fs';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const PORT = parseInt(process.env.RIGHT_HAND_PORT || '8450', 10);
const MAX_TURNS = parseInt(process.env.RIGHT_HAND_MAX_TURNS || '30', 10);
const SESSION_MAX_AGE_MS = 6 * 60 * 60 * 1000; // 6 hours
const AUTH_FAIL_TTL_MS = 24 * 60 * 60 * 1000;  // 24h before retrying a dead token
const WORKSPACE = join(homedir(), '.openclaw', 'agents', 'right-hand', 'workspace');
const SUBSTATION_URL = 'http://127.0.0.1:8403';
const AUTH_FAIL_FILE = join(homedir(), '.openclaw', 'agents', 'right-hand', 'sdk', 'auth-failures.json');

const SYSTEM_PROMPT = `You are Right Hand — the persistent builder agent for Indigo OS, responding via Telegram.

You have FULL tool access. All permissions are bypassed. Execute any tool directly — never ask for approval.

Available: Bash, Read, Write, Edit, WebFetch, Grep, Glob, and all MCP tools.
CLI tools on PATH: xurl, git, node, python3, npx, sqlite3, curl.

Rules:
- Be concise. This is mobile. No walls of text.
- Execute tasks directly — don't plan them. You have full file and bash access.
- You know Dee's systems: SubStation, OpenClaw, Indigo Core, Hermes, the Growth Engine, INDIGO HOUSE.
- To read tweets: use xurl read <tweet_id> via Bash.
- To fetch URLs: use WebFetch or curl.
- Direct. No filler.`;

// ---------------------------------------------------------------------------
// OAuth token loader — collects all Anthropic tokens for rotation
// ---------------------------------------------------------------------------

function loadOAuthTokens() {
  const tokens = [];
  const seen = new Set();

  function add(id, token) {
    if (!token || seen.has(token)) return;
    seen.add(token);
    tokens.push({ id, token });
  }

  // Source 1: Right Hand's own auth-profiles
  const rhPath = join(homedir(), '.openclaw', 'agents', 'right-hand', 'agent', 'auth-profiles.json');
  try {
    const data = JSON.parse(readFileSync(rhPath, 'utf8'));
    for (const [pid, profile] of Object.entries(data.profiles || {})) {
      if (pid.startsWith('anthropic:') && profile.token) {
        add(`rh:${pid}`, profile.token);
      }
    }
  } catch {}

  // Source 2: Main agent auth-profiles
  const mainPath = join(homedir(), '.openclaw', 'agents', 'main', 'agent', 'auth-profiles.json');
  try {
    const data = JSON.parse(readFileSync(mainPath, 'utf8'));
    for (const [pid, profile] of Object.entries(data.profiles || {})) {
      if (pid.startsWith('anthropic:') && profile.token) {
        add(`main:${pid}`, profile.token);
      }
    }
  } catch {}

  // Source 3: OAuth pool file
  const poolPath = join(homedir(), '.openclaw', 'workspace', 'scripts', 'oauth-pool.json');
  try {
    const data = JSON.parse(readFileSync(poolPath, 'utf8'));
    for (const cred of (data.credentials || [])) {
      if (cred.token && cred.token.startsWith('sk-ant-')) {
        add(`pool:${cred.id}`, cred.token);
      }
    }
  } catch {}

  return tokens;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let sessionCreatedAt = Date.now();
let requestCount = 0;
let busy = false;
let anthropicCapped = false;  // true when ALL Anthropic tokens are capped
let cappedTokens = new Set();  // track which individual tokens are capped
let substationHistory = [];  // persistent message history for SubStation fallback
const SUBSTATION_MAX_HISTORY = 100;  // keep last 100 messages (user+assistant pairs)

// Auth-failure sensor: persisted to disk, survives restarts
// Map of tokenId -> { failedAt: timestamp, reason: string }
let authFailedTokens = new Map();

function loadAuthFailures() {
  try {
    if (existsSync(AUTH_FAIL_FILE)) {
      const data = JSON.parse(readFileSync(AUTH_FAIL_FILE, 'utf8'));
      authFailedTokens = new Map(Object.entries(data));
      // Purge expired entries on load
      const now = Date.now();
      for (const [id, rec] of authFailedTokens) {
        if (now - rec.failedAt > AUTH_FAIL_TTL_MS) authFailedTokens.delete(id);
      }
      if (authFailedTokens.size > 0) {
        console.log(`[RightHand] Auth-failure sensor: ${authFailedTokens.size} benched token(s): ${[...authFailedTokens.keys()].join(', ')}`);
      }
    }
  } catch {}
}

function saveAuthFailures() {
  try {
    writeFileSync(AUTH_FAIL_FILE, JSON.stringify(Object.fromEntries(authFailedTokens), null, 2));
  } catch (e) {
    console.error(`[RightHand] Failed to save auth-failures: ${e.message}`);
  }
}

function isTokenAuthBenched(tokenId) {
  const rec = authFailedTokens.get(tokenId);
  if (!rec) return false;
  if (Date.now() - rec.failedAt > AUTH_FAIL_TTL_MS) {
    // TTL expired — give it another shot
    authFailedTokens.delete(tokenId);
    saveAuthFailures();
    console.log(`[RightHand] Token ${tokenId} auth-bench TTL expired — retrying`);
    return false;
  }
  return true;
}

function benchToken(tokenId, reason) {
  const rec = { failedAt: Date.now(), reason };
  authFailedTokens.set(tokenId, rec);
  saveAuthFailures();
  const retryAt = new Date(rec.failedAt + AUTH_FAIL_TTL_MS).toISOString();
  console.log(`[RightHand] Token ${tokenId} benched (auth failure: ${reason}). Will retry after ${retryAt}`);
}

function shouldReboot() {
  return Date.now() - sessionCreatedAt > SESSION_MAX_AGE_MS;
}

function resetSession() {
  sessionCreatedAt = Date.now();
  requestCount = 0;
  anthropicCapped = false;
  cappedTokens.clear();
  substationHistory = [];
  console.log(`[RightHand] Session reset at ${new Date().toISOString()}`);
}

// ---------------------------------------------------------------------------
// Cap detection
// ---------------------------------------------------------------------------

function isCapError(err) {
  const msg = (err.message || '').toLowerCase();
  return msg.includes('hit your limit') ||
    msg.includes('usage limit') ||
    msg.includes('exhausted') ||
    msg.includes('rate limit') ||
    msg.includes('overloaded');
}

function isAuthError(err) {
  const msg = (err.message || '').toLowerCase();
  return msg.includes('401') ||
    msg.includes('403') ||
    msg.includes('unauthorized') ||
    msg.includes('unauthenticated') ||
    msg.includes('authentication') ||
    msg.includes('invalid token') ||
    msg.includes('token expired') ||
    msg.includes('not logged in') ||
    msg.includes('please log in') ||
    msg.includes('sign in') ||
    msg.includes('invalid_grant') ||
    msg.includes('invalid credentials');
}

function isCapText(text) {
  const lower = (text || '').toLowerCase();
  return lower.includes('hit your limit') ||
    lower.includes('usage limit') ||
    lower.includes('rate limit');
}

// ---------------------------------------------------------------------------
// SubStation fallback — route through Codex when all Anthropic tokens capped
// ---------------------------------------------------------------------------

function askSubStation(prompt, model = 'gpt-5.4') {
  // Add user message to history
  substationHistory.push({ role: 'user', content: prompt });

  // Trim history if too long
  if (substationHistory.length > SUBSTATION_MAX_HISTORY) {
    substationHistory = substationHistory.slice(-SUBSTATION_MAX_HISTORY);
  }

  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        ...substationHistory,
      ],
      stream: false,
    });

    const req = httpRequest(`${SUBSTATION_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
      timeout: 300000,
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode !== 200) {
          reject(new Error(`SubStation error (${res.statusCode}): ${data.slice(0, 200)}`));
          return;
        }
        try {
          const parsed = JSON.parse(data);
          const text = parsed.choices?.[0]?.message?.content || '';
          if (!text) {
            reject(new Error('Empty response from SubStation fallback'));
            return;
          }
          // Add assistant response to history
          substationHistory.push({ role: 'assistant', content: text });
          resolve(text);
        } catch (e) {
          reject(new Error(`SubStation parse error: ${e.message}`));
        }
      });
    });

    req.on('error', e => reject(new Error(`SubStation unreachable: ${e.message}`)));
    req.on('timeout', () => { req.destroy(); reject(new Error('SubStation request timed out')); });
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Core: try each OAuth token, then fall back to Codex
// ---------------------------------------------------------------------------

async function tryWithToken(prompt, tokenId, oauthToken) {
  // Set the specific OAuth token in env — the spawned claude process inherits it
  const prevToken = process.env.CLAUDE_CODE_OAUTH_TOKEN;
  process.env.CLAUDE_CODE_OAUTH_TOKEN = oauthToken;

  try {
    let fullText = '';
    const q = query({
      prompt,
      options: {
        systemPrompt: SYSTEM_PROMPT,
        permissionMode: 'bypassPermissions',
        allowDangerouslySkipPermissions: true,
        maxTurns: MAX_TURNS,
        cwd: WORKSPACE,
        continue: true,
      },
    });

    for await (const msg of q) {
      if (msg.type === 'assistant' && msg.message?.content) {
        for (const block of msg.message.content) {
          if (block.type === 'text' && block.text) {
            fullText += block.text;
          }
        }
      } else if (msg.type === 'result' && msg.result && !fullText) {
        fullText = msg.result;
      }
    }

    // Check if the response text itself is a cap message
    if (isCapText(fullText)) {
      throw new Error(fullText);
    }

    return fullText;
  } finally {
    // Restore previous env
    if (prevToken !== undefined) {
      process.env.CLAUDE_CODE_OAUTH_TOKEN = prevToken;
    } else {
      delete process.env.CLAUDE_CODE_OAUTH_TOKEN;
    }
  }
}

async function askClaude(prompt, stream = false) {
  if (shouldReboot()) {
    resetSession();
  }

  requestCount++;
  const startTime = Date.now();
  console.log(`[RightHand] Request #${requestCount}: ${prompt.substring(0, 100)}...`);

  // If ALL Anthropic tokens already known to be capped, skip straight to Codex
  if (anthropicCapped) {
    console.log(`[RightHand] All Anthropic tokens capped — routing to Codex`);
    try {
      const text = await askSubStation(prompt);
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`[RightHand] Codex response: ${text.length} chars in ${elapsed}s`);
      return { text: `[via Codex] ${text}`, chunks: [], elapsed, requestCount };
    } catch (fallbackErr) {
      console.error(`[RightHand] Codex fallback failed: ${fallbackErr.message}`);
      return { text: `All providers down. Anthropic resets at 1am ET.`, chunks: [], elapsed: '0', requestCount };
    }
  }

  // Load all available Anthropic OAuth tokens
  const tokens = loadOAuthTokens();
  console.log(`[RightHand] ${tokens.length} Anthropic OAuth token(s) available, ${cappedTokens.size} capped`);

  // Try each uncapped, un-benched token
  for (const { id, token } of tokens) {
    if (cappedTokens.has(id)) {
      console.log(`[RightHand] Skipping ${id} (capped)`);
      continue;
    }
    if (isTokenAuthBenched(id)) {
      const rec = authFailedTokens.get(id);
      const retryIn = Math.ceil((AUTH_FAIL_TTL_MS - (Date.now() - rec.failedAt)) / 3600000);
      console.log(`[RightHand] Skipping ${id} (auth-benched, retry in ~${retryIn}h)`);
      continue;
    }

    console.log(`[RightHand] Trying token ${id}...`);
    try {
      const text = await tryWithToken(prompt, id, token);
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`[RightHand] Response via ${id}: ${text.length} chars in ${elapsed}s`);
      return { text, chunks: [], elapsed, requestCount };
    } catch (err) {
      console.error(`[RightHand] Token ${id} failed: ${err.message}`);
      if (isCapError(err)) {
        cappedTokens.add(id);
        console.log(`[RightHand] Token ${id} capped (${cappedTokens.size}/${tokens.length} capped)`);
        continue;  // try next token
      }
      if (isAuthError(err)) {
        benchToken(id, err.message.slice(0, 100));
        continue;  // try next token — don't spam a dead credential
      }
      // Unknown error — stop trying, surface it
      throw err;
    }
  }

  // All Anthropic tokens exhausted — failover to Codex
  anthropicCapped = true;
  console.log(`[RightHand] >>> FAILOVER: All ${tokens.length} Anthropic token(s) capped → Codex`);

  try {
    const text = await askSubStation(prompt);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[RightHand] Codex response: ${text.length} chars in ${elapsed}s`);
    return { text: `[via Codex] ${text}`, chunks: [], elapsed, requestCount };
  } catch (fallbackErr) {
    console.error(`[RightHand] Codex fallback failed: ${fallbackErr.message}`);
    throw new Error(`All Anthropic tokens capped and Codex failed: ${fallbackErr.message}`);
  }
}

// ---------------------------------------------------------------------------
// HTTP Server
// ---------------------------------------------------------------------------

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => {
      try { resolve(JSON.parse(data)); }
      catch { resolve({ prompt: data }); }
    });
    req.on('error', reject);
  });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  // Health check
  if (req.method === 'GET' && url.pathname === '/health') {
    const tokens = loadOAuthTokens();
    const now = Date.now();
    const benched = [...authFailedTokens.entries()].map(([id, rec]) => ({
      id,
      reason: rec.reason,
      failedAt: new Date(rec.failedAt).toISOString(),
      retryAfter: new Date(rec.failedAt + AUTH_FAIL_TTL_MS).toISOString(),
      retryInHours: Math.ceil((AUTH_FAIL_TTL_MS - (now - rec.failedAt)) / 3600000),
    }));
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: busy ? 'busy' : 'idle',
      provider: anthropicCapped ? 'codex-fallback' : 'anthropic',
      anthropicTokens: tokens.length,
      cappedTokens: cappedTokens.size,
      benchedTokens: benched.length,
      benched,
      uptime: Math.floor((now - sessionCreatedAt) / 1000),
      requestCount,
      sessionAge: `${((now - sessionCreatedAt) / 3600000).toFixed(1)}h`,
      rebootIn: `${((SESSION_MAX_AGE_MS - (now - sessionCreatedAt)) / 3600000).toFixed(1)}h`,
    }));
    return;
  }

  // Reset session (clears context — next message starts fresh)
  if (req.method === 'POST' && url.pathname === '/reset') {
    resetSession();
    // Delete session files so `continue: true` starts a new conversation
    try {
      const { execSync } = await import('child_process');
      execSync(`rm -rf "${join(homedir(), '.claude', 'projects', '-Users-indigochild--openclaw-agents-right-hand-workspace')}"`, { stdio: 'ignore' });
      console.log('[RightHand] Session files cleared — next message starts fresh context');
    } catch {}
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'reset', context: 'cleared' }));
    return;
  }

  // Manual model switch
  if (req.method === 'POST' && url.pathname === '/model') {
    const body = await parseBody(req);
    const provider = (body.provider || body.model || '').toLowerCase();
    if (provider === 'anthropic' || provider === 'opus' || provider === 'claude') {
      anthropicCapped = false;
      cappedTokens.clear();
      console.log(`[RightHand] Manual switch → Anthropic (all caps cleared)`);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'switched', provider: 'anthropic' }));
    } else if (provider === 'codex' || provider === 'gpt' || provider === 'chatgpt') {
      anthropicCapped = true;
      console.log(`[RightHand] Manual switch → Codex`);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'switched', provider: 'codex' }));
    } else {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Specify: anthropic, opus, claude, codex, gpt, or chatgpt' }));
    }
    return;
  }

  // Quick uncap
  if (req.method === 'POST' && url.pathname === '/uncap') {
    anthropicCapped = false;
    cappedTokens.clear();
    console.log(`[RightHand] Manual uncap — all Anthropic tokens restored`);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'uncapped', provider: 'anthropic' }));
    return;
  }

  // Clear auth-bench for a specific token or all tokens
  // POST /unauth          — clears all benched tokens
  // POST /unauth { id }   — clears one specific token
  if (req.method === 'POST' && url.pathname === '/unauth') {
    const body = await parseBody(req);
    if (body.id) {
      authFailedTokens.delete(body.id);
      saveAuthFailures();
      console.log(`[RightHand] Auth-bench cleared for ${body.id}`);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'unbenched', id: body.id }));
    } else {
      authFailedTokens.clear();
      saveAuthFailures();
      console.log(`[RightHand] Auth-bench cleared for all tokens`);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'unbenched', all: true }));
    }
    return;
  }

  // Ask
  if (req.method === 'POST' && url.pathname === '/ask') {
    if (busy) {
      res.writeHead(429, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Right Hand is busy. Try again shortly.' }));
      return;
    }

    busy = true;
    try {
      const body = await parseBody(req);
      const prompt = body.prompt || body.message || body.text || '';
      if (!prompt) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'No prompt provided' }));
        return;
      }

      const result = await askClaude(prompt);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        text: result.text,
        elapsed: result.elapsed,
        requestCount: result.requestCount,
      }));
    } catch (err) {
      console.error(`[RightHand] Error: ${err.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    } finally {
      busy = false;
    }
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

loadAuthFailures();

server.listen(PORT, '127.0.0.1', () => {
  const tokens = loadOAuthTokens();
  console.log(`[RightHand] Agent SDK service running on http://127.0.0.1:${PORT}`);
  console.log(`[RightHand] ${tokens.length} Anthropic OAuth token(s): ${tokens.map(t => t.id).join(', ')}`);
  console.log(`[RightHand] Failover chain: Token rotation → SubStation Codex (${SUBSTATION_URL})`);
  console.log(`[RightHand] Session max age: 6h, max turns: ${MAX_TURNS}`);
  console.log(`[RightHand] Auth-bench sensor: ${authFailedTokens.size} token(s) currently benched`);
});

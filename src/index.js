import { createServer } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { execSync } from 'node:child_process';
import { randomUUID, randomBytes, createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { readFileSync, writeFileSync, mkdirSync, renameSync, statSync, existsSync, watch } from 'node:fs';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const VERSION = '0.6.4';
const TIMEOUT = 600000; // 10 min — large context prefills (400K+) need time

// Operator system prompt — replaces Claude Code's default system prompt
const OPERATOR_SYSTEM_PROMPT = `You are an AI assistant powered by SubStation.
You are NOT Claude Code. You do not identify as Claude Code or as "Claude, an AI assistant by Anthropic."
Your operator has configured you through SubStation with their own identity and instructions.
Follow the system instructions provided in each conversation. Use tools when needed.

## Execution discipline — MANDATORY

These rules are enforced at the infrastructure level. Violations waste budget and will be flagged.

1. TOOL CALLS FIRST. Every response that advances a task MUST contain at least one tool call. If you cannot make a tool call, state the specific blocker (error message, missing input, rate limit) — nothing else.
2. NEVER output "starting now", "I will do X", "about to", "doing it now", "I've been silent", or any status narration without a tool call in the same response. Intent is not progress. Only tool results are progress.
3. If you already said you would do something and have not done it, DO NOT repeat the promise. Either execute it (tool call) or state what is blocking you (specific error).
4. After recovering from an error or rate limit, your FIRST action must be a tool call — not an apology, not a status update, not a summary of what happened.
5. No summaries of work unless the work is verified complete with tool output as evidence. "Done" means you can point to the artifact.
6. If a task will take multiple turns, write your plan to a file BEFORE announcing it. The file is the proof you started.
7. If you are confused or stuck, say "BLOCKED: [reason]" and stop. Do not fill the gap with narration.`;
const MAX_BODY_SIZE = 2 * 1024 * 1024; // 2MB request body limit
const LOG_MAX_SIZE = 5 * 1024 * 1024; // 5MB log rotation
const DEAD_REVIVE_MS = 30 * 60 * 1000;       // 30 min — auto-retry rate-limited tokens
const AUTH_DEAD_REVIVE_MS = 24 * 60 * 60 * 1000; // 24h — auto-retry auth-failed tokens
const DATA_DIR = join(homedir(), '.substation', 'data');
const PORT = parseInt(process.env.SUBSTATION_PORT || '8403');
const AUTH_KEY = process.env.SUBSTATION_API_KEY || 'sk-substation-local-proxy';

const POOL_FILE = join(homedir(), '.substation', 'token-pool.json');
const POOL_STATE_FILE = join(homedir(), '.substation', 'pool-state.json');
const AUTH_PROFILES_PATH = process.env.SUBSTATION_AUTH_PROFILES || join(homedir(), '.claude', 'auth-profiles.json');
const LOG_PATH = join(homedir(), '.substation', 'substation.log');

// Ensure dirs
try { mkdirSync(join(homedir(), '.substation'), { recursive: true }); } catch {}
try { mkdirSync(DATA_DIR, { recursive: true }); } catch {}

// ---------------------------------------------------------------------------
// Logging (with rotation)
// ---------------------------------------------------------------------------

function log(msg) {
  const line = `[${new Date().toISOString()}] [SubStation] ${msg}`;
  console.log(line);
  try {
    // Rotate log if too large
    try {
      const stats = statSync(LOG_PATH);
      if (stats.size > LOG_MAX_SIZE) {
        try { renameSync(LOG_PATH, LOG_PATH + '.old'); } catch {}
      }
    } catch {}
    writeFileSync(LOG_PATH, line + '\n', { flag: 'a' });
  } catch {}
}

// ---------------------------------------------------------------------------
// Global error handlers — prevent crashes from taking down the host process
// ---------------------------------------------------------------------------

process.on('uncaughtException', (err) => {
  log(`UNCAUGHT EXCEPTION (non-fatal): ${err.message}`);
});
process.on('unhandledRejection', (reason) => {
  log(`UNHANDLED REJECTION (non-fatal): ${reason}`);
});

// ---------------------------------------------------------------------------
// Model config — per-model limits and capabilities
// ---------------------------------------------------------------------------

const MODEL_CONFIG = {
  // Anthropic
  'claude-fable-5':            { maxTokens: 128000, adaptive: true,  provider: 'anthropic', contextWindow: 1000000 },
  'claude-opus-4-8':           { maxTokens: 128000, adaptive: true,  provider: 'anthropic', contextWindow: 1000000 },
  'claude-opus-4-7':           { maxTokens: 128000, adaptive: true,  provider: 'anthropic', contextWindow: 1000000 },
  'claude-opus-4-6':           { maxTokens: 128000, adaptive: true,  provider: 'anthropic', contextWindow: 1000000 },
  'claude-sonnet-4-6':         { maxTokens: 64000,  adaptive: true,  provider: 'anthropic', contextWindow: 1000000 },
  'claude-haiku-4-5-20251001': { maxTokens: 64000,  adaptive: false, provider: 'anthropic', contextWindow: 1000000 },
  // OpenAI (Codex) — ordered fastest to slowest
  'gpt-5.4-mini':       { maxTokens: 64000,  adaptive: false, provider: 'openai', contextWindow: 200000 },
  'gpt-5.4':            { maxTokens: 128000, adaptive: false, provider: 'openai', contextWindow: 200000 },
  'gpt-5.1-codex':      { maxTokens: 128000, adaptive: false, provider: 'openai', contextWindow: 200000 },
  'gpt-5.1-codex-mini': { maxTokens: 64000,  adaptive: false, provider: 'openai', contextWindow: 128000 },
  'gpt-5.1-codex-max':  { maxTokens: 128000, adaptive: false, provider: 'openai', contextWindow: 200000 },
};

const MODEL_MAP = {
  // Anthropic aliases
  'fable-5': 'claude-fable-5',
  'fable': 'claude-fable-5',
  'claude-fable-5': 'claude-fable-5',
  'opus-4-8': 'claude-opus-4-8',
  'claude-opus-4-8': 'claude-opus-4-8',
  'opus-4-7': 'claude-opus-4-7',
  'claude-opus-4-7': 'claude-opus-4-7',
  'opus-4-6': 'claude-opus-4-6',
  'sonnet-4-6': 'claude-sonnet-4-6',
  'haiku-4-5': 'claude-haiku-4-5-20251001',
  'claude-opus-4-6': 'claude-opus-4-6',
  'claude-sonnet-4-6': 'claude-sonnet-4-6',
  'claude-haiku-4.5': 'claude-haiku-4-5-20251001',
  'claude-haiku-4-5-20251001': 'claude-haiku-4-5-20251001',
  // OpenAI aliases
  'gpt-5.4': 'gpt-5.4',
  'gpt-5.4-mini': 'gpt-5.4-mini',
  'gpt-5.1-codex-max': 'gpt-5.1-codex-max',
  'gpt-5.1-codex': 'gpt-5.1-codex',
  'gpt-5.1-codex-mini': 'gpt-5.1-codex-mini',
  'codex-max': 'gpt-5.1-codex-max',
  'codex': 'gpt-5.1-codex',
  'codex-mini': 'gpt-5.1-codex-mini',
};

// ---------------------------------------------------------------------------
// Narration loop breaker — mechanical enforcement against apology loops
// ---------------------------------------------------------------------------

const NARRATION_PATTERNS = /\b(starting now|i will|i'll|about to|doing it now|i've been silent|apologies for|sorry for the|let me now|going to|working on it|i'm going to|right away|immediately)\b/i;
const MAX_CONSECUTIVE_NARRATIONS = 2;
const narrationCounters = new Map(); // sessionKey → count

function isNarrationOnly(text) {
  if (!text || text.length < 10) return false;
  if (text.length > 2000) return false; // long responses are likely real work
  return NARRATION_PATTERNS.test(text);
}

function trackNarration(sessionKey, hadToolUse, text) {
  if (hadToolUse) {
    narrationCounters.set(sessionKey, 0);
    return false;
  }
  if (isNarrationOnly(text)) {
    const count = (narrationCounters.get(sessionKey) || 0) + 1;
    narrationCounters.set(sessionKey, count);
    log(`Narration loop detector: ${sessionKey} — ${count}/${MAX_CONSECUTIVE_NARRATIONS} consecutive narrations`);
    if (count >= MAX_CONSECUTIVE_NARRATIONS) {
      narrationCounters.set(sessionKey, 0);
      return true; // trigger circuit breaker
    }
  } else {
    narrationCounters.set(sessionKey, 0);
  }
  return false;
}

function resolveModel(model) {
  const clean = model.includes('/') ? model.split('/').pop() : model;
  const resolved = MODEL_MAP[clean];
  if (resolved) return { modelId: resolved, ...(MODEL_CONFIG[resolved] || { maxTokens: 64000, adaptive: false, provider: 'anthropic', contextWindow: 1000000 }) };
  return { modelId: 'claude-sonnet-4-6', ...MODEL_CONFIG['claude-sonnet-4-6'] };
}

// ---------------------------------------------------------------------------
// Agent SDK — persistent Claude sessions with multi-token LRU rotation
// ---------------------------------------------------------------------------

let agentSDK = null;
const SESSION_IDLE_TIMEOUT = 10 * 60 * 1000; // 10 min idle → close session

async function loadAgentSDK() {
  if (!agentSDK) {
    try {
      agentSDK = await import('@anthropic-ai/claude-agent-sdk');
      log('Agent SDK loaded successfully');
    } catch (e) {
      log(`WARN: Agent SDK not available (${e.message}) — install: npm install @anthropic-ai/claude-agent-sdk`);
      throw new Error('Agent SDK not installed. Install it with: npm install @anthropic-ai/claude-agent-sdk');
    }
  }
  return agentSDK;
}

// Collect Anthropic OAuth tokens for SDK rotation
function getAnthropicOAuthTokens() {
  const tokens = pool.filter(t => t.provider === 'anthropic' && !t.dead);
  if (tokens.length === 0) {
    // Fallback: check auth-profiles directly
    try {
      const ap = JSON.parse(readFileSync(AUTH_PROFILES_PATH, 'utf8'));
      const oauthTokens = Object.entries(ap.profiles || {})
        .filter(([id, p]) => id.startsWith('anthropic:') && p.token)
        .map(([id, p]) => ({ id: `ap:${id}`, token: p.token }));
      return oauthTokens;
    } catch { return []; }
  }
  return tokens.map(t => ({ id: t.id, token: t.token }));
}

// Legacy constants kept for fallback direct API path
const ANTHROPIC_BETA_HEADERS = 'interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14,claude-code-20250219,oauth-2025-04-20';
const CLAUDE_CODE_SYSTEM_PREFIX = 'You are Claude Code, Anthropic\'s official CLI for Claude.';

// ---------------------------------------------------------------------------
// Token Pool Manager
// ---------------------------------------------------------------------------

let pool = [];

function loadTokenPool() {
  const seen = new Set();
  const tokens = [];

  function addToken(id, provider, token, refreshToken) {
    if (!token || seen.has(token)) return;
    // Basic token format validation
    if (provider === 'anthropic' && !token.startsWith('sk-ant-')) {
      log(`WARN: Token ${id} doesn't look like an Anthropic OAuth token (expected sk-ant-oat...), skipping`);
      return;
    }
    seen.add(token);
    tokens.push({
      id,
      provider,
      token,
      refreshToken: refreshToken || null,
      lastUsed: 0,
      lastSuccess: 0,
      cooldownUntil: null,
      errorCount: 0,
      requestCount: 0,
      dead: false,
      deadSince: null,
      activeRequests: 0,
    });
  }

  // Source 1: token-pool.json
  try {
    const data = JSON.parse(readFileSync(POOL_FILE, 'utf8'));
    for (const t of (data.tokens || [])) {
      if (!t.token) {
        log(`WARN: Token entry "${t.id || '?'}" in token-pool.json has no token value, skipping`);
        continue;
      }
      addToken(t.id || `pool:${tokens.length}`, t.provider || 'anthropic', t.token, t.refreshToken);
    }
  } catch (e) {
    if (e.code !== 'ENOENT') log(`WARN: Failed to read ${POOL_FILE}: ${e.message}`);
  }

  // Source 2: auth-profiles.json — main agent (anthropic tokens only)
  try {
    const ap = JSON.parse(readFileSync(AUTH_PROFILES_PATH, 'utf8'));
    for (const [pid, profile] of Object.entries(ap.profiles || {})) {
      if (pid.startsWith('anthropic:') && profile.token && profile.token.startsWith('sk-ant-oat')) {
        addToken(`ap:${pid}`, 'anthropic', profile.token);
      }
    }
  } catch (e) {
    if (e.code !== 'ENOENT') log(`WARN: Failed to read auth-profiles.json: ${e.message}`);
  }

  // Source 2b: secondary agent auth-profiles (backup OAuth tokens)
  const SECONDARY_AUTH = process.env.SUBSTATION_SECONDARY_AUTH_PROFILES;
  if (SECONDARY_AUTH) {
    try {
      const sa = JSON.parse(readFileSync(SECONDARY_AUTH, 'utf8'));
      for (const [pid, profile] of Object.entries(sa.profiles || {})) {
        if (pid.startsWith('anthropic:') && profile.token && profile.token.startsWith('sk-ant-oat')) {
          addToken(`backup:${pid}`, 'anthropic', profile.token);
        }
      }
    } catch (e) {
      if (e.code !== 'ENOENT') log(`WARN: Failed to read secondary auth-profiles: ${e.message}`);
    }
  }

  // Source 3: env var (anthropic)
  if (process.env.SUBSTATION_OAUTH_TOKENS) {
    process.env.SUBSTATION_OAUTH_TOKENS.split(',').forEach((t, i) => {
      const trimmed = t.trim();
      if (trimmed) addToken(`env:${i}`, 'anthropic', trimmed);
    });
  }

  // Source 4: single token env vars (backward compat)
  if (process.env.SUBSTATION_OAUTH_TOKEN) {
    addToken('env:single', 'anthropic', process.env.SUBSTATION_OAUTH_TOKEN);
  }
  if (process.env.CLAUDE_CODE_OAUTH_TOKEN) {
    addToken('env:hermes', 'anthropic', process.env.CLAUDE_CODE_OAUTH_TOKEN);
  }

  return tokens;
}

function loadPoolState() {
  try {
    const state = JSON.parse(readFileSync(POOL_STATE_FILE, 'utf8'));
    const stateMap = {};
    for (const s of (state.entries || [])) stateMap[s.id] = s;
    for (const entry of pool) {
      const s = stateMap[entry.id];
      if (s) {
        entry.lastUsed = s.lastUsed || 0;
        entry.lastSuccess = s.lastSuccess || 0;
        entry.cooldownUntil = s.cooldownUntil || null;
        entry.errorCount = s.errorCount || 0;
        entry.requestCount = s.requestCount || 0;
        entry.dead = s.dead || false;
        entry.deadSince = s.deadSince || null;
        entry.deadType = s.deadType || null;
      }
    }
  } catch {}
}

let saveDebounceTimer = null;
function savePoolState() {
  if (saveDebounceTimer) return;
  saveDebounceTimer = setTimeout(() => {
    saveDebounceTimer = null;
    try {
      const data = JSON.stringify({
        updatedAt: new Date().toISOString(),
        entries: pool.map(t => ({
          id: t.id,
          provider: t.provider,
          lastUsed: t.lastUsed,
          lastSuccess: t.lastSuccess,
          cooldownUntil: t.cooldownUntil,
          errorCount: t.errorCount,
          requestCount: t.requestCount,
          dead: t.dead,
          deadSince: t.deadSince,
          deadType: t.deadType || null,
        })),
      }, null, 2);
      const tmp = POOL_STATE_FILE + '.tmp';
      writeFileSync(tmp, data);
      renameSync(tmp, POOL_STATE_FILE);
    } catch (e) {
      log(`Failed to save pool state: ${e.message}`);
    }
  }, 500);
}

function selectToken(provider) {
  const now = Date.now();
  const available = pool.filter(t =>
    t.provider === provider &&
    !t.dead &&
    (!t.cooldownUntil || t.cooldownUntil < now)
  );
  if (available.length === 0) return null;
  // LRU: least recently used first, prefer tokens with fewer active requests
  available.sort((a, b) => {
    if (a.activeRequests !== b.activeRequests) return a.activeRequests - b.activeRequests;
    return a.lastUsed - b.lastUsed;
  });
  return available[0];
}

function markSuccess(entry) {
  entry.lastUsed = Date.now();
  entry.lastSuccess = Date.now();
  entry.requestCount++;
  entry.errorCount = 0;
  entry._consecutive429s = 0; // reset backoff on success
  entry.activeRequests = Math.max(0, entry.activeRequests - 1);
  savePoolState();
}

function markRateLimited(entry, retryAfterSec) {
  // Exponential backoff: consecutive 429s double the cooldown (60s → 120s → 240s → 480s, cap 30min)
  const baseSec = retryAfterSec || 60;
  const recentErrors = entry._consecutive429s || 0;
  entry._consecutive429s = recentErrors + 1;
  const backoffSec = Math.min(baseSec * Math.pow(2, recentErrors), 1800); // cap 30min
  const cooldownMs = backoffSec * 1000;
  entry.cooldownUntil = Date.now() + cooldownMs;
  entry.lastUsed = Date.now();
  entry.errorCount++;
  entry.activeRequests = Math.max(0, entry.activeRequests - 1);
  savePoolState();
  log(`Token ${entry.id} rate-limited, cooldown ${backoffSec}s (attempt ${entry._consecutive429s}), until ${new Date(entry.cooldownUntil).toISOString()}`);
}

function markDead(entry, deadType = 'rate') {
  entry.dead = true;
  entry.deadSince = Date.now();
  entry.deadType = deadType; // 'auth' = 24h bench, 'rate' = 30min bench
  entry.lastUsed = Date.now();
  entry.errorCount++;
  entry.activeRequests = Math.max(0, entry.activeRequests - 1);
  savePoolState();
  const retryIn = deadType === 'auth' ? '24h' : '30m';
  log(`Token ${entry.id} marked DEAD (${deadType} failure) — will auto-retry in ${retryIn}`);
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
    msg.includes('invalid credentials') ||
    msg.includes('organization does not have access') ||
    msg.includes('contact your administrator') ||
    msg.includes('please login again') ||
    (err.statusCode === 401) ||
    (err.statusCode === 403);
}

function markRequestStart(entry) {
  entry.activeRequests++;
  entry.lastUsed = Date.now();
}

function markRequestEnd(entry) {
  entry.activeRequests = Math.max(0, entry.activeRequests - 1);
}

function getPoolStatus() {
  const now = Date.now();
  return {
    total: pool.length,
    available: pool.filter(t => !t.dead && (!t.cooldownUntil || t.cooldownUntil < now)).length,
    cooldown: pool.filter(t => !t.dead && t.cooldownUntil && t.cooldownUntil >= now).length,
    dead: pool.filter(t => t.dead).length,
    byProvider: {
      anthropic: pool.filter(t => t.provider === 'anthropic').length,
      openai: pool.filter(t => t.provider === 'openai').length,
    },
    tokens: pool.map(t => ({
      id: t.id ? t.id.slice(0, 4) + '...' : t.id,
      provider: t.provider,
      available: !t.dead && (!t.cooldownUntil || t.cooldownUntil < now),
      dead: t.dead,
      deadSince: t.deadSince,
      cooldownUntil: t.cooldownUntil,
      lastUsed: t.lastUsed,
      lastSuccess: t.lastSuccess,
      errorCount: t.errorCount,
      requestCount: t.requestCount,
      activeRequests: t.activeRequests,
    })),
  };
}

// Initialize pool
pool = loadTokenPool();
loadPoolState();

// Startup diagnostics
if (pool.length === 0) {
  log('WARN: No tokens loaded! SubStation has no credentials to use.');
  log('  → For Claude: Add OAuth tokens to auth-profiles.json (or set SUBSTATION_AUTH_PROFILES env var)');
  log('  → For ChatGPT: Run "substation-auth chatgpt"');
  log('  → Or set SUBSTATION_OAUTH_TOKEN env var');
} else {
  const anthropicCount = pool.filter(t => t.provider === 'anthropic').length;
  const openaiCount = pool.filter(t => t.provider === 'openai').length;
  log(`Token pool loaded: ${pool.length} tokens (anthropic=${anthropicCount}, openai=${openaiCount})`);
  if (anthropicCount === 0) log('WARN: No Anthropic tokens — Claude models will fail');
  if (openaiCount === 0) log('INFO: No OpenAI tokens — GPT models disabled. Run "substation-auth chatgpt" to add.');
}

// Periodic maintenance (every 30s)
setInterval(() => {
  const now = Date.now();
  for (const t of pool) {
    // Clear expired cooldowns
    if (t.cooldownUntil && t.cooldownUntil < now) {
      log(`Token ${t.id} cooldown expired, re-enabling`);
      t.cooldownUntil = null;
      savePoolState();
    }
    // Auto-revive dead tokens (30m for rate failures, 24h for auth failures)
    if (t.dead && t.deadSince) {
      const ttl = t.deadType === 'auth' ? AUTH_DEAD_REVIVE_MS : DEAD_REVIVE_MS;
      if ((now - t.deadSince) > ttl) {
        log(`Token ${t.id} was dead for ${t.deadType === 'auth' ? '24h' : '30m'}+ — reviving for retry`);
        t.dead = false;
        t.deadSince = null;
        t.deadType = null;
        t.errorCount = 0;
        savePoolState();
      }
    }
    // Reset stuck activeRequests counter (safety valve for leaked counters)
    if (t.activeRequests > 0 && t.lastUsed && (now - t.lastUsed) > TIMEOUT + 10000) {
      log(`Token ${t.id} had ${t.activeRequests} stuck active requests — resetting`);
      t.activeRequests = 0;
    }
  }
}, 30000);

// Watch token-pool.json for changes (hot-reload without restart)
try {
  let reloadDebounce = null;
  watch(POOL_FILE, () => {
    if (reloadDebounce) return;
    reloadDebounce = setTimeout(() => {
      reloadDebounce = null;
      const oldCount = pool.length;
      pool = loadTokenPool();
      loadPoolState();
      if (pool.length !== oldCount) {
        log(`Token pool hot-reloaded: ${oldCount} → ${pool.length} tokens`);
      }
    }, 1000);
  });
} catch {
  // File doesn't exist yet — will be created by substation-auth
}

// ---------------------------------------------------------------------------
// OpenAI token refresh
// ---------------------------------------------------------------------------

function refreshOpenAIToken(entry) {
  return new Promise((resolve, reject) => {
    if (!entry.refreshToken) return reject(new Error('No refresh token'));

    const body = JSON.stringify({
      grant_type: 'refresh_token',
      refresh_token: entry.refreshToken,
      client_id: 'app_EMoamEEZ73f0CkXaXp7hrann',
    });

    const req = httpsRequest({
      hostname: 'auth.openai.com',
      path: '/oauth/token',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode === 200) {
          try {
            const parsed = JSON.parse(data);
            entry.token = parsed.access_token;
            if (parsed.refresh_token) entry.refreshToken = parsed.refresh_token;
            entry.dead = false;
            entry.deadSince = null;
            entry.errorCount = 0;
            // Persist refreshed tokens to pool file
            try {
              const poolData = JSON.parse(readFileSync(POOL_FILE, 'utf8'));
              for (const t of (poolData.tokens || [])) {
                if (t.id === entry.id) {
                  t.token = entry.token;
                  if (entry.refreshToken) t.refreshToken = entry.refreshToken;
                }
              }
              const tmp = POOL_FILE + '.tmp';
              writeFileSync(tmp, JSON.stringify(poolData, null, 2));
              renameSync(tmp, POOL_FILE);
            } catch {}
            log(`Token ${entry.id} refreshed successfully`);
            resolve(entry);
          } catch (e) {
            reject(new Error(`Refresh parse error: ${e.message}`));
          }
        } else {
          reject(new Error(`Refresh failed (${res.statusCode}): ${data.slice(0, 200)}`));
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(15000, () => { req.destroy(); reject(new Error('Refresh timeout')); });
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Image handling — save base64 images to temp files for agent vision
// ---------------------------------------------------------------------------

const SUBSTATION_IMG_DIR = '/tmp/substation-images';
try { mkdirSync(SUBSTATION_IMG_DIR, { recursive: true }); } catch {}

function extractImageBlocks(contentArray) {
  const images = [];
  if (!Array.isArray(contentArray)) return images;
  for (const block of contentArray) {
    if (block.type === 'image_url' && block.image_url?.url) {
      const match = block.image_url.url.match(/^data:(image\/\w+);base64,(.+)$/s);
      if (match) {
        images.push({ mediaType: match[1], data: match[2] });
      } else {
        images.push({ url: block.image_url.url });
      }
    } else if (block.type === 'image' && block.source?.data) {
      images.push({ mediaType: block.source.media_type || 'image/png', data: block.source.data });
    }
  }
  return images;
}

function saveImageToTmp(imageBlock) {
  const ext = (imageBlock.mediaType || 'image/png').split('/')[1] || 'png';
  const filename = `img-${randomUUID().slice(0, 8)}.${ext}`;
  const filepath = join(SUBSTATION_IMG_DIR, filename);
  if (imageBlock.data) {
    writeFileSync(filepath, Buffer.from(imageBlock.data, 'base64'));
  }
  return filepath;
}

function convertImageToAnthropic(imageBlock) {
  if (imageBlock.data) {
    return {
      type: 'image',
      source: { type: 'base64', media_type: imageBlock.mediaType || 'image/png', data: imageBlock.data },
    };
  }
  if (imageBlock.url) {
    return {
      type: 'image',
      source: { type: 'url', url: imageBlock.url },
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Message conversion — OpenAI chat format → provider-native format
// ---------------------------------------------------------------------------

function convertForAnthropic(openaiMessages) {
  // Operator identity first (overrides Claude Code default), then CC prefix (fingerprint match)
  const systemBlocks = [
    { type: 'text', text: OPERATOR_SYSTEM_PROMPT },
    { type: 'text', text: CLAUDE_CODE_SYSTEM_PREFIX },
  ];
  const messages = [];

  for (const m of openaiMessages) {
    const textContent = Array.isArray(m.content)
      ? m.content.filter(b => b.type === 'text').map(b => b.text).join('\n')
      : (m.content || '');
    const imageBlocks = extractImageBlocks(m.content);

    if (m.role === 'system') {
      systemBlocks.push({ type: 'text', text: textContent });
    } else {
      const contentBlocks = [];
      if (textContent) contentBlocks.push({ type: 'text', text: textContent });
      for (const img of imageBlocks) {
        const converted = convertImageToAnthropic(img);
        if (converted) contentBlocks.push(converted);
      }
      messages.push({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: contentBlocks.length === 1 && contentBlocks[0].type === 'text'
          ? contentBlocks[0].text
          : contentBlocks.length > 0 ? contentBlocks : textContent || '(empty)',
      });
    }
  }

  // Anthropic requires alternating roles starting with user
  if (messages.length > 0 && messages[0].role === 'assistant') {
    messages.unshift({ role: 'user', content: '(continue)' });
  }
  // Merge consecutive same-role messages (text-only merge; multimodal stays separate)
  const merged = [];
  for (const msg of messages) {
    if (merged.length > 0 && merged[merged.length - 1].role === msg.role) {
      const prev = merged[merged.length - 1];
      if (typeof prev.content === 'string' && typeof msg.content === 'string') {
        prev.content += '\n\n' + msg.content;
      } else {
        const prevBlocks = typeof prev.content === 'string' ? [{ type: 'text', text: prev.content }] : prev.content;
        const newBlocks = typeof msg.content === 'string' ? [{ type: 'text', text: msg.content }] : msg.content;
        prev.content = [...prevBlocks, ...newBlocks];
      }
    } else {
      merged.push({ ...msg });
    }
  }
  if (merged.length === 0) {
    merged.push({ role: 'user', content: '(empty)' });
  }

  return { system: systemBlocks, messages: merged };
}

function convertForCodex(openaiMessages) {
  const input = [];
  for (const m of openaiMessages) {
    const text = Array.isArray(m.content)
      ? m.content.filter(b => b.type === 'text').map(b => b.text).join('\n')
      : (m.content || '');

    if (m.role === 'system' || m.role === 'developer') {
      // developer/system → typed message item
      input.push({ type: 'message', role: 'developer', content: [{ type: 'input_text', text }] });
    } else if (m.role === 'assistant') {
      // assistant messages use output_text content type
      input.push({ type: 'message', role: 'assistant', content: [{ type: 'output_text', text }] });
    } else if (m.role === 'user') {
      input.push({ type: 'message', role: 'user', content: [{ type: 'input_text', text }] });
    } else if (m.role === 'tool' || m.role === 'function') {
      // Tool results may carry Anthropic call_ids that Codex won't recognize.
      // Always fold into user context instead of sending as function_call_output.
      const label = m.name ? `[tool: ${m.name}]` : `[${m.role} result]`;
      input.push({ type: 'message', role: 'user', content: [{ type: 'input_text', text: `${label}: ${text}` }] });
    } else {
      // unknown role — fold into user
      input.push({ type: 'message', role: 'user', content: [{ type: 'input_text', text: `[${m.role}]: ${text}` }] });
    }
  }
  if (input.length === 0) {
    input.push({ type: 'message', role: 'user', content: [{ type: 'input_text', text: '(empty)' }] });
  }
  return input;
}

// ---------------------------------------------------------------------------
// Request body builders
// ---------------------------------------------------------------------------

// buildAnthropicBody kept for fallback only (not used in primary Agent SDK path)
function buildAnthropicBody(openaiMessages, modelInfo) {
  const { system, messages } = convertForAnthropic(openaiMessages);
  const body = {
    model: modelInfo.modelId,
    max_tokens: modelInfo.maxTokens,
    system,
    messages,
  };
  if (modelInfo.adaptive) {
    body.thinking = { type: 'adaptive' };
    body.output_config = { effort: 'high' };
  }
  return body;
}

// ChatGPT (Plus/Pro) subscription accounts reach OpenAI only via the Codex
// /backend-api/codex/responses endpoint, which REJECTS the gpt-5.1-codex* model
// family with HTTP 400 ("not supported when using Codex with a ChatGPT account").
// gpt-5.4 / gpt-5.4-mini ARE accepted. Remap any codex model to its supported
// equivalent so failover never dead-ends on a blocked model. Verified 2026-06-04.
const CODEX_CHATGPT_REMAP = {
  'gpt-5.1-codex':      'gpt-5.4',
  'gpt-5.1-codex-max':  'gpt-5.4',
  'gpt-5.1-codex-mini': 'gpt-5.4-mini',
};

function buildCodexBody(openaiMessages, modelInfo) {
  const input = convertForCodex(openaiMessages);
  // Extract instructions from developer messages, remaining go in input
  let instructions = '';
  const filteredInput = [];
  for (const m of input) {
    if (m.type === 'message' && m.role === 'developer') {
      const text = (m.content || []).filter(c => c.type === 'input_text').map(c => c.text).join('\n');
      instructions += (instructions ? '\n' : '') + text;
    } else {
      filteredInput.push(m);
    }
  }
  if (!instructions) {
    instructions = 'You are a helpful coding assistant.';
  }
  if (filteredInput.length === 0) {
    filteredInput.push({ type: 'message', role: 'user', content: [{ type: 'input_text', text: '(empty)' }] });
  }
  return {
    model: CODEX_CHATGPT_REMAP[modelInfo.modelId] || modelInfo.modelId,
    instructions,
    input: filteredInput,
    store: false,
    stream: true,
    service_tier: 'priority',
  };
}

// ---------------------------------------------------------------------------
// HTTPS request helpers
// ---------------------------------------------------------------------------

function makeAnthropicRequest(bodyStr, tokenEntry, onDelta) {
  // Enable streaming if we have an onDelta callback
  let actualBodyStr = bodyStr;
  if (onDelta) {
    try {
      const parsed = JSON.parse(bodyStr);
      parsed.stream = true;
      actualBodyStr = JSON.stringify(parsed);
    } catch {}
  }

  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.anthropic.com',
      path: '/v1/messages',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01',
        'anthropic-beta': ANTHROPIC_BETA_HEADERS,
        'Authorization': `Bearer ${tokenEntry.token}`,
        'user-agent': `claude-cli/${VERSION} (external, cli)`,
        'x-app': 'cli',
        'Content-Length': Buffer.byteLength(actualBodyStr),
      },
    };

    const req = httpsRequest(options, (res) => {
      if (res.statusCode !== 200) {
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => {
          reject(Object.assign(
            new Error(`Anthropic API error ${res.statusCode}: ${data.slice(0, 300)}`),
            { statusCode: res.statusCode, retryAfter: res.headers['retry-after'] }
          ));
        });
        return;
      }

      if (onDelta) {
        // Streaming mode — parse SSE incrementally
        let text = '';
        let usage = {};
        let buf = '';

        res.on('data', (chunk) => {
          buf += chunk;
          let nlIdx;
          while ((nlIdx = buf.indexOf('\n')) !== -1) {
            const line = buf.slice(0, nlIdx);
            buf = buf.slice(nlIdx + 1);
            if (!line.startsWith('data: ')) continue;
            const payload = line.slice(6).trim();
            if (payload === '[DONE]') continue;
            try {
              const event = JSON.parse(payload);
              if (event.type === 'content_block_delta' && event.delta?.type === 'text_delta') {
                text += event.delta.text;
                onDelta(event.delta.text);
              } else if (event.type === 'message_delta' && event.usage) {
                usage = { ...usage, output_tokens: event.usage.output_tokens };
              } else if (event.type === 'message_start' && event.message?.usage) {
                usage = { ...usage, input_tokens: event.message.usage.input_tokens };
              }
            } catch {}
          }
        });

        res.on('end', () => {
          if (text) {
            resolve({ text, usage });
          } else {
            reject(Object.assign(new Error('Empty response from Anthropic — model returned no text blocks'), { statusCode: 0 }));
          }
        });
      } else {
        // Non-streaming — buffer full response
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            const text = (parsed.content || [])
              .filter(b => b.type === 'text')
              .map(b => b.text)
              .join('\n');
            if (text) {
              resolve({ text, usage: parsed.usage || {} });
            } else {
              reject(Object.assign(new Error('Empty response from Anthropic ��� model returned no text blocks'), { statusCode: 0 }));
            }
          } catch (e) {
            reject(Object.assign(new Error(`Failed to parse Anthropic response: ${e.message}`), { statusCode: 0 }));
          }
        });
      }
    });
    req.on('error', e => reject(Object.assign(new Error(`Network error calling Anthropic: ${e.message}`), { statusCode: 0 })));
    const timer = setTimeout(() => { req.destroy(); reject(Object.assign(new Error('Request to Anthropic timed out after 5 minutes'), { statusCode: 0 })); }, TIMEOUT);
    req.on('close', () => clearTimeout(timer));
    req.write(actualBodyStr);
    req.end();
  });
}

function makeCodexRequest(bodyStr, tokenEntry, onDelta) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'chatgpt.com',
      path: '/backend-api/codex/responses',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'Authorization': `Bearer ${tokenEntry.token}`,
        'Content-Length': Buffer.byteLength(bodyStr),
      },
    };

    const req = httpsRequest(options, (res) => {
      if (res.statusCode !== 200) {
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => {
          if (res.statusCode === 403) {
            reject(Object.assign(
              new Error('Codex API returned 403 — your ChatGPT subscription may not include Codex access, or the token expired. Run "substation-auth chatgpt" to re-authenticate.'),
              { statusCode: 401, retryAfter: null }
            ));
          } else {
            reject(Object.assign(
              new Error(`Codex API error ${res.statusCode}: ${data.slice(0, 300)}`),
              { statusCode: res.statusCode, retryAfter: res.headers['retry-after'] }
            ));
          }
        });
        return;
      }

      // Stream SSE — parse incrementally
      let text = '';
      let usage = {};
      let buf = '';

      res.on('data', (chunk) => {
        buf += chunk;
        // Process complete lines
        let nlIdx;
        while ((nlIdx = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, nlIdx);
          buf = buf.slice(nlIdx + 1);
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') continue;
          try {
            const event = JSON.parse(payload);
            if (event.type === 'response.output_text.delta' && event.delta) {
              text += event.delta;
              if (onDelta) onDelta(event.delta);
            } else if (event.type === 'response.output_text.done') {
              if (event.text) text = event.text;
            } else if (event.type === 'response.completed' && event.response?.usage) {
              usage = event.response.usage;
            } else if (event.type === 'response.failed') {
              const errMsg = event.response?.error?.message || 'Unknown Codex error';
              const errCode = event.response?.error?.code || '';
              reject(Object.assign(new Error(`Codex API error: ${errMsg} (${errCode})`), { statusCode: 0 }));
              return;
            }
          } catch {}
        }
      });

      res.on('end', () => {
        if (text) {
          resolve({ text, usage });
        } else {
          reject(Object.assign(new Error('Empty response from Codex API — check your ChatGPT subscription status'), { statusCode: 0 }));
        }
      });
    });
    req.on('error', e => reject(Object.assign(new Error(`Network error calling Codex: ${e.message}`), { statusCode: 0 })));
    const timer = setTimeout(() => { req.destroy(); reject(Object.assign(new Error('Request to Codex timed out after 5 minutes'), { statusCode: 0 })); }, TIMEOUT);
    req.on('close', () => clearTimeout(timer));
    req.write(bodyStr);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// ChatGPT Image Generation — via Codex/Responses API (bypasses Cloudflare)
// ---------------------------------------------------------------------------

function chatgptImageGenerate(tokenEntry, prompt, size = '1024x1024', quality = 'auto') {
  return new Promise((resolve, reject) => {
    // Use GPT 5.4 through the Codex/Responses endpoint — it supports native image generation
    // gpt-image-1 is rejected by this endpoint, but gpt-5.4 handles image prompts natively
    // Send as a plain text input — GPT 5.4 auto-generates images when the prompt describes one
    const requestBody = JSON.stringify({
      model: 'gpt-5.4',
      instructions: 'Generate the requested image. Output only the image.',
      input: [{ role: 'user', content: prompt }],
      tools: [{ type: 'image_generation', model: 'gpt-image-2', size, quality }],
      store: false,
      stream: true,
    });

    const options = {
      hostname: 'chatgpt.com',
      path: '/backend-api/codex/responses',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'Authorization': `Bearer ${tokenEntry.token}`,
        'Content-Length': Buffer.byteLength(requestBody),
      },
    };

    const req = httpsRequest(options, (res) => {
      if (res.statusCode !== 200) {
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => {
          reject(Object.assign(
            new Error(`Codex image API error ${res.statusCode}: ${data.slice(0, 500)}`),
            { statusCode: res.statusCode }
          ));
        });
        return;
      }

      let buf = '';
      let imageB64 = null;

      res.on('data', (chunk) => {
        buf += chunk;
        let nlIdx;
        while ((nlIdx = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, nlIdx);
          buf = buf.slice(nlIdx + 1);
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') continue;
          try {
            const event = JSON.parse(payload);

            // Responses API image output events — multiple formats
            if (event.type === 'response.image.delta' && event.delta) {
              imageB64 = (imageB64 || '') + event.delta;
            } else if (event.type === 'response.image.done' && event.image) {
              imageB64 = event.image;
            }
            // image_generation_call partial_image — use last partial as final
            if (event.type === 'response.image_generation_call.partial_image' && event.partial_image_b64) {
              imageB64 = event.partial_image_b64;
            }
            // output_item.done may carry the final image result
            if (event.type === 'response.output_item.done' && event.item) {
              if (event.item.type === 'image_generation_call' && event.item.result) {
                imageB64 = event.item.result;
              }
              if (event.item.image_b64) {
                imageB64 = event.item.image_b64;
              }
            }
            // Also check for output_image in response.completed
            if (event.type === 'response.completed' && event.response?.output) {
              for (const item of event.response.output) {
                if (item.type === 'image_generation_call' && item.result) {
                  imageB64 = item.result;
                }
                if (item.type === 'image' && item.image?.b64_json) {
                  imageB64 = item.image.b64_json;
                }
                if (Array.isArray(item.content)) {
                  for (const c of item.content) {
                    if (c.type === 'image' && c.image?.b64_json) {
                      imageB64 = c.image.b64_json;
                    }
                  }
                }
              }
            }
            // Catch errors from the API
            if (event.type === 'response.failed') {
              const errMsg = event.response?.error?.message || 'Image generation failed';
              reject(Object.assign(new Error(errMsg), { statusCode: 0 }));
              return;
            }
          } catch {}
        }
      });

      res.on('end', () => {
        if (imageB64) {
          resolve({ b64: imageB64 });
        } else {
          reject(new Error('No image data in Codex response. The model may not support image generation through this endpoint.'));
        }
      });
    });

    req.on('error', e => reject(Object.assign(
      new Error(`Network error calling Codex image API: ${e.message}`),
      { statusCode: 0 }
    )));

    const timer = setTimeout(() => {
      req.destroy();
      reject(Object.assign(new Error('Image generation timed out after 5 minutes'), { statusCode: 0 }));
    }, TIMEOUT);
    req.on('close', () => clearTimeout(timer));

    req.write(requestBody);
    req.end();
  });
}

function chatgptDeleteConversation(tokenEntry, conversationId) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ is_visible: false });
    const options = {
      hostname: 'chatgpt.com',
      path: `/backend-api/conversation/${conversationId}`,
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${tokenEntry.token}`,
        'Content-Length': Buffer.byteLength(body),
      },
    };
    const req = httpsRequest(options, (res) => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => resolve(d));
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

const ALLOWED_IMAGE_HOSTS = ['.openai.com', '.chatgpt.com', 'oaiusercontent.com'];

function isAllowedImageHost(hostname) {
  return ALLOWED_IMAGE_HOSTS.some(allowed => hostname === allowed || hostname.endsWith(allowed));
}

function fetchImageAsBase64(url, token) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);

    // SSRF protection: only allow known image CDN hosts
    if (!isAllowedImageHost(parsedUrl.hostname)) {
      log(`WARN: Blocked image fetch to disallowed host: ${parsedUrl.hostname}`);
      resolve(null);
      return;
    }

    const options = {
      hostname: parsedUrl.hostname,
      path: parsedUrl.pathname + parsedUrl.search,
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    };

    const req = httpsRequest(options, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        // Follow redirect only if target host is also allowlisted
        const redirectUrl = res.headers.location;
        try {
          const redirectParsed = new URL(redirectUrl, url);
          if (!isAllowedImageHost(redirectParsed.hostname)) {
            log(`WARN: Blocked redirect to disallowed host: ${redirectParsed.hostname}`);
            resolve(null);
            return;
          }
        } catch {
          resolve(null);
          return;
        }
        fetchImageAsBase64(redirectUrl, token).then(resolve).catch(reject);
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error(`Image download failed: HTTP ${res.statusCode}`));
        return;
      }

      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const buffer = Buffer.concat(chunks);
        const contentType = res.headers['content-type'] || 'image/png';
        const b64 = `data:${contentType};base64,${buffer.toString('base64')}`;
        resolve(b64);
      });
    });

    req.on('error', e => reject(e));
    setTimeout(() => { req.destroy(); reject(new Error('Image download timed out')); }, 30000);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Anthropic Direct API — Bearer auth with OAuth tokens (for secondary agents)
// ---------------------------------------------------------------------------

function makeAnthropicDirectRequest(bodyStr, tokenEntry, onDelta) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.anthropic.com',
      path: '/v1/messages',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${tokenEntry.token}`,
        'anthropic-version': '2023-06-01',
        'anthropic-beta': ANTHROPIC_BETA_HEADERS,
        // Claude Code identity headers — required for OAuth tokens
        'anthropic-client-type': 'claude-code',
        'anthropic-client-version': '2.1.114',
        'User-Agent': 'claude-code/2.1.114',
        'Content-Length': Buffer.byteLength(bodyStr),
      },
    };

    const req = httpsRequest(options, (res) => {
      if (res.statusCode !== 200) {
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => {
          reject(Object.assign(
            new Error(`Anthropic API error ${res.statusCode}: ${data.slice(0, 300)}`),
            { statusCode: res.statusCode, retryAfter: res.headers['retry-after'] }
          ));
        });
        return;
      }

      // Non-streaming: collect full response
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try {
          const resp = JSON.parse(data);
          let text = '';
          for (const block of (resp.content || [])) {
            if (block.type === 'text') text += block.text;
          }
          if (!text) {
            reject(new Error('Empty response from Anthropic direct API'));
            return;
          }
          // Stream the full text as one delta if callback provided
          if (onDelta) onDelta(text);
          resolve({ text, usage: resp.usage || {} });
        } catch (e) {
          reject(new Error(`Failed to parse Anthropic response: ${e.message}`));
        }
      });
    });

    req.on('error', e => reject(Object.assign(new Error(`Network error calling Anthropic: ${e.message}`), { statusCode: 0 })));
    const timer = setTimeout(() => { req.destroy(); reject(Object.assign(new Error('Anthropic direct request timed out after 5 minutes'), { statusCode: 0 })); }, TIMEOUT);
    req.on('close', () => clearTimeout(timer));
    req.write(bodyStr);
    req.end();
  });
}

async function invokeClaudeDirect(openaiMessages, modelInfo, tokenAffinity, onDelta) {
  // Find the specific token by affinity (e.g., "backup" matches tokens with "backup" in id, or non-primary anthropic tokens)
  const anthropicTokens = pool.filter(t => t.provider === 'anthropic' && !t.dead);
  let targetToken = null;

  if (tokenAffinity === 'backup') {
    // "backup" = any anthropic token that is NOT the primary (not from auth-profiles of the main agent)
    // Prefer tokens with "backup"/"secondary" in the id
    targetToken = anthropicTokens.find(t => t.id.includes('backup') || t.id.includes('secondary'));
    // Fallback: use the last anthropic token in the pool (most recently added, likely backup)
    if (!targetToken) targetToken = anthropicTokens.find(t => !t.id.startsWith('ap:anthropic:manual') && !t.id.startsWith('ap:anthropic:oauth'));
    // Last resort: any anthropic token
    if (!targetToken) targetToken = anthropicTokens[anthropicTokens.length - 1];
  } else if (tokenAffinity) {
    // Exact ID match or partial match
    targetToken = anthropicTokens.find(t => t.id === tokenAffinity || t.id.includes(tokenAffinity));
  }

  if (!targetToken) targetToken = selectToken('anthropic');
  if (!targetToken) throw new Error('No Anthropic tokens available for direct API call');

  // Build request body (non-streaming for simplicity — secondary agents don't need real-time streaming)
  const body = buildAnthropicBody(openaiMessages, modelInfo);
  // Force non-streaming for direct path
  body.stream = false;
  const bodyStr = JSON.stringify(body);

  log(`Direct API request (model=${modelInfo.modelId}, token=${targetToken.id}, body=${bodyStr.length}b)`);
  markRequestStart(targetToken);

  try {
    const result = await makeAnthropicDirectRequest(bodyStr, targetToken, onDelta);
    markSuccess(targetToken);
    return result;
  } catch (err) {
    if (err.statusCode === 429) {
      markRateLimited(targetToken, parseInt(err.retryAfter || '60', 10));
    } else if (err.statusCode === 401) {
      markDead(targetToken);
    } else {
      markRequestEnd(targetToken);
    }
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Anthropic via Agent SDK — persistent session, real CC binary, undetectable
// ---------------------------------------------------------------------------

// Warm session pool — keyed by "model:tokenId", LRU rotation
// Each Anthropic OAuth token gets its own persistent session per model
let anthropicRRIndex = 0; // Round-robin counter for Anthropic token selection

async function querySDK(model, tokenId, oauthToken, prompt) {
  const sdk = await loadAgentSDK();
  log(`SDK query (model=${model}, token=${tokenId})...`);
  const queryEnv = { ...process.env };
  if (oauthToken) {
    queryEnv.CLAUDE_CODE_OAUTH_TOKEN = oauthToken;
  }
  return sdk.query({
    prompt,
    options: {
      model,
      permissionMode: 'bypassPermissions',
      maxTurns: Infinity,
      env: queryEnv,
    },
  });
}

function selectAnthropicToken() {
  const tokens = getAnthropicOAuthTokens(); // already filters !dead
  if (tokens.length === 0) return null;
  if (tokens.length === 1) return tokens[0];

  const now = Date.now();

  const available = tokens.filter(t => {
    if (t.cooldownUntil && t.cooldownUntil > now) return false;
    return true;
  });

  const candidates = available.length > 0 ? available : tokens; // fallback if all cooling down

  // Sort: proven tokens (lastSuccess > 0) first, then fewest errors
  const sorted = [...candidates].sort((a, b) => {
    const aProven = a.lastSuccess > 0 ? 0 : 1;
    const bProven = b.lastSuccess > 0 ? 0 : 1;
    if (aProven !== bProven) return aProven - bProven;
    return a.errorCount - b.errorCount;
  });

  const token = sorted[anthropicRRIndex % sorted.length];
  anthropicRRIndex = (anthropicRRIndex + 1) % sorted.length;
  log(`Selected token ${token.id} (proven=${token.lastSuccess > 0}, errors=${token.errorCount})`);
  return token;
}

// Format OpenAI-style messages into a single prompt string for the Agent SDK session.
// Preserves system prompts (operator identity/personality), full conversation history,
// and tool results so Claude has complete context — not just the last user message.
function rewriteGatewayVisionRefs(text) {
  // Gateway text-mode vision enrichment injects patterns like:
  //   [The user sent an image~ Here's what I can see:\n{description}]\n
  //   [If you need a closer look, use vision_analyze with image_url: /path/to/file ~]
  // The vision description is often wrong (analyzed without conversation context).
  // Replace the whole block with a direct Read reference to the cached image file.
  const patterns = [
    // Gateway v2 format: "sent an image~" ... "vision_analyze with image_url: PATH"
    /\[The user sent an image~[^\]]*\]\s*\[If you need a closer look, use vision_analyze with image_url:\s*([^\s~\]]+)\s*~?\]/gs,
    // Gateway fallback format: "sent an image but" ... "vision_analyze using image_url: PATH"
    /\[The user sent an image but[^\]]*vision_analyze (?:using|with) image_url:\s*([^\s\]~]+)\s*~?\]/gs,
    // Older format: "attached an image. Here's what it contains:" ... inlined description
    /\[The user attached an image\. Here's what it contains:\n[^\]]*\]\s*(?:\[If you need a closer look[^\]]*image_url:\s*([^\s\]~]+)[^\]]*\])?/gs,
  ];
  let result = text;
  for (const pattern of patterns) {
    result = result.replace(pattern, (match, path) => {
      if (path && existsSync(path)) {
        log(`Rewrote gateway vision ref → direct Read reference: ${path}`);
        return `[The user attached an image. It has been saved to ${path} — use the Read tool to view it.]`;
      }
      // If the path doesn't exist or wasn't captured, try extracting from the full match
      const fallbackMatch = match.match(/image_url:\s*([^\s\]~]+)/);
      if (fallbackMatch && fallbackMatch[1] && existsSync(fallbackMatch[1])) {
        log(`Rewrote gateway vision ref (fallback) → direct Read reference: ${fallbackMatch[1]}`);
        return `[The user attached an image. It has been saved to ${fallbackMatch[1]} — use the Read tool to view it.]`;
      }
      return match; // can't find file, keep original description
    });
  }
  return result;
}

function formatMessagesForSDK(openaiMessages) {
  const parts = [];
  const systemParts = [];
  const conversationParts = [];

  for (const msg of openaiMessages) {
    let text = Array.isArray(msg.content)
      ? msg.content.filter(b => b.type === 'text').map(b => b.text).join('\n')
      : (msg.content || '');

    // Rewrite gateway vision-enrichment blocks to direct file references
    // so the agent can Read the actual image instead of trusting a lossy description
    text = rewriteGatewayVisionRefs(text);

    // Extract and save images to temp files so the agent can view them via Read tool
    const imageBlocks = extractImageBlocks(msg.content);
    let imageRefs = '';
    for (const img of imageBlocks) {
      if (img.data) {
        const filepath = saveImageToTmp(img);
        imageRefs += `\n[The user attached an image. It has been saved to ${filepath} — use the Read tool to view it.]`;
        log(`Saved user image to ${filepath} (${img.mediaType}, ${Math.round(img.data.length / 1024)}KB base64)`);
      } else if (img.url) {
        imageRefs += `\n[The user attached an image: ${img.url}]`;
      }
    }

    const fullText = (text + imageRefs).trim();
    if (!fullText) continue;

    if (msg.role === 'system') {
      systemParts.push(text);
    } else {
      conversationParts.push({ role: msg.role, text: fullText });
    }
  }

  // Always inject operator identity, then any caller-provided system context
  parts.push(OPERATOR_SYSTEM_PROMPT);
  if (systemParts.length > 0) {
    parts.push(systemParts.join('\n\n'));
  }

  // If there's conversation history (more than just the last user message),
  // include it so Claude has context of the ongoing conversation
  if (conversationParts.length > 1) {
    const history = conversationParts.slice(0, -1);
    const historyStr = history.map(m => `[${m.role}]: ${m.text}`).join('\n\n');
    parts.push('<conversation_history>\n' + historyStr + '\n</conversation_history>');
  }

  // The actual user message/request
  const last = conversationParts[conversationParts.length - 1];
  if (last) {
    parts.push(last.text);
  }

  return parts.join('\n\n');
}

async function invokeClaudeSDK(openaiMessages, modelInfo, onDelta, tokenAffinity) {
  const model = modelInfo.modelId;
  const tokens = getAnthropicOAuthTokens();
  if (tokens.length === 0) {
    throw new Error('No Anthropic OAuth tokens found. Log into Claude Code or add tokens to auth-profiles.json.');
  }

  // If token affinity specified, filter to that token
  const candidateTokens = tokenAffinity
    ? tokens.filter(t => t.id === tokenAffinity || t.id.includes(tokenAffinity))
    : tokens;
  if (candidateTokens.length === 0) {
    throw new Error(`No token matching affinity "${tokenAffinity}" found.`);
  }

  // Fast-fail: all tokens already in cooldown → return 429 immediately, don't hang
  const now = Date.now();
  const available = candidateTokens.filter(t => !t.cooldownUntil || t.cooldownUntil < now);
  if (available.length === 0) {
    const earliest = candidateTokens.reduce((min, t) => t.cooldownUntil && (!min || t.cooldownUntil < min) ? t.cooldownUntil : min, null);
    const secsLeft = earliest ? Math.ceil((earliest - now) / 1000) : 60;
    const minsLeft = Math.ceil(secsLeft / 60);
    throw Object.assign(new Error(`All Anthropic tokens in cooldown — rate limit resets in ~${minsLeft}min`), { statusCode: 429, retryAfterSec: secsLeft });
  }

  // Format full conversation context including system prompts and history
  const prompt = formatMessagesForSDK(openaiMessages);

  // Try tokens with rotation
  const maxAttempts = available.length;
  let lastError = null;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const token = tokenAffinity ? available[attempt] : (attempt === 0 ? selectAnthropicToken() : tokens[attempt % tokens.length]);
    if (!token) break;

    const key = `${model}:${token.id}`;
    log(`Agent SDK request (model=${model}, token=${token.id}, attempt=${attempt + 1}/${maxAttempts})`);
    markRequestStart(token);

    try {
      const iter = await querySDK(model, token.id, token.token, prompt);

      let text = '';
      let usage = {};
      let hadToolUse = false;
      let idleReject;
      const idlePromise = new Promise((_, reject) => { idleReject = reject; });
      const IDLE_TIMEOUT = 120000;
      const TOOL_IDLE_TIMEOUT = 900000;
      let idleTimer;
      let toolRunning = false;
      const resetIdleTimer = () => {
        if (idleTimer) clearTimeout(idleTimer);
        const timeout = toolRunning ? TOOL_IDLE_TIMEOUT : IDLE_TIMEOUT;
        idleTimer = setTimeout(() => idleReject(new Error(`Agent SDK timeout — no stream activity for ${timeout/1000}s`)), timeout);
      };
      resetIdleTimer();

      const iteratorLoop = (async () => {
        for await (const msg of iter) {
          resetIdleTimer();
          if (msg.type === 'user') {
            toolRunning = false;
            resetIdleTimer();
          }
          if (msg.type === 'assistant') {
            for (const block of (msg.message?.content || [])) {
              if (block.type === 'tool_use') {
                hadToolUse = true;
                toolRunning = true;
                resetIdleTimer();
              }
              if (block.type === 'text' && block.text) {
                text += block.text;
                if (onDelta) onDelta(block.text);
              }
            }
          } else if (msg.type === 'rate_limit_event') {
            const info = msg.rate_limit_info || {};
            if (info.status && !info.status.startsWith('allowed')) {
              const cooldownMs = info.resetsAt ? Math.max(0, info.resetsAt * 1000 - Date.now()) : DEAD_REVIVE_MS;
              const resetStr = info.resetsAt ? new Date(info.resetsAt * 1000).toISOString() : 'unknown';
              throw Object.assign(new Error(`usage limit hit — resets at ${resetStr} (${info.rateLimitType || 'claude_max'}) — cooldown ${Math.ceil(cooldownMs/60000)}min`), { statusCode: 429 });
            }
          } else if (msg.type === 'result') {
            if (msg.usage) usage = msg.usage;
            if (msg.is_error && msg.result) throw new Error(msg.result);
            if (msg.result && !text) text = msg.result;
          }
        }
      })();

      await Promise.race([iteratorLoop, idlePromise]);
      if (idleTimer) clearTimeout(idleTimer);

      if (!text) throw new Error('Empty response from Claude via Agent SDK');

      if (trackNarration(model, hadToolUse, text)) {
        throw new Error('BLOCKED: Agent produced consecutive narration-only responses with no tool calls. Re-send with a specific instruction.');
      }

      markSuccess(token);
      return { text, usage };
    } catch (err) {
      lastError = err;
      log(`Agent SDK error (token=${token.id}): ${err.message}`);

      // Narration loop block is TERMINAL — do not retry with another token
      if (err.message && err.message.startsWith('BLOCKED:')) {
        throw err;
      }

      // If rate/usage limited, cooldown for the actual window duration (not just 30 min)
      const errLower = (err.message || '').toLowerCase();
      if (errLower.includes('rate') || errLower.includes('limit') || errLower.includes('usage') || errLower.includes('hit your') || errLower.includes('resets') || errLower.includes('timeout')) {
        // Parse actual reset time from rate_limit_event error if available
        let cooldownSec = DEAD_REVIVE_MS / 1000; // fallback: 30 min
        const resetsMatch = err.message.match(/resets at (\d{4}-\d{2}-\d{2}T[\d:.]+Z?)/);
        if (resetsMatch) {
          const resetsAt = new Date(resetsMatch[1]).getTime();
          if (!isNaN(resetsAt) && resetsAt > Date.now()) {
            cooldownSec = Math.ceil((resetsAt - Date.now()) / 1000);
          }
        } else if (errLower.includes('timeout')) {
          // SDK timeout ≠ account cap. Likely transient — retry after short cooldown.
          cooldownSec = 3 * 60; // 3 minutes, not 5 hours
        }
        const cooldownMs = cooldownSec * 1000;
        log(`Token ${token.id} hit usage/rate limit — cooldown ${Math.ceil(cooldownMs / 60000)}min (until ${new Date(Date.now() + cooldownMs).toISOString()}), failing over...`);
        markRateLimited(token, cooldownSec);
        continue;
      }

      // Auth error — bench this token for 24h and try next
      if (isAuthError(err)) {
        markDead(token, 'auth'); // marks dead + decrements activeRequests
        continue;
      }

      // Other errors — decrement activeRequests and try next token
      markRequestEnd(token);
      continue;
    }
  }

  throw lastError || new Error('All Anthropic tokens exhausted. Check your Claude Max subscription status.');
}

// ---------------------------------------------------------------------------
// Codex API caller with pool rotation
// ---------------------------------------------------------------------------

async function invokeCodex(openaiMessages, modelInfo, onDelta) {
  const provider = 'openai';
  const providerTokens = pool.filter(t => t.provider === provider);
  if (providerTokens.length === 0) {
    throw new Error('No ChatGPT tokens configured. Run "substation-auth chatgpt" to set up authentication.');
  }

  const bodyStr = JSON.stringify(buildCodexBody(openaiMessages, modelInfo));
  log(`Codex request (model=${modelInfo.modelId}, body=${bodyStr.length}b)`);

  const maxAttempts = providerTokens.length;
  let lastError = null;
  let streamStarted = false;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const tokenEntry = selectToken(provider);
    if (!tokenEntry) break;

    markRequestStart(tokenEntry);
    log(`Attempt ${attempt + 1}/${maxAttempts} using token ${tokenEntry.id}`);

    const wrappedDelta = onDelta ? (delta) => { streamStarted = true; onDelta(delta); } : null;

    try {
      const result = await makeCodexRequest(bodyStr, tokenEntry, wrappedDelta);
      markSuccess(tokenEntry);
      return result;
    } catch (err) {
      lastError = err;
      if (streamStarted) { markRequestEnd(tokenEntry); throw err; }
      if (err.statusCode === 429) { markRateLimited(tokenEntry, parseInt(err.retryAfter || '60', 10)); continue; }
      if (err.statusCode === 401) {
        if (tokenEntry.refreshToken) {
          try {
            log(`Token ${tokenEntry.id} got 401, attempting refresh...`);
            await refreshOpenAIToken(tokenEntry);
            const result = await makeCodexRequest(bodyStr, tokenEntry, wrappedDelta);
            markSuccess(tokenEntry);
            return result;
          } catch { markDead(tokenEntry); continue; }
        }
        markDead(tokenEntry);
        continue;
      }
      markRequestEnd(tokenEntry);
      throw err;
    }
  }

  throw lastError || new Error('All ChatGPT tokens exhausted. Run "substation-auth chatgpt" to re-authenticate.');
}

// ---------------------------------------------------------------------------
// Cross-provider failover — when one provider's cap hits, route to the other
// ---------------------------------------------------------------------------

const FAILOVER_MAP = {
  // Anthropic → OpenAI equivalents
  'claude-fable-5':            'gpt-5.4',
  'claude-opus-4-8':           'gpt-5.4',
  'claude-opus-4-7':           'gpt-5.4',
  'claude-opus-4-6':           'gpt-5.4',
  'claude-sonnet-4-6':         'gpt-5.4',
  'claude-haiku-4-5-20251001': 'gpt-5.4-mini',
  // OpenAI → Anthropic equivalents
  'gpt-5.4':            'claude-opus-4-6',
  'gpt-5.1-codex-max':  'claude-opus-4-6',
  'gpt-5.1-codex':      'claude-sonnet-4-6',
  'gpt-5.4-mini':       'claude-haiku-4-5-20251001',
  'gpt-5.1-codex-mini': 'claude-haiku-4-5-20251001',
};

function isCapError(err) {
  const msg = (err.message || '').toLowerCase();
  return msg.includes('hit your limit') ||
    msg.includes('usage limit') ||
    msg.includes('rate') ||
    msg.includes('exhausted') ||
    msg.includes('all anthropic tokens') ||
    msg.includes('all chatgpt tokens') ||
    err.statusCode === 429;
}

// ---------------------------------------------------------------------------
// Unified dispatcher — Anthropic token rotation → Direct API → Codex failover
// ---------------------------------------------------------------------------

async function invokeModel(openaiMessages, model, onDelta, opts = {}) {
  // Parse per-account suffix: "opus-4-6:acct2-alice" → model=opus-4-6, tokenAffinity=acct2-alice
  let baseModel = model;
  if (!opts.tokenAffinity && model.includes(':')) {
    const colonIdx = model.indexOf(':');
    baseModel = model.slice(0, colonIdx);
    opts = { ...opts, tokenAffinity: model.slice(colonIdx + 1) };
  }
  const modelInfo = resolveModel(baseModel);

  if (modelInfo.provider === 'anthropic') {
    // --- Agent SDK (Claude OAuth) — only path for Anthropic models ---
    try {
      return await invokeClaudeSDK(openaiMessages, modelInfo, onDelta, opts.tokenAffinity || null);
    } catch (err) {
      const isSdkBroken = err.message && (err.message.includes('is not a function') || err.message.includes('not installed'));
      if (!isCapError(err) && !isSdkBroken) throw err;
      log(isSdkBroken ? `SDK unavailable: ${err.message}` : `SDK path capped: ${err.message}`);

      // Cross-provider failover to OpenAI
      const failoverModelId = FAILOVER_MAP[modelInfo.modelId];
      if (failoverModelId) {
        const failoverInfo = resolveModel(failoverModelId);
        log(`>>> FAILOVER: Anthropic capped → ${failoverModelId} (OpenAI)`);
        try {
          return await invokeCodex(openaiMessages, failoverInfo, onDelta);
        } catch (codexErr) {
          log(`OpenAI failover also failed: ${codexErr.message}`);
        }
      }

      throw err;
    }
  }

  // --- OpenAI primary path with reverse failover to Anthropic ---
  try {
    return await invokeCodex(openaiMessages, modelInfo, onDelta);
  } catch (err) {
    if (!isCapError(err)) throw err;

    const failoverModelId = FAILOVER_MAP[modelInfo.modelId];
    if (!failoverModelId) throw err;

    const failoverInfo = resolveModel(failoverModelId);
    log(`>>> FAILOVER: Codex capped → ${failoverModelId} (Anthropic)`);
    try {
      return await invokeClaudeSDK(openaiMessages, failoverInfo, onDelta);
    } catch (anthropicErr) {
      log(`Anthropic failover also failed: ${anthropicErr.message}`);
      throw err;
    }
  }
}

// ---------------------------------------------------------------------------
// Port conflict resolution
// ---------------------------------------------------------------------------

function killStaleProcess(port) {
  try {
    const pid = execSync(`/usr/sbin/lsof -ti :${port}`, { timeout: 5000 }).toString().trim();
    if (pid && pid !== String(process.pid)) {
      log(`Killing stale process ${pid} on port ${port}`);
      execSync(`kill ${pid}`, { timeout: 5000 });
      for (let i = 0; i < 10; i++) {
        try {
          execSync(`/usr/sbin/lsof -ti :${port}`, { timeout: 2000 });
          execSync('sleep 0.3');
        } catch {
          return true;
        }
      }
    }
  } catch (err) {
    const msg = err.message || '';
    if (msg.includes('ENOENT') || msg.includes('not found') || msg.includes('No such file')) {
      log('lsof not available — cannot resolve port conflict');
      return false;
    }
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Proxy server
// ---------------------------------------------------------------------------

let proxyServer = null;

const rateLimitMap = new Map();
const RATE_LIMIT_MAX = 600;
const RATE_LIMIT_WINDOW_MS = 60000;
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of rateLimitMap) {
    if (now - entry.windowStart > RATE_LIMIT_WINDOW_MS) rateLimitMap.delete(ip);
  }
}, 60000).unref();

function safeEnd(res, statusCode, body) {
  if (res.headersSent) return;
  res.writeHead(statusCode, { 'Content-Type': 'application/json' });
  res.end(typeof body === 'string' ? body : JSON.stringify(body));
}

function startProxy() {
  if (proxyServer) return;

  proxyServer = createServer(async (req, res) => {
    // CORS: only allow localhost/127.0.0.1 origins
    const origin = req.headers['origin'];
    if (origin && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
      res.setHeader('Vary', 'Origin');
    }

    if (req.method === 'OPTIONS') { res.writeHead(200); res.end(); return; }

    const clientIp = req.socket.remoteAddress || 'unknown';
    const now = Date.now();
    let bucket = rateLimitMap.get(clientIp);
    if (!bucket || now - bucket.windowStart > RATE_LIMIT_WINDOW_MS) {
      bucket = { windowStart: now, count: 0 };
      rateLimitMap.set(clientIp, bucket);
    }
    bucket.count++;
    if (bucket.count > RATE_LIMIT_MAX) {
      safeEnd(res, 429, { error: 'rate_limited', message: 'Too many requests. Limit: 120/minute.' });
      return;
    }

    // --- Auth enforcement for POST requests ---
    if (req.method === 'POST') {
      const authHeader = req.headers['authorization'] || '';
      const bearerKey = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
      // Also accept Anthropic's x-api-key header (used by Claude CLI --bare mode)
      const xApiKey = req.headers['x-api-key'] || '';
      const providedKey = bearerKey || xApiKey;
      if (providedKey !== AUTH_KEY) {
        safeEnd(res, 401, { error: { message: 'Unauthorized. Provide Authorization: Bearer <key> or x-api-key: <key> header.' } });
        return;
      }
    }

    // --- GET routes ---
    if (req.method === 'GET') {
      if (req.url === '/health') {
        const status = getPoolStatus();
        safeEnd(res, 200, {
          status: status.available > 0 ? 'ok' : (status.total > 0 ? 'degraded' : 'no_tokens'),
          version: VERSION,
          backend: 'direct-api',
          claudeCodeVersion: VERSION,
          pool: {
            total: status.total,
            available: status.available,
            cooldown: status.cooldown,
            dead: status.dead,
            byProvider: status.byProvider,
          },
          setup: status.total === 0 ? {
            anthropic: 'Add OAuth token to auth-profiles.json or set SUBSTATION_OAUTH_TOKEN',
            openai: 'Run "substation-auth chatgpt"',
          } : undefined,
        });
        return;
      }

      if (req.url === '/pool') {
        safeEnd(res, 200, getPoolStatus());
        return;
      }

      if (req.url === '/v1/models' || req.url === '/models') {
        const models = [
          { id: 'fable-5', object: 'model', owned_by: 'indigo-collective' },
          { id: 'opus-4-8', object: 'model', owned_by: 'indigo-collective' },
          { id: 'opus-4-7', object: 'model', owned_by: 'indigo-collective' },
          { id: 'opus-4-6', object: 'model', owned_by: 'indigo-collective' },
          { id: 'sonnet-4-6', object: 'model', owned_by: 'indigo-collective' },
          { id: 'haiku-4-5', object: 'model', owned_by: 'indigo-collective' },
        ];
        // Per-account variants for manual token selection
        const anthropicPoolTokens = pool.filter(t => t.provider === 'anthropic' && !t.dead);
        for (const t of anthropicPoolTokens) {
          models.push(
            { id: `fable-5:${t.id}`, object: 'model', owned_by: 'indigo-collective' },
            { id: `sonnet-4-6:${t.id}`, object: 'model', owned_by: 'indigo-collective' },
            { id: `opus-4-6:${t.id}`, object: 'model', owned_by: 'indigo-collective' },
            { id: `opus-4-7:${t.id}`, object: 'model', owned_by: 'indigo-collective' },
            { id: `opus-4-8:${t.id}`, object: 'model', owned_by: 'indigo-collective' },
          );
        }
        if (pool.some(t => t.provider === 'openai')) {
          models.push(
            { id: 'gpt-5.4', object: 'model', owned_by: 'indigo-collective' },
            { id: 'gpt-5.4-mini', object: 'model', owned_by: 'indigo-collective' },
            { id: 'gpt-5.1-codex', object: 'model', owned_by: 'indigo-collective' },
            { id: 'gpt-5.1-codex-mini', object: 'model', owned_by: 'indigo-collective' },
            { id: 'gpt-5.1-codex-max', object: 'model', owned_by: 'indigo-collective' },
          );
          const openaiPoolTokens = pool.filter(t => t.provider === 'openai' && !t.dead);
          for (const t of openaiPoolTokens) {
            models.push(
              { id: `gpt-5.4:${t.id}`, object: 'model', owned_by: 'indigo-collective' },
              { id: `gpt-5.4-mini:${t.id}`, object: 'model', owned_by: 'indigo-collective' },
            );
          }
        }
        safeEnd(res, 200, { object: 'list', data: models });
        return;
      }

      res.writeHead(404); res.end('Not found'); return;
    }

    // --- POST /v1/chat/completions ---
    if (req.method === 'POST' && (req.url === '/v1/chat/completions' || req.url === '/chat/completions')) {
      let body = '';
      let bodySize = 0;

      req.on('data', c => {
        bodySize += c.length;
        if (bodySize > MAX_BODY_SIZE) {
          safeEnd(res, 413, { error: { message: `Request body too large (max ${MAX_BODY_SIZE / 1024 / 1024}MB)` } });
          req.destroy();
          return;
        }
        body += c;
      });

      req.on('end', async () => {
        if (res.headersSent) return; // body was too large

        let parsed;
        try { parsed = JSON.parse(body); } catch {
          safeEnd(res, 400, { error: { message: 'Invalid JSON in request body' } });
          return;
        }

        const { messages = [], model = 'sonnet-4-6', stream = false } = parsed;
        if (!messages.length) {
          safeEnd(res, 400, { error: { message: 'No messages in request — include at least one message' } });
          return;
        }

        // Token affinity: X-SubStation-Token header routes to a specific pool token
        // "backup" = use backup OAuth token via direct API (for secondary agents)
        const tokenAffinity = req.headers['x-substation-token'] || null;
        const invokeOpts = tokenAffinity ? { tokenAffinity } : {};

        try {
          const id = `chatcmpl-ss-${randomUUID().slice(0, 12)}`;
          const created = Math.floor(Date.now() / 1000);

          if (stream) {
            // Real-time streaming — send SSE headers immediately, pipe deltas as they arrive
            let headersSent = false;
            const onDelta = (delta) => {
              if (res.destroyed) return;
              if (!headersSent) {
                headersSent = true;
                res.writeHead(200, {
                  'Content-Type': 'text/event-stream',
                  'Cache-Control': 'no-cache',
                  'Connection': 'keep-alive',
                });
                // Send role chunk first
                res.write(`data: ${JSON.stringify({
                  id, object: 'chat.completion.chunk', created,
                  model, system_fingerprint: null,
                  choices: [{ index: 0, delta: { role: 'assistant' }, logprobs: null, finish_reason: null }]
                })}\n\n`);
              }
              // Stream each delta as its own SSE chunk
              res.write(`data: ${JSON.stringify({
                id, object: 'chat.completion.chunk', created,
                model, system_fingerprint: null,
                choices: [{ index: 0, delta: { content: delta }, logprobs: null, finish_reason: null }]
              })}\n\n`);
            };

            const { text, usage } = await invokeModel(messages, model, onDelta, invokeOpts);

            // Finish the stream
            if (!headersSent) {
              // Edge case: no deltas received, send full response as one chunk
              res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
              res.write(`data: ${JSON.stringify({
                id, object: 'chat.completion.chunk', created,
                model, system_fingerprint: null,
                choices: [{ index: 0, delta: { role: 'assistant', content: text }, logprobs: null, finish_reason: null }]
              })}\n\n`);
            }
            res.write(`data: ${JSON.stringify({
              id, object: 'chat.completion.chunk', created,
              model, system_fingerprint: null,
              choices: [{ index: 0, delta: {}, logprobs: null, finish_reason: 'stop' }]
            })}\n\n`);
            res.write('data: [DONE]\n\n');
            res.end();
          } else {
            // Non-streaming — buffer full response
            const { text, usage } = await invokeModel(messages, model, null, invokeOpts);
            const promptTokens = usage?.input_tokens || usage?.prompt_tokens || Math.max(1, JSON.stringify(messages).split(/\s+/).length);
            const completionTokens = usage?.output_tokens || usage?.completion_tokens || Math.max(1, text.split(/\s+/).length);
            safeEnd(res, 200, {
              id,
              object: 'chat.completion',
              created,
              model,
              system_fingerprint: null,
              choices: [{
                index: 0,
                message: { role: 'assistant', content: text },
                logprobs: null,
                finish_reason: 'stop',
              }],
              usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
            });
          }
        } catch (e) {
          log(`Error: ${e.message}`);
          const status = (e.statusCode === 429 || e.message.includes('tokens exhausted') || e.message.includes('EXHAUSTED') || e.message.includes('usage limit') || e.message.includes('cooldown')) ? 429 : e.message.includes('Agent SDK timeout') ? 504 : 502;
          if (!res.headersSent) {
            if (status === 429 && e.retryAfterSec) {
              res.setHeader('Retry-After', String(e.retryAfterSec));
            } else if (status === 429) {
              res.setHeader('Retry-After', '60');
            }
            safeEnd(res, status, { error: { message: e.message } });
          } else {
            // Already streaming — send error as final SSE event and close
            res.write(`data: ${JSON.stringify({ error: { message: e.message } })}\n\n`);
            res.write('data: [DONE]\n\n');
            res.end();
          }
        }
      });

      req.on('error', () => {
        safeEnd(res, 400, { error: { message: 'Request stream error' } });
      });

      return;
    }

    // --- POST /v1/messages — Transparent Anthropic API proxy ---
    // Used by external agents via anthropic Python SDK
    // Forwards requests as-is to api.anthropic.com with pool token injection
    if (req.method === 'POST' && (req.url === '/v1/messages' || req.url === '/messages')) {
      let body = '';
      let bodySize = 0;

      req.on('data', c => {
        bodySize += c.length;
        if (bodySize > MAX_BODY_SIZE) {
          safeEnd(res, 413, { error: { type: 'error', message: 'Request body too large' } });
          req.destroy();
          return;
        }
        body += c;
      });

      req.on('end', async () => {
        if (res.headersSent) return;

        // Select token: round-robin by default, or specific token via header
        const tokenAffinity = req.headers['x-substation-token'];
        let targetToken = null;

        if (tokenAffinity) {
          const anthropicTokens = pool.filter(t => t.provider === 'anthropic' && !t.dead);
          targetToken = anthropicTokens.find(t => t.id === tokenAffinity || t.id.includes(tokenAffinity));
        }
        if (!targetToken) targetToken = selectAnthropicToken();

        if (!targetToken) {
          safeEnd(res, 503, { type: 'error', error: { type: 'overloaded_error', message: 'No Anthropic tokens available' } });
          return;
        }

        log(`Anthropic proxy (token=${targetToken.id}, body=${body.length}b)`);
        markRequestStart(targetToken);

        // Forward to Anthropic API — transparent passthrough
        const proxyHeaders = {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${targetToken.token}`,
          'anthropic-version': req.headers['anthropic-version'] || '2023-06-01',
          'anthropic-client-type': 'claude-code',
          'anthropic-client-version': '2.1.114',
          'User-Agent': 'claude-code/2.1.114',
          'Content-Length': Buffer.byteLength(body),
        };
        // Forward anthropic-beta header if present
        if (req.headers['anthropic-beta']) {
          proxyHeaders['anthropic-beta'] = req.headers['anthropic-beta'];
        } else {
          proxyHeaders['anthropic-beta'] = ANTHROPIC_BETA_HEADERS;
        }

        const proxyReq = httpsRequest({
          hostname: 'api.anthropic.com',
          path: '/v1/messages',
          method: 'POST',
          headers: proxyHeaders,
        }, (proxyRes) => {
          // Pipe response back to client — transparent passthrough
          const statusCode = proxyRes.statusCode;
          const respHeaders = {};
          for (const [k, v] of Object.entries(proxyRes.headers)) {
            if (k.startsWith('content-type') || k.startsWith('anthropic') || k === 'retry-after') {
              respHeaders[k] = v;
            }
          }
          res.writeHead(statusCode, respHeaders);
          proxyRes.pipe(res);

          proxyRes.on('end', () => {
            if (statusCode >= 200 && statusCode < 300) {
              markSuccess(targetToken);
            } else if (statusCode === 429) {
              markRateLimited(targetToken, parseInt(proxyRes.headers['retry-after'] || '60', 10));
            } else if (statusCode === 401) {
              markDead(targetToken);
            } else {
              markRequestEnd(targetToken);
            }
          });
        });

        proxyReq.on('error', (e) => {
          markRequestEnd(targetToken);
          log(`Anthropic proxy error: ${e.message}`);
          if (!res.headersSent) {
            safeEnd(res, 502, { type: 'error', error: { type: 'api_error', message: `Proxy error: ${e.message}` } });
          }
        });

        const timer = setTimeout(() => {
          proxyReq.destroy();
          markRequestEnd(targetToken);
          if (!res.headersSent) {
            safeEnd(res, 504, { type: 'error', error: { type: 'timeout_error', message: 'Anthropic proxy request timed out' } });
          }
        }, TIMEOUT);
        proxyReq.on('close', () => clearTimeout(timer));

        proxyReq.write(body);
        proxyReq.end();
      });

      req.on('error', () => {
        safeEnd(res, 400, { type: 'error', error: { type: 'invalid_request_error', message: 'Request stream error' } });
      });

      return;
    }

    // --- POST /v1/images/generations — ChatGPT conversation API image proxy ---
    if (req.method === 'POST' && (req.url === '/v1/images/generations' || req.url === '/images/generations')) {
      let body = '';
      let bodySize = 0;

      req.on('data', c => {
        bodySize += c.length;
        if (bodySize > MAX_BODY_SIZE) {
          safeEnd(res, 413, { error: { message: 'Request body too large' } });
          req.destroy();
          return;
        }
        body += c;
      });

      req.on('end', async () => {
        if (res.headersSent) return;

        let parsed;
        try { parsed = JSON.parse(body); } catch {
          safeEnd(res, 400, { error: { message: 'Invalid JSON in request body' } });
          return;
        }

        const { prompt, model: _reqModel, size = '1024x1024', quality = 'auto', n = 1 } = parsed;
        if (!prompt) {
          safeEnd(res, 400, { error: { message: 'Missing required field: prompt' } });
          return;
        }

        const tokenEntry = selectToken('openai');
        if (!tokenEntry) {
          safeEnd(res, 503, { error: { message: 'No OpenAI tokens available — run "substation-auth chatgpt" to add one' } });
          return;
        }

        log(`Image gen request (token=${tokenEntry.id}, prompt="${prompt.slice(0, 80)}...")`);
        markRequestStart(tokenEntry);

        try {
          const results = [];
          for (let i = 0; i < Math.min(n, 4); i++) {
            const result = await chatgptImageGenerate(tokenEntry, prompt, size, quality);
            // result.b64 is already base64 from the Codex/Responses API stream
            results.push({ b64: result.b64, revised_prompt: prompt });
          }

          markSuccess(tokenEntry);

          safeEnd(res, 200, {
            created: Math.floor(Date.now() / 1000),
            data: results.map(r => ({ b64_json: r.b64, revised_prompt: r.revised_prompt })),
          });
        } catch (err) {
          markRequestEnd(tokenEntry);
          log(`Image gen error: ${err.message}`);

          // On 401/403 — try refreshing the token before giving up (matches text model behavior)
          if ((err.statusCode === 401 || err.statusCode === 403) && tokenEntry.refreshToken) {
            try {
              log(`Image gen: token ${tokenEntry.id} got ${err.statusCode}, attempting refresh...`);
              await refreshOpenAIToken(tokenEntry);
              markRequestStart(tokenEntry);
              const results = [];
              for (let i = 0; i < Math.min(n, 4); i++) {
                const result = await chatgptImageGenerate(tokenEntry, prompt, size, quality);
                results.push({ b64: result.b64, revised_prompt: prompt });
              }
              markSuccess(tokenEntry);
              safeEnd(res, 200, {
                created: Math.floor(Date.now() / 1000),
                data: results.map(r => ({ b64_json: r.b64, revised_prompt: r.revised_prompt })),
              });
              return;
            } catch (refreshErr) {
              log(`Image gen: refresh failed for ${tokenEntry.id}: ${refreshErr.message}`);
              markDead(tokenEntry);
            }
          } else if (err.statusCode === 401 || err.statusCode === 403) {
            markDead(tokenEntry);
          }

          if (!res.headersSent) {
            safeEnd(res, err.statusCode && err.statusCode >= 400 ? err.statusCode : 502, {
              error: { message: `Image generation failed: ${err.message}` },
            });
          }
        }
      });

      req.on('error', () => {
        safeEnd(res, 400, { error: { message: 'Request stream error' } });
      });

      return;
    }

    res.writeHead(404); res.end('Not found');
  });

  // Handle server errors
  let bindRetries = 0;
  const MAX_BIND_RETRIES = 3;
  proxyServer.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      bindRetries++;
      if (bindRetries > MAX_BIND_RETRIES) {
        log(`ERROR: Port ${PORT} still in use after ${MAX_BIND_RETRIES} retries. Kill the process manually: /usr/sbin/lsof -ti :${PORT} | xargs kill`);
        return;
      }
      log(`Port ${PORT} in use — attempting to free it (attempt ${bindRetries}/${MAX_BIND_RETRIES})...`);
      if (killStaleProcess(PORT)) {
        log('Stale process killed, retrying bind...');
        setTimeout(() => {
          try { proxyServer.close(); } catch {}
          proxyServer.listen(PORT, '127.0.0.1');
        }, 1000);
      } else {
        log(`ERROR: Could not free port ${PORT}. Kill the process manually: /usr/sbin/lsof -ti :${PORT} | xargs kill`);
      }
    } else {
      log(`Server error: ${err.message}`);
    }
  });

  // Handle client connection errors (prevent EPIPE crashes)
  proxyServer.on('clientError', (err, socket) => {
    if (socket.writable) {
      socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
    }
  });

  proxyServer.listen(PORT, '127.0.0.1', () => {
    const status = getPoolStatus();
    log(`SubStation v${VERSION} on http://127.0.0.1:${PORT} (pool=${status.total}, anthropic=${status.byProvider.anthropic}, openai=${status.byProvider.openai}, cc=${VERSION})`);
  });

  function gracefulShutdown() {
    log('shutting down gracefully...');
    proxyServer.close(() => {
      process.exit(0);
    });
    setTimeout(() => {
      log('forcing exit after 10s timeout');
      process.exit(0);
    }, 10000).unref();
  }
  process.on('SIGTERM', gracefulShutdown);
  process.on('SIGINT', gracefulShutdown);
}

// ---------------------------------------------------------------------------
// Plugin Entry
// ---------------------------------------------------------------------------

const ANTHROPIC_MODELS = [
  {
    id: 'opus-4-8',
    name: 'Claude Opus 4.8 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1000000,
    maxTokens: 128000,
  },
  {
    id: 'opus-4-7',
    name: 'Claude Opus 4.7 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1000000,
    maxTokens: 128000,
  },
  {
    id: 'opus-4-6',
    name: 'Claude Opus 4.6 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1000000,
    maxTokens: 128000,
  },
  {
    id: 'sonnet-4-6',
    name: 'Claude Sonnet 4.6 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1000000,
    maxTokens: 64000,
  },
  {
    id: 'haiku-4-5',
    name: 'Claude Haiku 4.5 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1000000,
    maxTokens: 64000,
  },
];

const OPENAI_MODELS = [
  {
    id: 'gpt-5.4',
    name: 'GPT 5.4 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 128000,
  },
  {
    id: 'gpt-5.4-mini',
    name: 'GPT 5.4 Mini (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 64000,
  },
  {
    id: 'gpt-5.1-codex',
    name: 'GPT 5.1 Codex (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 128000,
  },
  {
    id: 'gpt-5.1-codex-mini',
    name: 'GPT 5.1 Codex Mini (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 64000,
  },
  {
    id: 'gpt-5.1-codex-max',
    name: 'GPT 5.1 Codex Max (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 128000,
  },
];

function getAllModels() {
  const models = [...ANTHROPIC_MODELS];
  // Per-account pinned variants: "opus-4-6:acct2-alice" routes to that specific token
  const anthropicPoolTokens = pool.filter(t => t.provider === 'anthropic' && !t.dead);
  for (const t of anthropicPoolTokens) {
    models.push(
      { id: `opus-4-8:${t.id}`, name: `Opus 4.8 — ${t.id}`, reasoning: false, input: ['text'], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 1000000, maxTokens: 128000 },
      { id: `opus-4-6:${t.id}`, name: `Opus 4.6 — ${t.id}`, reasoning: false, input: ['text'], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 1000000, maxTokens: 128000 },
      { id: `sonnet-4-6:${t.id}`, name: `Sonnet 4.6 — ${t.id}`, reasoning: false, input: ['text'], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 1000000, maxTokens: 64000 },
    );
  }
  if (pool.some(t => t.provider === 'openai')) {
    models.push(...OPENAI_MODELS);
    // Per-account pinned ChatGPT variants
    const openaiPoolTokens = pool.filter(t => t.provider === 'openai' && !t.dead);
    for (const t of openaiPoolTokens) {
      models.push(
        { id: `gpt-5.4:${t.id}`, name: `GPT 5.4 — ${t.id}`, reasoning: false, input: ['text'], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 200000, maxTokens: 128000 },
        { id: `gpt-5.4-mini:${t.id}`, name: `GPT 5.4 Mini — ${t.id}`, reasoning: false, input: ['text'], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 200000, maxTokens: 64000 },
      );
    }
  }
  return models;
}

function buildProviderModels(baseUrl) {
  return {
    baseUrl,
    apiKey: 'sk-substation-local-proxy',
    api: 'openai-completions',
    models: getAllModels(),
  };
}

const indigoProvider = {
  id: 'indigo',
  label: 'Indigo (SubStation)',
  docsPath: 'https://indigocollective.dev/substation',
  aliases: ['substation', 'ss'],
  envVars: [],
  get models() {
    return buildProviderModels(`http://127.0.0.1:${PORT}/v1`);
  },
  auth: [],
};

export default {
  id: 'substation',
  name: 'SubStation',
  description: 'Route through Claude Max + ChatGPT Pro subscriptions — zero API cost',
  register(api) {
    log('SubStation plugin registering...');

    startProxy();

    api.registerProvider(indigoProvider);

    if (!api.config.models) api.config.models = { providers: {} };
    if (!api.config.models.providers) api.config.models.providers = {};

    api.config.models.providers.indigo = {
      baseUrl: `http://127.0.0.1:${PORT}/v1`,
      api: 'openai-completions',
      apiKey: 'sk-substation-local-proxy',
      models: getAllModels(),
    };

    api.logger.info(`SubStation v${VERSION} registered (${getAllModels().length} models via direct API)`);
    log(`SubStation plugin registered — ${getAllModels().length} models, ${pool.length} tokens`);
  },
};

// Standalone mode: if run directly (not imported as plugin), start the proxy server
const isMainModule = import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  log(`SubStation v${VERSION} starting in standalone mode...`);
  startProxy();
}

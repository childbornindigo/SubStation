import { createServer } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { execSync } from 'node:child_process';
import { randomUUID, randomBytes, createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { readFileSync, writeFileSync, mkdirSync, renameSync, statSync, watch } from 'node:fs';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const VERSION = '0.6.3';
const TIMEOUT = 300000; // 5 min

// Operator system prompt — replaces Claude Code's default system prompt
const OPERATOR_SYSTEM_PROMPT = `You are an AI assistant powered by SubStation.
You are NOT Claude Code. You do not identify as Claude Code or as "Claude, an AI assistant by Anthropic."
Your operator has configured you through SubStation with their own identity and instructions.
Follow the system instructions provided in each conversation. Use tools when needed.`;
const MAX_BODY_SIZE = 2 * 1024 * 1024; // 2MB request body limit
const LOG_MAX_SIZE = 5 * 1024 * 1024; // 5MB log rotation
const DEAD_REVIVE_MS = 30 * 60 * 1000;       // 30 min — auto-retry rate-limited tokens
const AUTH_DEAD_REVIVE_MS = 24 * 60 * 60 * 1000; // 24h — auto-retry auth-failed tokens
const DATA_DIR = join(homedir(), '.substation', 'data');
const PORT = parseInt(process.env.SUBSTATION_PORT || '8403');

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
  'claude-opus-4-6':           { maxTokens: 128000, adaptive: true,  provider: 'anthropic', contextWindow: 200000 },
  'claude-sonnet-4-6':         { maxTokens: 64000,  adaptive: true,  provider: 'anthropic', contextWindow: 200000 },
  'claude-haiku-4-5-20251001': { maxTokens: 64000,  adaptive: false, provider: 'anthropic', contextWindow: 200000 },
  // OpenAI (Codex) — ordered fastest to slowest
  'gpt-5.4-mini':       { maxTokens: 64000,  adaptive: false, provider: 'openai', contextWindow: 200000 },
  'gpt-5.4':            { maxTokens: 128000, adaptive: false, provider: 'openai', contextWindow: 200000 },
  'gpt-5.1-codex':      { maxTokens: 128000, adaptive: false, provider: 'openai', contextWindow: 200000 },
  'gpt-5.1-codex-mini': { maxTokens: 64000,  adaptive: false, provider: 'openai', contextWindow: 128000 },
  'gpt-5.1-codex-max':  { maxTokens: 128000, adaptive: false, provider: 'openai', contextWindow: 200000 },
};

const MODEL_MAP = {
  // Anthropic aliases
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

function resolveModel(model) {
  const clean = model.includes('/') ? model.split('/').pop() : model;
  const resolved = MODEL_MAP[clean];
  if (resolved) return { modelId: resolved, ...(MODEL_CONFIG[resolved] || { maxTokens: 64000, adaptive: false, provider: 'anthropic', contextWindow: 200000 }) };
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
  entry.activeRequests = Math.max(0, entry.activeRequests - 1);
  savePoolState();
}

function markRateLimited(entry, retryAfterSec) {
  const cooldownMs = Math.min((retryAfterSec || 60) * 1000, 3600000); // cap at 1hr
  entry.cooldownUntil = Date.now() + cooldownMs;
  entry.lastUsed = Date.now();
  entry.errorCount++;
  entry.activeRequests = Math.max(0, entry.activeRequests - 1);
  savePoolState();
  log(`Token ${entry.id} rate-limited, cooldown ${retryAfterSec || 60}s until ${new Date(entry.cooldownUntil).toISOString()}`);
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
      id: t.id,
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
    const content = Array.isArray(m.content)
      ? m.content.filter(b => b.type === 'text').map(b => b.text).join('\n')
      : (m.content || '');

    if (m.role === 'system') {
      systemBlocks.push({ type: 'text', text: content });
    } else {
      messages.push({
        role: m.role === 'user' ? 'user' : 'assistant',
        content,
      });
    }
  }

  // Anthropic requires alternating roles starting with user
  if (messages.length > 0 && messages[0].role === 'assistant') {
    messages.unshift({ role: 'user', content: '(continue)' });
  }
  // Merge consecutive same-role messages
  const merged = [];
  for (const msg of messages) {
    if (merged.length > 0 && merged[merged.length - 1].role === msg.role) {
      merged[merged.length - 1].content += '\n\n' + msg.content;
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
    model: modelInfo.modelId,
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
        'anthropic-client-version': '1.0.33',
        'User-Agent': 'claude-code/1.0.33',
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
const warmSessions = new Map(); // "model:tokenId" → { session, tokenId, lastUsed, active, cooldownUntil }
let anthropicRRIndex = 0; // Round-robin counter for Anthropic token selection

async function getOrCreateSession(model, tokenId, oauthToken) {
  const key = `${model}:${tokenId}`;
  const existing = warmSessions.get(key);
  if (existing) {
    existing.lastUsed = Date.now();
    return existing.session;
  }

  const sdk = await loadAgentSDK();
  log(`Creating persistent session for ${model} with token ${tokenId} (first request slow, then instant)...`);

  // Pass the specific OAuth token via env so each session uses its own account
  const sessionEnv = { ...process.env };
  if (oauthToken) {
    sessionEnv.CLAUDE_CODE_OAUTH_TOKEN = oauthToken;
  }
  const session = sdk.unstable_v2_createSession({
    model,
    permissionMode: 'bypassPermissions',
    allowDangerouslySkipPermissions: true,
    pathToClaudeCodeExecutable: process.env.CLAUDE_BIN || '/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js',
    env: sessionEnv,
  });
  log(`Session created for ${key} with token ${tokenId}`);
  const entry = { session, tokenId, lastUsed: Date.now(), active: 0, cooldownUntil: null };
  warmSessions.set(key, entry);

  // Auto-close idle sessions
  const interval = setInterval(() => {
    const e = warmSessions.get(key);
    if (e && Date.now() - e.lastUsed > SESSION_IDLE_TIMEOUT) {
      log(`Closing idle session ${key}`);
      try { e.session.close(); } catch {}
      warmSessions.delete(key);
      clearInterval(interval);
    }
  }, 60000);

  return session;
}

function selectAnthropicToken() {
  const tokens = getAnthropicOAuthTokens();
  if (tokens.length === 0) return null;
  if (tokens.length === 1) return tokens[0];

  // Round-robin: rotate through tokens, skipping any in cooldown
  const now = Date.now();
  for (let i = 0; i < tokens.length; i++) {
    const idx = (anthropicRRIndex + i) % tokens.length;
    const t = tokens[idx];

    // Check if this token is in cooldown
    let inCooldown = false;
    for (const [key, entry] of warmSessions) {
      if (entry.tokenId === t.id && entry.cooldownUntil && now < entry.cooldownUntil) {
        inCooldown = true;
        break;
      }
    }
    if (inCooldown) {
      log(`Token ${t.id} in cooldown, skipping`);
      continue;
    }

    // Advance counter for next call
    anthropicRRIndex = (idx + 1) % tokens.length;
    log(`Round-robin selected token ${t.id} (index ${idx}/${tokens.length})`);
    return t;
  }

  // All in cooldown — return first and let it fail/retry
  log('WARN: All Anthropic tokens in cooldown, using first available');
  anthropicRRIndex = 1 % tokens.length;
  return tokens[0];
}

// Format OpenAI-style messages into a single prompt string for the Agent SDK session.
// Preserves system prompts (operator identity/personality), full conversation history,
// and tool results so Claude has complete context — not just the last user message.
function formatMessagesForSDK(openaiMessages) {
  const parts = [];
  const systemParts = [];
  const conversationParts = [];

  for (const msg of openaiMessages) {
    const text = Array.isArray(msg.content)
      ? msg.content.filter(b => b.type === 'text').map(b => b.text).join('\n')
      : (msg.content || '');

    if (!text.trim()) continue;

    if (msg.role === 'system') {
      systemParts.push(text);
    } else {
      conversationParts.push({ role: msg.role, text });
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

async function invokeClaudeSDK(openaiMessages, modelInfo, onDelta) {
  const model = modelInfo.modelId;
  const tokens = getAnthropicOAuthTokens();
  if (tokens.length === 0) {
    throw new Error('No Anthropic OAuth tokens found. Log into Claude Code or add tokens to auth-profiles.json.');
  }

  // Format full conversation context including system prompts and history
  const prompt = formatMessagesForSDK(openaiMessages);

  // Try tokens with rotation
  const maxAttempts = tokens.length;
  let lastError = null;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const token = attempt === 0 ? selectAnthropicToken() : tokens[attempt % tokens.length];
    if (!token) break;

    const key = `${model}:${token.id}`;
    log(`Agent SDK request (model=${model}, token=${token.id}, attempt=${attempt + 1}/${maxAttempts})`);

    try {
      const session = await getOrCreateSession(model, token.id, token.token);

      // Wrap send+stream in a timeout to catch silent hangs (e.g. usage limits)
      const SDK_TIMEOUT = 120000; // 2 min timeout per attempt
      const result = await Promise.race([
        (async () => {
          await session.send(prompt);

          let text = '';
          let usage = {};

          for await (const msg of session.stream()) {
            log(`Stream msg type=${msg.type}: ${JSON.stringify(msg).substring(0, 300)}`);
            if (msg.type === 'assistant') {
              for (const block of (msg.message?.content || [])) {
                if (block.type === 'text' || 'text' in block) {
                  const chunk = block.text || '';
                  if (chunk) {
                    text += chunk;
                    if (onDelta) onDelta(chunk);
                  }
                }
              }
            } else if (msg.type === 'result') {
              if (msg.usage) usage = msg.usage;
              if (msg.result && !text) text = msg.result;
            }
          }

          return { text, usage };
        })(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Agent SDK timeout — likely hit usage limit (no response in 2min)')), SDK_TIMEOUT)),
      ]);

      const { text, usage } = result;

      // Check if the response itself contains a usage limit message
      const textLower = (text || '').toLowerCase();
      if (textLower.includes('hit your limit') || textLower.includes('usage limit') || textLower.includes('resets ')) {
        throw new Error(`Usage limit detected in response: ${text.substring(0, 200)}`);
      }

      if (!text) throw new Error('Empty response from Claude via Agent SDK');

      // Mark success
      const entry = warmSessions.get(key);
      if (entry) entry.lastUsed = Date.now();

      return { text, usage };
    } catch (err) {
      lastError = err;
      log(`Agent SDK error (token=${token.id}): ${err.message}`);

      // Destroy broken session
      const entry = warmSessions.get(key);
      if (entry) {
        try { entry.session.close(); } catch {}
        warmSessions.delete(key);
      }

      // If rate/usage limited, cooldown this token for 30 min and try next
      const errLower = (err.message || '').toLowerCase();
      if (errLower.includes('rate') || errLower.includes('limit') || errLower.includes('usage') || errLower.includes('hit your') || errLower.includes('resets')) {
        const cooldownMs = DEAD_REVIVE_MS; // 30 min
        log(`Token ${token.id} hit usage/rate limit — cooldown ${cooldownMs/60000}min, trying next...`);
        // Mark ALL sessions for this token as in cooldown
        for (const [k, e] of warmSessions) {
          if (e.tokenId === token.id) e.cooldownUntil = Date.now() + cooldownMs;
        }
        // Also create a placeholder entry if no session exists yet
        if (!warmSessions.has(key)) {
          warmSessions.set(key, { session: null, tokenId: token.id, lastUsed: 0, active: 0, cooldownUntil: Date.now() + cooldownMs });
        }
        continue;
      }

      // Auth error — bench this token for 24h and try next
      if (isAuthError(err)) {
        const poolEntry = pool.find(p => p.id === token.id);
        if (poolEntry) {
          markDead(poolEntry, 'auth');
        } else {
          log(`Auth error on token ${token.id} (not in pool, skipping for this session)`);
        }
      }
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
  'claude-opus-4-6':           'gpt-5.4',
  'claude-sonnet-4-6':         'gpt-5.1-codex',
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
  const modelInfo = resolveModel(model);

  if (modelInfo.provider === 'anthropic') {
    // --- Step 1: Try Agent SDK with token rotation (already rotates internally) ---
    let sdkErr = null;
    try {
      if (opts.tokenAffinity) {
        return await invokeClaudeDirect(openaiMessages, modelInfo, opts.tokenAffinity, onDelta);
      }
      return await invokeClaudeSDK(openaiMessages, modelInfo, onDelta);
    } catch (err) {
      sdkErr = err;
      if (!isCapError(err)) throw err;
      log(`SDK path exhausted all tokens: ${err.message}`);
    }

    // --- Step 2: Try Direct API with each pool token (different auth path) ---
    const directTokens = pool.filter(t => t.provider === 'anthropic' && !t.dead);
    for (const token of directTokens) {
      try {
        log(`Trying Direct API with token ${token.id}...`);
        return await invokeClaudeDirect(openaiMessages, modelInfo, token.id, onDelta);
      } catch (directErr) {
        if (isCapError(directErr)) {
          log(`Direct API token ${token.id} also capped`);
          continue;
        }
        throw directErr;
      }
    }

    // --- Step 3: All Anthropic tokens exhausted → failover to Codex ---
    const failoverModelId = FAILOVER_MAP[modelInfo.modelId];
    if (!failoverModelId) throw sdkErr;

    const failoverInfo = resolveModel(failoverModelId);
    const codexTokens = pool.filter(t => t.provider === 'openai' && !t.dead);
    if (codexTokens.length === 0) {
      log(`No Codex tokens available for failover`);
      throw sdkErr;
    }

    log(`>>> FAILOVER: All Anthropic tokens exhausted → ${failoverModelId} (Codex)`);
    try {
      return await invokeCodex(openaiMessages, failoverInfo, onDelta);
    } catch (codexErr) {
      log(`Codex failover also failed: ${codexErr.message}`);
      throw sdkErr;  // return original Anthropic error
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
    const pid = execSync(`lsof -ti :${port}`, { timeout: 5000 }).toString().trim();
    if (pid && pid !== String(process.pid)) {
      log(`Killing stale process ${pid} on port ${port}`);
      execSync(`kill ${pid}`, { timeout: 5000 });
      // Wait for it to die
      for (let i = 0; i < 10; i++) {
        try {
          execSync(`lsof -ti :${port}`, { timeout: 2000 });
          execSync('sleep 0.3');
        } catch {
          return true; // port freed
        }
      }
    }
  } catch {
    // lsof returned empty = port is free
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Proxy server
// ---------------------------------------------------------------------------

let proxyServer = null;

function safeEnd(res, statusCode, body) {
  if (res.headersSent) return;
  res.writeHead(statusCode, { 'Content-Type': 'application/json' });
  res.end(typeof body === 'string' ? body : JSON.stringify(body));
}

function startProxy() {
  if (proxyServer) return;

  proxyServer = createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') { res.writeHead(200); res.end(); return; }

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
          { id: 'opus-4-6', object: 'model', owned_by: 'indigo-collective' },
          { id: 'sonnet-4-6', object: 'model', owned_by: 'indigo-collective' },
          { id: 'haiku-4-5', object: 'model', owned_by: 'indigo-collective' },
        ];
        if (pool.some(t => t.provider === 'openai')) {
          models.push(
            { id: 'gpt-5.4', object: 'model', owned_by: 'indigo-collective' },
            { id: 'gpt-5.4-mini', object: 'model', owned_by: 'indigo-collective' },
            { id: 'gpt-5.1-codex', object: 'model', owned_by: 'indigo-collective' },
            { id: 'gpt-5.1-codex-mini', object: 'model', owned_by: 'indigo-collective' },
            { id: 'gpt-5.1-codex-max', object: 'model', owned_by: 'indigo-collective' },
          );
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
          const status = (e.message.includes('tokens exhausted') || e.message.includes('EXHAUSTED') || e.statusCode === 429) ? 429 : 502;
          if (!res.headersSent) {
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
          'anthropic-client-version': '1.0.33',
          'User-Agent': 'claude-code/1.0.33',
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

    res.writeHead(404); res.end('Not found');
  });

  // Handle server errors
  proxyServer.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      log(`Port ${PORT} in use — attempting to free it...`);
      if (killStaleProcess(PORT)) {
        log('Stale process killed, retrying bind...');
        setTimeout(() => {
          try { proxyServer.close(); } catch {}
          proxyServer.listen(PORT, '127.0.0.1');
        }, 1000);
      } else {
        log(`ERROR: Could not free port ${PORT}. Kill the process manually: lsof -ti :${PORT} | xargs kill`);
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
}

// ---------------------------------------------------------------------------
// Plugin Entry
// ---------------------------------------------------------------------------

const ANTHROPIC_MODELS = [
  {
    id: 'opus-4-6',
    name: 'Claude Opus 4.6 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 128000,
  },
  {
    id: 'sonnet-4-6',
    name: 'Claude Sonnet 4.6 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 64000,
  },
  {
    id: 'haiku-4-5',
    name: 'Claude Haiku 4.5 (SubStation)',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
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
  if (pool.some(t => t.provider === 'openai')) {
    models.push(...OPENAI_MODELS);
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

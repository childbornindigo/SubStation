// auth-policy.js — SubStation token auth/bench policy (extracted for testability)
//
// Root cause this module fixes (2026-07-11): a single transient 401/403 from
// Anthropic's edge was benching the sole valid OAuth token for 24h, causing the
// daily "No Anthropic OAuth tokens found" outage. See handoff_substation_oauth_*.
//
// Three fixes live here:
//   1. AUTH_FAIL_THRESHOLD — require N *consecutive* auth failures before benching.
//   2. AUTH_DEAD_REVIVE_MS  — 24h -> 5min, so even a real bench self-heals fast.
//   (Fix 3, live-credentials re-read, lives in index.js getAnthropicOAuthTokens.)

export const DEAD_REVIVE_MS = 30 * 60 * 1000;        // 30 min — rate-limited tokens
export const AUTH_DEAD_REVIVE_MS = 5 * 60 * 1000;    // 5 min  — auth-failed tokens (was 24h)
export const AUTH_FAIL_THRESHOLD = 3;                // consecutive auth failures before benching

// Classify an error as an auth failure (unchanged behavior — same matchers as before).
export function isAuthError(err) {
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

// Record a consecutive auth failure on the pool entry. Returns true ONLY when the
// entry has now hit the threshold and should actually be benched. A single blip
// (count < threshold) returns false — the token stays in service.
export function recordAuthFailure(entry, threshold = AUTH_FAIL_THRESHOLD) {
  entry._consecutiveAuthFails = (entry._consecutiveAuthFails || 0) + 1;
  return entry._consecutiveAuthFails >= threshold;
}

// Reset the consecutive-auth-failure counter (call on any success).
export function resetAuthFailures(entry) {
  entry._consecutiveAuthFails = 0;
}

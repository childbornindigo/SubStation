// auth-policy.test.mjs — TDD proof that a single transient 401/403 no longer
// benches the sole valid OAuth token. Plain node ESM, no test framework.
//
// Bug: a single transient 401/403 from Anthropic's edge benched the sole valid
// OAuth token for 24h → daily "No Anthropic OAuth tokens found" outage.
// Fix lives in ../dist/auth-policy.js (AUTH_FAIL_THRESHOLD=3, AUTH_DEAD_REVIVE_MS=5min).

import assert from 'node:assert/strict';
import {
  isAuthError,
  recordAuthFailure,
  resetAuthFailures,
  AUTH_FAIL_THRESHOLD,
  AUTH_DEAD_REVIVE_MS,
  DEAD_REVIVE_MS,
} from '../dist/auth-policy.js';

function fail(msg) {
  console.error(`FAIL: ${msg}`);
  process.exit(1);
}

function check(label, fn) {
  try {
    fn();
    console.log(`  PASS: ${label}`);
  } catch (e) {
    fail(`${label}\n    ${e.message}`);
  }
}

// (a) Classification unchanged
console.log('(a) isAuthError classification unchanged:');
check('statusCode 403 → true', () => assert.equal(isAuthError({ statusCode: 403 }), true));
check('statusCode 401 → true', () => assert.equal(isAuthError({ statusCode: 401 }), true));
check("'organization does not have access' → true", () =>
  assert.equal(isAuthError({ message: 'organization does not have access' }), true));
check("'request timeout' → false", () =>
  assert.equal(isAuthError({ message: 'request timeout' }), false));
check('empty {} → false', () => assert.equal(isAuthError({}), false));

// (b) THE KEY PROOF — a single 403 does NOT bench
console.log('(b) single 403 does NOT bench (needs 3 consecutive):');
{
  const entry = {};
  const r1 = recordAuthFailure(entry);
  const r2 = recordAuthFailure(entry);
  const r3 = recordAuthFailure(entry);
  check('1st failure (1/3) → false (stays in service)', () => assert.equal(r1, false));
  check('2nd failure (2/3) → false (stays in service)', () => assert.equal(r2, false));
  check('3rd failure (3/3) → true (now benches)', () => assert.equal(r3, true));
}

// (c) Success resets the transient count
console.log('(c) a good response resets the count:');
{
  const entry = {};
  recordAuthFailure(entry); // 1/3
  recordAuthFailure(entry); // 2/3
  resetAuthFailures(entry); // success clears
  const afterReset = recordAuthFailure(entry); // back to 1/3
  check('after reset, next failure → false (count back to 1/3)', () =>
    assert.equal(afterReset, false));
  check('reset zeroed the counter before increment', () =>
    assert.equal(entry._consecutiveAuthFails, 1));
}

// (d) Constants prove the 24h→5min and threshold fixes
console.log('(d) constants:');
check('AUTH_DEAD_REVIVE_MS === 5*60*1000 (NOT 24h)', () =>
  assert.equal(AUTH_DEAD_REVIVE_MS, 5 * 60 * 1000));
check('AUTH_FAIL_THRESHOLD === 3', () => assert.equal(AUTH_FAIL_THRESHOLD, 3));
check('DEAD_REVIVE_MS === 30*60*1000 (unchanged rate-limit revive)', () =>
  assert.equal(DEAD_REVIVE_MS, 30 * 60 * 1000));
// Guard against a silent regression back to the 24h bench.
check('AUTH_DEAD_REVIVE_MS is NOT the old 24h value', () =>
  assert.notEqual(AUTH_DEAD_REVIVE_MS, 24 * 60 * 60 * 1000));

// REAL-GUARD PROOF — this test distinguishes OLD-broken from NEW-fixed behavior.
// Under the OLD policy (threshold=1) a single failure WOULD bench. If this test
// harness were toothless it would not catch that; here we prove it does.
console.log('(guard) OLD policy (threshold=1) WOULD bench on a single failure:');
{
  const oldEntry = {};
  const benchedImmediately = recordAuthFailure(oldEntry, 1);
  check('recordAuthFailure(oldEntry, 1) → true (old bug reproduced)', () =>
    assert.equal(benchedImmediately, true));
  // And confirm the NEW default (threshold=3) does NOT bench that same first failure.
  const newEntry = {};
  check('same first failure under default threshold=3 → false (bug fixed)', () =>
    assert.equal(recordAuthFailure(newEntry), false));
}

console.log('\nALL PASS');
process.exit(0);

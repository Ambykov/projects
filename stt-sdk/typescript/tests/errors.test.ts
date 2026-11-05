import { describe, it, expect } from 'vitest';
import { STTError, translateCloseCode } from '../src/errors.js';

describe('STTError', () => {
  it('is retryable for 4 Gap 1 codes', () => {
    for (const code of [
      'pool_unavailable',
      'upstream_ready_timeout',
      'upstream_recv_timeout',
      'upstream_connect_failed',
    ]) {
      expect(new STTError(code, 'msg').isRetryable).toBe(true);
    }
  });

  it('is retryable for close codes session_limit, idle_timeout', () => {
    expect(new STTError('session_limit', 'msg').isRetryable).toBe(true);
    expect(new STTError('idle_timeout', 'msg').isRetryable).toBe(true);
  });

  it('is not retryable for auth/protocol/client errors', () => {
    for (const code of [
      'unauthorized',
      'protocol_error',
      'client_timeout',
      'ws_not_open',
      'connection_closed',
      'ws_error',
      'unknown_code',
    ]) {
      expect(new STTError(code, 'msg').isRetryable).toBe(false);
    }
  });

  it('explicit retryable override wins over whitelist', () => {
    expect(new STTError('unauthorized', 'msg', true).isRetryable).toBe(true);
    expect(new STTError('pool_unavailable', 'msg', false).isRetryable).toBe(false);
  });

  it('sets code, message, name, instanceof Error', () => {
    const e = new STTError('code1', 'msg1');
    expect(e.code).toBe('code1');
    expect(e.message).toBe('code1: msg1');
    expect(e.name).toBe('STTError');
    expect(e instanceof Error).toBe(true);
  });
});

describe('translateCloseCode', () => {
  it('4401 → unauthorized, not retryable', () => {
    const e = translateCloseCode(4401, 'auth fail');
    expect(e.code).toBe('unauthorized');
    expect(e.isRetryable).toBe(false);
  });

  it('4429 → session_limit, retryable', () => {
    const e = translateCloseCode(4429, 'limit');
    expect(e.code).toBe('session_limit');
    expect(e.isRetryable).toBe(true);
  });

  it('4408 → idle_timeout, retryable', () => {
    const e = translateCloseCode(4408, 'idle');
    expect(e.code).toBe('idle_timeout');
    expect(e.isRetryable).toBe(true);
  });

  it('unknown code → connection_closed, not retryable', () => {
    const e = translateCloseCode(1006, 'abnormal');
    expect(e.code).toBe('connection_closed');
    expect(e.message).toContain('code=1006');
    expect(e.isRetryable).toBe(false);
  });
});

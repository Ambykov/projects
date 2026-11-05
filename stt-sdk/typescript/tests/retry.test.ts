import { describe, it, expect, vi } from 'vitest';
import { RetryPolicy, STTError, sleep } from '../src/errors.js';

describe('RetryPolicy.delay (exponential backoff)', () => {
  it('default: 1s, 2s, 4s, 8s, capped at 10s', () => {
    const p = new RetryPolicy();
    expect(p.delay(0)).toBe(1000);
    expect(p.delay(1)).toBe(2000);
    expect(p.delay(2)).toBe(4000);
    expect(p.delay(3)).toBe(8000);
    expect(p.delay(4)).toBe(10000); // cap
    expect(p.delay(10)).toBe(10000); // cap
  });

  it('respects custom baseDelayMs/maxDelayMs', () => {
    const p = new RetryPolicy({ baseDelayMs: 100, maxDelayMs: 500 });
    expect(p.delay(0)).toBe(100);
    expect(p.delay(1)).toBe(200);
    expect(p.delay(2)).toBe(400);
    expect(p.delay(3)).toBe(500); // cap
  });
});

describe('RetryPolicy.run', () => {
  it('returns on first success (no retry)', async () => {
    const p = new RetryPolicy({ maxAttempts: 3, baseDelayMs: 1 });
    const fn = vi.fn().mockResolvedValue('ok');
    await expect(p.run(fn)).resolves.toBe('ok');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('retries on retryable STTError, succeeds on 2nd attempt', async () => {
    const p = new RetryPolicy({ maxAttempts: 3, baseDelayMs: 1 });
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new STTError('pool_unavailable', 'busy'))
      .mockResolvedValueOnce('ok');
    await expect(p.run(fn)).resolves.toBe('ok');
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it('throws immediately on non-retryable STTError', async () => {
    const p = new RetryPolicy({ maxAttempts: 3, baseDelayMs: 1 });
    const err = new STTError('unauthorized', 'bad key');
    const fn = vi.fn().mockRejectedValue(err);
    await expect(p.run(fn)).rejects.toBe(err);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('throws immediately on non-STTError (no retry)', async () => {
    const p = new RetryPolicy({ maxAttempts: 3, baseDelayMs: 1 });
    const err = new Error('boom');
    const fn = vi.fn().mockRejectedValue(err);
    await expect(p.run(fn)).rejects.toBe(err);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('throws last error after maxAttempts exhausted', async () => {
    const p = new RetryPolicy({ maxAttempts: 3, baseDelayMs: 1 });
    const err = new STTError('pool_unavailable', 'busy');
    const fn = vi.fn().mockRejectedValue(err);
    await expect(p.run(fn)).rejects.toBe(err);
    expect(fn).toHaveBeenCalledTimes(3);
  });
});

describe('sleep', () => {
  it('resolves after ~ms', async () => {
    const start = Date.now();
    await sleep(50);
    expect(Date.now() - start).toBeGreaterThanOrEqual(40);
  });
});

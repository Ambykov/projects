export class STTError extends Error {
  constructor(
    public code: string,
    message: string,
    private readonly _retryableOverride?: boolean,
  ) {
    super(`${code}: ${message}`);
    this.name = 'STTError';
  }

  /**
   * Retryable errors:
   * - 4 Gap 1 codes (ADR-0014): pool_unavailable, upstream_ready_timeout,
   *   upstream_recv_timeout, upstream_connect_failed
   * - close codes: session_limit (4429), idle_timeout (4408)
   * Not retryable: unauthorized (4401), protocol_error, client_timeout, ws_not_open
   *
   * Override: explicit `retryable` in constructor wins over whitelist.
   */
  get isRetryable(): boolean {
    if (this._retryableOverride !== undefined) {
      return this._retryableOverride;
    }
    return [
      'pool_unavailable',
      'upstream_ready_timeout',
      'upstream_recv_timeout',
      'upstream_connect_failed',
      'session_limit',
      'idle_timeout',
    ].includes(this.code);
  }
}

export interface RetryPolicyOptions {
  /** Max attempts (including first). Default 3. */
  maxAttempts?: number;
  /** Base delay for exponential backoff (ms). Default 1000. */
  baseDelayMs?: number;
  /** Max delay cap (ms). Default 10000. */
  maxDelayMs?: number;
}

/**
 * Retry policy with exponential backoff: 1s, 2s, 4s... (capped at maxDelayMs).
 * Mirrors Python SDK RetryPolicy (sdk/python/stt_client/errors.py).
 *
 * Usage:
 *   const policy = new RetryPolicy({ maxAttempts: 3 });
 *   const result = await policy.run(() => session.endUtterance());
 */
export class RetryPolicy {
  readonly maxAttempts: number;
  readonly baseDelayMs: number;
  readonly maxDelayMs: number;

  constructor(opts: RetryPolicyOptions = {}) {
    this.maxAttempts = opts.maxAttempts ?? 3;
    this.baseDelayMs = opts.baseDelayMs ?? 1000;
    this.maxDelayMs = opts.maxDelayMs ?? 10000;
  }

  /** Exponential backoff: base * 2^attempt, capped at maxDelayMs. */
  delay(attempt: number): number {
    return Math.min(this.baseDelayMs * Math.pow(2, attempt), this.maxDelayMs);
  }

  /**
   * Run `fn` with retry on retryable STTError.
   * Non-retryable errors (including non-STTError) are re-thrown immediately.
   */
  async run<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: STTError | undefined;
    for (let attempt = 0; attempt < this.maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (e) {
        if (!(e instanceof STTError) || !e.isRetryable) {
          throw e;
        }
        lastError = e;
        if (attempt < this.maxAttempts - 1) {
          const d = this.delay(attempt);
          console.warn(
            `[RetryPolicy] retryable error ${e.code}, attempt ${attempt + 1}/${this.maxAttempts} in ${d}ms`,
          );
          await sleep(d);
        }
      }
    }
    // lastError always set if loop exhausted (maxAttempts >= 1)
    throw lastError!;
  }
}

/** Sleep helper. */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Translate WebSocket close codes to typed STTError.
 * Used by Session (session.ts) when WS closes before/after ready.
 *
 * 4401 → unauthorized (not retryable)
 * 4429 → session_limit (retryable)
 * 4408 → idle_timeout (retryable)
 * other → connection_closed (not retryable, code in message)
 */
export function translateCloseCode(code: number, reason: string): STTError {
  if (code === 4401) return new STTError('unauthorized', 'Invalid API key', false);
  if (code === 4429) return new STTError('session_limit', 'Too many concurrent sessions', true);
  if (code === 4408) return new STTError('idle_timeout', 'Session idle timeout', true);
  return new STTError('connection_closed', `WebSocket closed: code=${code}`, false);
}

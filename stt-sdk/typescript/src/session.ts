import { AudioCapture } from './audio.js';
import { STTError, RetryPolicy, translateCloseCode } from './errors.js';
import type { SessionOptions, ServerMessage, FinalResult } from './types.js';

/**
 * Session: long-lived WebSocket на сессию диалога.
 *
 * Two APIs:
 *
 * 1. **Streaming (real-time mic)** — `startUtterance()` → `endUtterance()`.
 *    NO auto-retry. Errors (pool_unavailable, session_limit) are thrown to caller.
 *    Rationale: audio is already captured and sent — on pool_unavailable it's lost.
 *    Caller should show user "service overloaded, repeat the phrase" — this is
 *    a signal for ops to scale the service (see docs/ops/voice-core-scaling.md).
 *
 * 2. **Buffered (prepared audio)** — `sendUtterance(chunks)`.
 *    Auto-retry via RetryPolicy (3 attempts, 1s/2s/4s backoff).
 *    Use for files, batch, tests — where audio can be re-sent.
 */
export class Session {
  private ws: WebSocket | null = null;
  private capture: AudioCapture | null = null;
  private opts: SessionOptions;
  private retryPolicy: RetryPolicy;
  private isRecording = false;
  private waitingFinal = false;

  constructor(opts: SessionOptions) {
    this.opts = opts;
    this.retryPolicy = opts.retryPolicy ?? new RetryPolicy();
  }

  /** Подключиться и дождаться ready. NO auto-retry (caller sees errors).
   *
   * @param timeoutMs — если ready не пришёл за это время — STTError('connection_timeout').
   * Default 10000 (10 sec). Браузерный WebSocket сам таймаутит ~30-60s, но SDK не ждёт так долго.
   */
  async connect(timeoutMs: number = 10000): Promise<void> {
    const url = new URL(this.opts.baseUrl);
    url.searchParams.set('token', this.opts.token);

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url.toString());
      this.ws.binaryType = 'arraybuffer';
      let connected = false;
      const timeoutId = setTimeout(() => {
        if (!connected) {
          try { this.ws?.close(); } catch { /* ignore */ }
          reject(new STTError(
            'connection_timeout',
            `No ready from server within ${timeoutMs}ms`,
            true,
          ));
        }
      }, timeoutMs);

      this.ws.onmessage = (e: MessageEvent) => {
        if (e.data instanceof ArrayBuffer) return;
        const msg: ServerMessage = JSON.parse(e.data);

        if (msg.type === 'ready') {
          connected = true;
          clearTimeout(timeoutId);
          this.opts.onReady?.();
          resolve();
        } else if (msg.type === 'partial') {
          this.opts.onPartial?.(msg.text);
        } else if (msg.type === 'final') {
          this.waitingFinal = false;
          this.opts.onFinal?.(msg);
        } else if (msg.type === 'error') {
          this.waitingFinal = false;
          this.opts.onError?.(new STTError(msg.code, msg.message));
        }
      };

      this.ws.onerror = () => {
        if (!connected) {
          clearTimeout(timeoutId);
          reject(new STTError('ws_error', 'WebSocket error', false));
        }
      };

      this.ws.onclose = (e: CloseEvent) => {
        clearTimeout(timeoutId);
        const err = translateCloseCode(e.code, e.reason);
        if (!connected) {
          reject(err);
        } else if (e.code !== 1000) {
          this.opts.onError?.(err);
        }
      };
    });
  }

  /**
   * Начать реплику: захват микрофона + отправка чанков. NO auto-retry.
   *
   * @param workletPath Optional override. Default: `opts.workletPath` from constructor.
   */
  async startUtterance(workletPath?: string): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new STTError('ws_not_open', 'WebSocket is not open — call connect() first', false);
    }
    if (this.isRecording) throw new Error('Already recording');
    if (this.waitingFinal) throw new Error('Previous utterance not finished — wait for final (Gap 2)');

    this.isRecording = true;
    const wPath = workletPath ?? this.opts.workletPath;
    if (!wPath) {
      this.isRecording = false;
      throw new STTError(
        'worklet_path_missing',
        'SessionOptions.workletPath is required (no hardcoded default since 0.2.0)',
        false,
      );
    }
    try {
      this.capture = new AudioCapture(16000, (pcm16) => {
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(pcm16.buffer);
        }
      });
      await this.capture.start(wPath);
    } catch (e) {
      this.isRecording = false;
      this.capture = null;
      throw e;
    }
  }

  /** Конец реплики: отправить end_of_utterance, дождаться final. NO auto-retry. */
  async endUtterance(): Promise<FinalResult> {
    if (!this.isRecording) throw new Error('Not recording');
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.isRecording = false;
      throw new STTError('ws_not_open', 'WebSocket is not open', false);
    }
    await this.capture?.stop();
    this.capture = null;
    this.isRecording = false;
    this.waitingFinal = true;

    this.ws!.send(JSON.stringify({ type: 'end_of_utterance' }));
    return await this.waitForFinal();
  }

  /**
   * Buffered API: send all PCM16 chunks, return final.
   *
   * Auto-retry via `retryPolicy` (default: 3 attempts, 1s/2s/4s backoff).
   * On each attempt opens a fresh WebSocket (previous one is closed/dead).
   *
   * Use for prepared audio (files, batch, tests). For real-time mic use
   * `startUtterance()` / `endUtterance()` (no auto-retry, caller sees errors).
   */
  async sendUtterance(
    chunks: ArrayBuffer[],
    onPartial?: (text: string) => void,
  ): Promise<FinalResult> {
    return this.retryPolicy.run(async () => {
      // fresh WS for each attempt (previous one might be closed by gateway after error)
      await this.close();
      await this.connect();

      const originalOnPartial = this.opts.onPartial;
      if (onPartial) this.opts.onPartial = onPartial;

      try {
        for (const chunk of chunks) {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            throw new STTError('ws_not_open', 'WebSocket closed during send', true);
          }
          this.ws.send(chunk);
        }
        this.ws!.send(JSON.stringify({ type: 'end_of_utterance' }));
        this.waitingFinal = true;
        return await this.waitForFinal();
      } finally {
        this.opts.onPartial = originalOnPartial;
      }
    });
  }

  /** Wait for final or error from current utterance. 30s timeout. */
  private waitForFinal(): Promise<FinalResult> {
    const originalOnFinal = this.opts.onFinal;
    const originalOnError = this.opts.onError;

    return new Promise<FinalResult>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.opts.onFinal = originalOnFinal;
        this.opts.onError = originalOnError;
        this.waitingFinal = false;
        reject(new STTError('client_timeout', 'No final in 30s', false));
      }, 30000);

      this.opts.onFinal = (r) => {
        clearTimeout(timeout);
        this.opts.onFinal = originalOnFinal;
        this.opts.onError = originalOnError;
        this.waitingFinal = false;
        originalOnFinal?.(r);
        resolve(r);
      };
      this.opts.onError = (e) => {
        clearTimeout(timeout);
        this.opts.onFinal = originalOnFinal;
        this.opts.onError = originalOnError;
        this.waitingFinal = false;
        originalOnError?.(e);
        reject(new STTError(e.code, e.message, e.isRetryable));
      };
    });
  }

  /** Закрыть сессию. */
  async close(): Promise<void> {
    await this.capture?.stop();
    this.capture = null;
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: 'stop' }));
        this.ws.close(1000);
      } catch {
        // ignore — WS might be already dead
      }
    }
    this.ws = null;
  }
}

import type { RetryPolicy } from './errors.js';

export interface ReadyMessage {
  type: 'ready';
  model: string;
  sample_rate: number;
  version: string;
}

export interface PartialResult {
  type: 'partial';
  text: string;
}

export interface FinalResult {
  type: 'final';
  text: string;
  timings?: {
    upstream_connect_ms?: number;
    upstream_first_byte_ms?: number;
    upstream_final_ms?: number;
    audio_bytes_received?: number;
  };
}

export interface ErrorMessage {
  type: 'error';
  code: string;
  message: string;
}

export type ServerMessage = ReadyMessage | PartialResult | FinalResult | ErrorMessage;

export interface SessionOptions {
  baseUrl: string;        // 'ws://localhost:8100/ws/stt' или 'wss://stt.company.local/ws/stt'
  token: string;
  /**
   * URL (or relative path) to the AudioWorklet resample-processor module.
   *
   * REQUIRED (breaking change in 0.2.0). SDK no longer has a hardcoded default
   * (`/static/resample-worklet.js` caused 404 on server deployments).
   *
   * Examples:
   *   - `'./resample-worklet.js'` — file next to your page
   *   - `'/static/resample-worklet.js'` — served by gateway
   *   - `'https://cdn.example.com/worklet.js'` — CDN
   *
   * The file must be a valid AudioWorkletProcessor with `registerProcessor('resample-processor', ...)`.
   * See `examples/resample-worklet.js` for reference.
   */
  workletPath: string;
  /**
   * Retry policy for buffered API (`sendUtterance`).
   * Used ONLY by sendUtterance(), not by startUtterance/endUtterance/connect
   * (streaming path: caller sees errors directly, no auto-retry).
   * Default: new RetryPolicy({ maxAttempts: 3 }).
   */
  retryPolicy?: RetryPolicy;
  onPartial?: (text: string) => void;
  onFinal?: (result: FinalResult) => void;
  onError?: (err: { code: string; message: string; isRetryable: boolean }) => void;
  onReady?: () => void;
}

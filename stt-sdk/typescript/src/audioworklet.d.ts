/**
 * Ambient declarations for AudioWorkletGlobalScope.
 *
 * `resample-worklet.ts` is compiled to `dist/resample-worklet.js` and loaded
 * via `audioWorklet.addModule()` — it runs in AudioWorkletGlobalScope, NOT
 * in Window scope. Standard TypeScript `lib` (ES2022/DOM/WebWorker) does not
 * include these globals. Declarations below fix `tsc` errors TS2304/TS2339.
 *
 * Refs:
 * - https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletGlobalScope
 * - https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor
 */

declare const sampleRate: number;
declare const currentTime: number;
declare const currentFrame: number;

declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: any);
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new (options?: any) => AudioWorkletProcessor,
): void;

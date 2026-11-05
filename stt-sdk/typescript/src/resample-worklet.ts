/// <reference lib="webworker" />

/**
 * ResampleProcessor: AudioWorklet для ресемплинга 48→16 кГц.
 * Сэмплируется в фоновом потоке AudioWorklet.
 */
class ResampleProcessor extends AudioWorkletProcessor {
  private buffer: Float32Array = new Float32Array(2048);
  private bufferLen = 0;
  private ratio: number;

  constructor(options: AudioWorkletNodeOptions) {
    super();
    const targetRate = options.processorOptions?.targetSampleRate || 16000;
    this.ratio = sampleRate / targetRate;
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    // Silence output — Chrome не приостановит worklet
    const output = outputs[0]?.[0];
    if (output) output.fill(0);

    const input = inputs[0]?.[0];
    if (!input) return true;

    // Grow if needed (редко)
    if (this.bufferLen + input.length > this.buffer.length) {
      const newBuf = new Float32Array(this.buffer.length * 2);
      newBuf.set(this.buffer.subarray(0, this.bufferLen));
      this.buffer = newBuf;
    }
    this.buffer.set(input, this.bufferLen);
    this.bufferLen += input.length;

    const outLen = Math.floor(this.bufferLen / this.ratio);
    if (outLen === 0) return true;

    const pcm16 = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * this.ratio;
      const idx = Math.floor(srcIdx);
      const frac = srcIdx - idx;
      const s0 = this.buffer[idx] ?? 0;
      const s1 = this.buffer[idx + 1] ?? s0;
      const sample = s0 + (s1 - s0) * frac;
      pcm16[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
    }

    // Consume integer part, keep remainder
    const consumed = Math.floor(outLen * this.ratio);
    this.buffer.copyWithin(0, consumed, this.bufferLen);
    this.bufferLen -= consumed;

    this.port.postMessage({ pcm16 }, [pcm16.buffer]);
    return true;
  }
}

registerProcessor('resample-processor', ResampleProcessor);

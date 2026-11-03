/**
 * ResampleProcessor: AudioWorklet для ресемплинга 48→16 кГц.
 * Сэмплируется в фоновом потоке AudioWorklet.
 */
class ResampleProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const targetRate = options.processorOptions?.targetSampleRate || 16000;
    this.ratio = sampleRate / targetRate;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;

    this.buffer = [...this.buffer, ...input];

    const outLen = Math.floor(this.buffer.length / this.ratio);
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

    this.buffer = this.buffer.slice(outLen * this.ratio);
    this.port.postMessage({ pcm16 }, [pcm16.buffer]);
    return true;
  }
}

registerProcessor('resample-processor', ResampleProcessor);

/**
 * AudioCapture: getUserMedia + AudioWorklet, ресемплинг 48→16 кГц, PCM16.
 */
import { STTError } from './errors.js';

export class AudioCapture {
  private stream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;

  constructor(
    private sampleRate: number = 16000,
    private onChunk: (pcm16: Int16Array) => void,
  ) {}

  async start(workletPath: string = '/static/resample-worklet.js'): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (e) {
      const name = (e as DOMException)?.name;
      if (name === 'NotAllowedError') {
        throw new STTError(
          'permission_denied',
          'Microphone permission denied — allow access in browser settings',
          false,
        );
      }
      if (name === 'NotFoundError') {
        throw new STTError(
          'microphone_not_found',
          'No microphone device found',
          false,
        );
      }
      throw new STTError(
        'get_user_media_failed',
        `getUserMedia failed: ${(e as Error)?.message ?? e}`,
        false,
      );
    }

    // AudioContext может быть 44100/48000 — ресемплим в AudioWorklet
    this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
    console.debug(`AudioContext: sampleRate=${this.audioContext.sampleRate}, target=${this.sampleRate}`);
    try {
      await this.audioContext.audioWorklet.addModule(workletPath);
    } catch (e) {
      throw new STTError(
        'worklet_load_failed',
        `Failed to load AudioWorklet from ${workletPath}: ${(e as Error)?.message ?? e}`,
        false,
      );
    }
    this.workletNode = new AudioWorkletNode(this.audioContext, 'resample-processor', {
      processorOptions: { targetSampleRate: this.sampleRate },
    });

    this.workletNode.port.onmessage = (e: MessageEvent) => {
      const pcm16 = e.data.pcm16 as Int16Array;
      if (pcm16 && pcm16.length > 0) {
        this.onChunk(pcm16);
      }
    };

    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.source.connect(this.workletNode);

    // Worklet не производит output (только port.postMessage). Без output Chrome
    // может приостановить worklet. Подключаем через gain=0, чтобы worklet вызывался.
    const gain = this.audioContext.createGain();
    gain.gain.value = 0;
    this.workletNode.connect(gain);
    gain.connect(this.audioContext.destination);
  }

  async stop(): Promise<void> {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.audioContext) {
      try {
        await this.audioContext.close();
      } catch (e) {
        console.warn('AudioContext.close error:', e);
      }
      this.audioContext = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
  }
}

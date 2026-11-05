// Node.js example: transcribe WAV file via STT Gateway WebSocket.
//
// TS SDK (@company/stt-client) is browser-only. In Node.js we speak the
// same protocol directly via `ws`. See docs/api/websocket.md.

import WebSocket from 'ws';
import { readFile } from 'node:fs/promises';
import { decode } from 'wav-decoder';

const SAMPLE_RATE = 16000;
const CHUNK_MS = 100;
const CHUNK_SAMPLES = (SAMPLE_RATE * CHUNK_MS) / 1000;

function float32ToPcm16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

async function loadWavPcm16(path) {
  const buf = await readFile(path);
  const audio = await decode(buf);
  if (audio.sampleRate !== SAMPLE_RATE) {
    throw new Error(`WAV must be ${SAMPLE_RATE} Hz, got ${audio.sampleRate}`);
  }
  if (audio.channelData.length !== 1) {
    throw new Error(`WAV must be mono, got ${audio.channelData.length} channels`);
  }
  return float32ToPcm16(audio.channelData[0]);
}

async function transcribe({ audioPath, baseUrl, token }) {
  const pcm16 = await loadWavPcm16(audioPath);

  const url = new URL(baseUrl);
  url.searchParams.set('token', token);

  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url.toString());

    const t0 = Date.now();
    let ttfp = null;
    let finalResult = null;
    let partialCount = 0;

    ws.on('message', async (data) => {
      const text = typeof data === 'string' ? data : Buffer.from(data).toString('utf8');
      const msg = JSON.parse(text);

      if (msg.type === 'ready') {
        for (let off = 0; off < pcm16.length; off += CHUNK_SAMPLES) {
          const end = Math.min(off + CHUNK_SAMPLES, pcm16.length);
          const chunk = pcm16.subarray(off, end);
          ws.send(Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength));
          await new Promise((r) => setTimeout(r, CHUNK_MS));
        }
        ws.send(JSON.stringify({ type: 'end_of_utterance' }));
      } else if (msg.type === 'partial') {
        if (ttfp === null) ttfp = Date.now() - t0;
        partialCount++;
        console.error(`[partial @ ${Date.now() - t0}ms] ${msg.text}`);
      } else if (msg.type === 'final') {
        finalResult = {
          text: msg.text,
          timings: msg.timings,
          client_ttfp_ms: ttfp,
          client_ttft_ms: Date.now() - t0,
          partials_count: partialCount,
        };
        ws.send(JSON.stringify({ type: 'stop' }));
        ws.close(1000);
        resolve(finalResult);
      } else if (msg.type === 'error') {
        ws.close();
        reject(new Error(`STT error: ${msg.code} — ${msg.message}`));
      }
    });

    ws.on('error', reject);
    ws.on('close', (code, reason) => {
      if (finalResult === null && code !== 1000) {
        reject(new Error(`WS closed code=${code} reason=${reason}`));
      }
    });
  });
}

const [, , audioPath] = process.argv;
const baseUrl = process.env.STT_URL || 'ws://localhost:8100/ws/stt';
const token = process.env.STT_API_KEY;

if (!audioPath) {
  console.error('Usage: STT_API_KEY=... node transcribe.mjs <path/to/audio.wav>');
  process.exit(1);
}
if (!token) {
  console.error('ERROR: set STT_API_KEY env variable');
  process.exit(1);
}

transcribe({ audioPath, baseUrl, token })
  .then((r) => {
    console.log('final:', r.text);
    console.log('timings:', JSON.stringify(r.timings));
    console.log(`client_ttfp_ms: ${r.client_ttfp_ms}`);
    console.log(`client_ttft_ms: ${r.client_ttft_ms}`);
    console.log(`partials_count: ${r.partials_count}`);
  })
  .catch((err) => {
    console.error('error:', err.message);
    process.exit(1);
  });

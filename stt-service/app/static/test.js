const WS_URL = `ws://${location.hostname}:8100/ws/stt`;
const SAMPLE_RATE = 16000;
const CHUNK_MS = 100;

let ws = null;
let mediaStream = null;
let audioContext = null;
let processor = null;
let source = null;
let startTime = null;
let ttfpRecorded = false;

// Запись аудио (временный буфер для текущей сессии)
let recordedChunks = [];
let audioBytes = 0;

// История записей
let recordings = [];
let recordingIdCounter = 0;

// Статистика
window._ttfp = null;
window._ttft = null;
window._audioBytes = 0;

const $ = id => document.getElementById(id);

// Build WAV from PCM16 chunks
function buildWav(chunks, sampleRate) {
    const totalLen = chunks.reduce((s, c) => s + c.length, 0);
    const buffer = new ArrayBuffer(44 + totalLen * 2);
    const view = new DataView(buffer);
    const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
    writeStr(0, 'RIFF'); view.setUint32(4, 36 + totalLen * 2, true);
    writeStr(8, 'WAVE'); writeStr(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); writeStr(36, 'data');
    view.setUint32(40, totalLen * 2, true);
    let off = 44;
    for (const c of chunks) { for (let i = 0; i < c.length; i++) { view.setInt16(off, c[i], true); off += 2; } }
    return new Blob([buffer], { type: 'audio/wav' });
}

// Добавить запись в историю
function addRecording(wavBlob, finalText, timings, durationSec, audioBytes) {
    const id = ++recordingIdCounter;
    const url = URL.createObjectURL(wavBlob);
    const rec = {
        id, url, blob: wavBlob, finalText, timings, durationSec, audioBytes,
        timestamp: new Date(),
        sizeKB: (wavBlob.size / 1024).toFixed(1),
    };
    recordings.push(rec);
    renderRecordings();
}

// Рендер истории записей
function renderRecordings() {
    const list = $('recordings-list');
    list.innerHTML = '';
    $('recordings-count').textContent = recordings.length;
    $('recordings-empty').style.display = recordings.length === 0 ? 'block' : 'none';
    $('delete-selected').disabled = recordings.length === 0;
    $('clear-all').disabled = recordings.length === 0;

    for (const rec of recordings) {
        const li = document.createElement('li');
        li.style.cssText = 'border:1px solid #ddd; border-radius:4px; padding:0.5em; margin:0.3em 0; display:flex; align-items:flex-start; gap:0.5em; background:#fafafa;';
        li.dataset.id = rec.id;

        // Checkbox
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.className = 'rec-checkbox';
        cb.dataset.id = rec.id;
        cb.addEventListener('change', updateDeleteButton);

        // Play button
        const play = document.createElement('button');
        play.textContent = '▶';
        play.title = 'Прослушать';
        play.style.cssText = 'font-size:1.2em; padding:0.2em 0.6em;';
        play.onclick = () => {
            const audio = new Audio(rec.url);
            audio.play();
        };

        // Info block
        const info = document.createElement('div');
        info.style.cssText = 'flex:1; font-size:0.9em;';
        const ts = rec.timestamp.toLocaleTimeString('ru-RU');
        const ttfp = rec.timings && rec.timings.upstream_first_byte_ms != null
            ? `${rec.timings.upstream_first_byte_ms.toFixed(0)} ms`
            : '—';
        const ttft = rec.timings && rec.timings.upstream_final_ms != null
            ? `${rec.timings.upstream_final_ms.toFixed(0)} ms`
            : '—';
        info.innerHTML = `
            <div><strong>${ts}</strong> · <em>${rec.durationSec.toFixed(2)} с, ${rec.sizeKB} KB</em></div>
            <div>Модель: <strong style="color:#2a2;">${rec.finalText || '(пусто)'}</strong></div>
            <div style="color:#666; font-size:0.85em;">Сервер: TTFP ${ttfp} · TTFT ${ttft}</div>
        `;

        // Delete button
        const del = document.createElement('button');
        del.textContent = '✕';
        del.title = 'Удалить';
        del.style.cssText = 'color:#c00; padding:0.2em 0.6em;';
        del.onclick = () => deleteRecording(rec.id);

        li.appendChild(cb);
        li.appendChild(play);
        li.appendChild(info);
        li.appendChild(del);
        list.appendChild(li);
    }
}

// Удалить одну запись
function deleteRecording(id) {
    const idx = recordings.findIndex(r => r.id === id);
    if (idx === -1) return;
    URL.revokeObjectURL(recordings[idx].url);
    recordings.splice(idx, 1);
    renderRecordings();
}

// Удалить выбранные
function deleteSelected() {
    const checked = Array.from(document.querySelectorAll('.rec-checkbox:checked'))
        .map(cb => parseInt(cb.dataset.id, 10));
    if (checked.length === 0) return;
    if (!confirm(`Удалить ${checked.length} записей?`)) return;
    for (const id of checked) {
        const idx = recordings.findIndex(r => r.id === id);
        if (idx !== -1) {
            URL.revokeObjectURL(recordings[idx].url);
            recordings.splice(idx, 1);
        }
    }
    renderRecordings();
}

// Очистить всё
function clearAll() {
    if (recordings.length === 0) return;
    if (!confirm(`Удалить все ${recordings.length} записей?`)) return;
    for (const rec of recordings) URL.revokeObjectURL(rec.url);
    recordings = [];
    renderRecordings();
}

// Обновить состояние кнопки delete-selected
function updateDeleteButton() {
    const any = document.querySelector('.rec-checkbox:checked') !== null;
    $('delete-selected').disabled = !any;
}

async function start() {
    console.log('[start] нажата кнопка Говорить');
    const token = $('token').value;
    startTime = performance.now();
    ttfpRecorded = false;
    window._ttfp = null;
    window._ttft = null;
    audioBytes = 0;
    recordedChunks = [];
    $('partial').textContent = '—';
    $('final').textContent = '—';
    $('ttfp').textContent = '—';
    $('stat-duration').textContent = '—';
    $('stat-bytes').textContent = '—';
    $('stat-ttfp').textContent = '—';
    $('stat-ttft').textContent = '—';
    $('stat-upstream-connect').textContent = '—';
    $('stat-upstream-first-byte').textContent = '—';
    $('stat-upstream-final').textContent = '—';
    $('level-bar-inner').style.width = '0%';
    $('level-text').textContent = '0';
    $('status').textContent = 'Подключение...';
    console.log('[start] WS URL:', `${WS_URL}?token=${token}`);

    ws = new WebSocket(`${WS_URL}?token=${token}`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = async () => {
        console.log('[ws.onopen] WebSocket подключён');
        $('status').textContent = 'Микрофон...';
        try {
            await startAudio();
            console.log('[startAudio] микрофон запущен');
            $('status').textContent = 'Слушаю...';
            $('stop').disabled = false;
            $('start').disabled = true;
        } catch (e) {
            console.error('[startAudio] ошибка:', e);
            $('status').textContent = `Ошибка микрофона: ${e.message}`;
        }
    };

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        console.log('[ws.onmessage]', msg.type, msg.text || '');
        if (msg.type === 'ready') return;
        if (msg.type === 'partial') {
            if (!ttfpRecorded && startTime) {
                window._ttfp = performance.now() - startTime;
                $('ttfp').textContent = `${window._ttfp.toFixed(0)} ms`;
                $('stat-ttfp').textContent = `${window._ttfp.toFixed(0)} ms`;
                ttfpRecorded = true;
            }
            $('partial').textContent = msg.text || '';
        } else if (msg.type === 'final') {
            window._ttft = performance.now() - startTime;
            $('final').textContent = msg.text || '(пусто)';
            $('partial').textContent = '—';
            $('stat-ttft').textContent = `${window._ttft.toFixed(0)} ms`;
            // Статистика длительности и размера
            const durationSec = audioBytes / 32000;  // 16000 samples/s * 2 bytes
            $('stat-duration').textContent = `${durationSec.toFixed(2)} с`;
            $('stat-bytes').textContent = `${(audioBytes / 1024).toFixed(1)} KB`;
            // Серверные тайминги
            if (msg.timings) {
                const t = msg.timings;
                $('stat-upstream-connect').textContent = t.upstream_connect_ms != null ? `${t.upstream_connect_ms.toFixed(0)} ms` : '—';
                $('stat-upstream-first-byte').textContent = t.upstream_first_byte_ms != null ? `${t.upstream_first_byte_ms.toFixed(0)} ms` : '—';
                $('stat-upstream-final').textContent = t.upstream_final_ms != null ? `${t.upstream_final_ms.toFixed(0)} ms` : '—';
            }
            // Собрать WAV и добавить в историю
            if (recordedChunks.length > 0) {
                const blob = buildWav(recordedChunks, 16000);
                addRecording(blob, msg.text || '', msg.timings || null, durationSec, audioBytes);
                recordedChunks = [];  // сброс для следующей записи
            }
            // Отправить stop после final и закрыть
            if (window._stopTimeout) { clearTimeout(window._stopTimeout); window._stopTimeout = null; }
            if (ws && ws.readyState === WebSocket.OPEN) {
                console.log('[final] отправка stop после final');
                ws.send(JSON.stringify({ type: 'stop' }));
                setTimeout(() => { if (ws) ws.close(); }, 500);
            }
        }
    };

    ws.onerror = (e) => {
        console.error('[ws.onerror]', e);
        $('status').textContent = `Ошибка WebSocket: ${e.message || 'см. Console'}`;
    };
    ws.onclose = (e) => {
        console.log('[ws.onclose]', e.code, e.reason);
        $('status').textContent = `Закрыто: ${e.code} ${e.reason || ''}`;
        stopAudio(); $('start').disabled = false; $('stop').disabled = true;
    };
}

async function startAudio() {
    console.log('[startAudio] запрос getUserMedia');
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    console.log('[startAudio] getUserMedia OK, tracks:', mediaStream.getTracks().length);
    audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    console.log('[startAudio] AudioContext sampleRate:', audioContext.sampleRate);
    source = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    let firstChunk = true;
    processor.onaudioprocess = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
            pcm16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
        }
        // Индикатор уровня микрофона (RMS)
        let sum = 0;
        for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
        const rms = Math.sqrt(sum / input.length);
        const level = Math.min(100, Math.round(rms * 1000));
        $('level-bar-inner').style.width = `${level}%`;
        $('level-text').textContent = level;
        // Запись чанка
        recordedChunks.push(new Int16Array(pcm16));
        audioBytes += pcm16.buffer.byteLength;
        window._audioBytes = audioBytes;
        if (firstChunk) {
            console.log('[onaudioprocess] первый чанк, bytes:', pcm16.buffer.byteLength, 'sampleRate:', audioContext.sampleRate);
            firstChunk = false;
        }
        ws.send(pcm16.buffer);
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
}

function stopAudio() {
    if (processor) { processor.disconnect(); processor = null; }
    if (source) { source.disconnect(); source = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
}

function stop() {
    console.log('[stop] нажата кнопка Стоп');
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('[stop] отправка end_of_utterance');
        ws.send(JSON.stringify({ type: 'end_of_utterance' }));
        // Ждём final, затем отправим stop (не закрывать ws через таймер)
        const stopTimeout = setTimeout(() => {
            console.log('[stop] TIMEOUT 20s, принудительное закрытие');
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'stop' }));
                ws.close();
            }
        }, 20000);
        window._stopTimeout = stopTimeout;
    }
    stopAudio();
    $('start').disabled = false;
    $('stop').disabled = true;
}

$('start').addEventListener('click', start);
$('stop').addEventListener('click', stop);
$('delete-selected').addEventListener('click', deleteSelected);
$('clear-all').addEventListener('click', clearAll);
renderRecordings();  // начальный рендер (пустой список)

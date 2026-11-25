import * as ort from 'onnxruntime-web';
import { MAX_MIDI, MIN_MIDI } from './pianoTheory';
import {
  downsampleMono,
  estimateKeyFromChroma,
  isPitchInKey,
  PianoNoteEvent,
  TrackRole,
  TranscribedPiece,
} from './pitchDsp';
import { estimateBpmFromNoteOnsets } from './basicPitchTranscriber';

export const BYTEDANCE_MODEL_URL =
  'https://huggingface.co/stelee/piano-transcription-onnx/resolve/main/piano_crnn.onnx';
const BYTEDANCE_CACHE_NAME = 'aurelia-bytedance-piano-v1';

export const BYTEDANCE_SAMPLE_RATE = 16000;
export const BYTEDANCE_FRAMES_PER_SECOND = 100;
export const BYTEDANCE_SEGMENT_SAMPLES = 160000; // 10.0 s @ 16 kHz
export const BYTEDANCE_HOP_SAMPLES = 128000; // 8.0 s hop -> 1.0 s context margin on each side
export const BYTEDANCE_SEGMENT_FRAMES = 1001; // 10.0 s * 100 fps + 1
const NUM_PIANO_KEYS = 88;

let cachedSessionPromise: Promise<ort.InferenceSession> | null = null;
let ortConfigured = false;

function isSoftwareWebGpuOrHeadless(): boolean {
  if (typeof navigator === 'undefined') return true;
  if (!('gpu' in navigator)) return true;
  if (typeof document !== 'undefined') {
    try {
      const c = document.createElement('canvas');
      const gl = c.getContext('webgl2') || c.getContext('webgl');
      if (gl) {
        const ext = gl.getExtension('WEBGL_debug_renderer_info');
        if (ext) {
          const renderer = String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '');
          if (/swiftshader|llvmpipe|software|mesa/i.test(renderer)) return true;
        }
      }
    } catch {
      return true;
    }
  }
  return false;
}

function configureOrtEnv(): void {
  if (ortConfigured) return;
  ortConfigured = true;
  if (typeof window !== 'undefined') {
    ort.env.wasm.wasmPaths = new URL('./ort/', window.location.href).href;
    // GitHub Pages & standard static hosts don't send COOP/COEP headers for SharedArrayBuffer
    const isolated = Boolean((window as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated);
    ort.env.wasm.numThreads = isolated
      ? Math.min(4, Math.max(1, navigator.hardwareConcurrency || 2))
      : 1;
    ort.env.wasm.simd = true;
  } else {
    ort.env.wasm.numThreads = 1;
  }
}

/**
 * Downloads (with streaming MB progress) and permanently caches the MIT-licensed
 * ByteDance High-Resolution Piano CRNN ONNX model in the browser's Cache Storage.
 */
async function fetchCachedByteDanceModel(
  onProgress?: (progress: number, stage: string) => void,
): Promise<Uint8Array> {
  // 1. Check local persistent browser Cache Storage first (instant on 2nd+ uploads)
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(BYTEDANCE_CACHE_NAME);
      const cached = await cache.match(BYTEDANCE_MODEL_URL);
      if (cached && cached.ok) {
        onProgress?.(0.14, 'Cargando IA ByteDance Piano desde caché local...');
        const buf = await cached.arrayBuffer();
        if (buf.byteLength > 100_000_000) {
          return new Uint8Array(buf);
        }
      }
    } catch {
      // Cache Storage unavailable (e.g., private browsing) — continue with direct fetch
    }
  }

  onProgress?.(0.06, 'Descargando modelo IA ByteDance Piano (solo la primera vez, 147 MB)...');
  const response = await fetch(BYTEDANCE_MODEL_URL, { mode: 'cors' });
  if (!response.ok) {
    throw new Error(`No se pudo descargar el modelo IA de piano (HTTP ${response.status}).`);
  }

  const contentLength = Number(response.headers.get('content-length') || 154_206_740);
  const reader = response.body?.getReader();

  let bytes: Uint8Array;
  if (!reader) {
    bytes = new Uint8Array(await response.arrayBuffer());
  } else {
    const chunks: Uint8Array[] = [];
    let received = 0;
    let lastReport = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        received += value.byteLength;
        const now = performance.now();
        if (now - lastReport > 140) {
          lastReport = now;
          const frac = Math.min(1, received / Math.max(1, contentLength));
          const mbDone = (received / (1024 * 1024)).toFixed(0);
          const mbTotal = (contentLength / (1024 * 1024)).toFixed(0);
          onProgress?.(
            0.06 + frac * 0.24,
            `Descargando IA ByteDance Piano (${mbDone} / ${mbTotal} MB — se guarda en caché)...`,
          );
        }
      }
    }

    bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  }

  // Persist into Cache Storage so future uploads skip the download completely
  if (typeof caches !== 'undefined' && bytes.byteLength > 100_000_000) {
    try {
      const cache = await caches.open(BYTEDANCE_CACHE_NAME);
      const copy = new Uint8Array(bytes.byteLength);
      copy.set(bytes);
      await cache.put(
        BYTEDANCE_MODEL_URL,
        new Response(copy.buffer, {
          status: 200,
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(bytes.byteLength),
          },
        }),
      );
    } catch {
      // Quota or storage restriction — still proceed cleanly in memory
    }
  }

  return bytes;
}

async function getByteDanceSession(
  onProgress?: (progress: number, stage: string) => void,
): Promise<ort.InferenceSession> {
  if (!cachedSessionPromise) {
    cachedSessionPromise = (async () => {
      configureOrtEnv();
      const modelBytes = await fetchCachedByteDanceModel(onProgress);
      onProgress?.(0.32, 'Inicializando red neuronal ByteDance Piano...');

      const preferWebGpu = !isSoftwareWebGpuOrHeadless();
      // IMPORTANT: graphOptimizationLevel MUST be 'disabled' for piano_crnn.onnx
      // (ORT MatMul+BN fusion mis-infers a rank-3 Gemm in this exported graph).
      if (preferWebGpu) {
        try {
          return await ort.InferenceSession.create(modelBytes, {
            executionProviders: ['webgpu', 'wasm'],
            graphOptimizationLevel: 'disabled',
          });
        } catch {
          // Fall back cleanly to SIMD WASM below
        }
      }

      return ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'disabled',
      });
    })().catch((err) => {
      cachedSessionPromise = null;
      throw err;
    });
  }
  return cachedSessionPromise;
}

/**
 * Resamples AudioBuffer cleanly to mono 16,000 Hz PCM with gentle DC removal
 * and peak leveling for quiet recordings (no destructive HPSS or compression).
 */
async function resampleBufferToMono16k(
  audioBuffer: AudioBuffer,
  maxDurationSec = 240,
): Promise<Float32Array> {
  const dur = Math.min(audioBuffer.duration, maxDurationSec);
  const targetLen = Math.max(1, Math.ceil(dur * BYTEDANCE_SAMPLE_RATE));

  let mono: Float32Array | null = null;
  const OfflineCtx =
    typeof window !== 'undefined'
      ? window.OfflineAudioContext ||
        (window as unknown as { webkitOfflineAudioContext?: typeof OfflineAudioContext })
          .webkitOfflineAudioContext
      : undefined;

  if (OfflineCtx) {
    try {
      const offline = new OfflineCtx(1, targetLen, BYTEDANCE_SAMPLE_RATE);
      const src = offline.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(offline.destination);
      src.start(0, 0, dur);
      const rendered = await offline.startRendering();
      mono = new Float32Array(rendered.getChannelData(0));
    } catch {
      mono = null;
    }
  }

  if (!mono) {
    const maxSamples = Math.min(audioBuffer.length, Math.floor(dur * audioBuffer.sampleRate));
    const left = audioBuffer.getChannelData(0).subarray(0, maxSamples);
    const right =
      audioBuffer.numberOfChannels > 1
        ? audioBuffer.getChannelData(1).subarray(0, maxSamples)
        : null;
    const mixed = new Float32Array(maxSamples);
    if (right) {
      for (let i = 0; i < maxSamples; i++) mixed[i] = 0.5 * (left[i] + right[i]);
    } else {
      mixed.set(left);
    }
    mono = downsampleMono(mixed, audioBuffer.sampleRate, BYTEDANCE_SAMPLE_RATE);
  }

  // Remove DC offset and gently bring very quiet phone recordings up toward -2 dB peak
  let sum = 0;
  for (let i = 0; i < mono.length; i++) sum += mono[i];
  const mean = sum / Math.max(1, mono.length);
  let peak = 0;
  for (let i = 0; i < mono.length; i++) {
    const v = mono[i] - mean;
    mono[i] = v;
    const a = Math.abs(v);
    if (a > peak) peak = a;
  }
  if (peak > 1e-4 && peak < 0.35) {
    const gain = Math.min(4.0, 0.85 / peak);
    for (let i = 0; i < mono.length; i++) mono[i] *= gain;
  }

  return mono;
}

export interface ByteDanceRawOutputs {
  numFrames: number;
  /** Flattened [numFrames * 88] onset regression probabilities (0..1) */
  regOnset: Float32Array;
  /** Flattened [numFrames * 88] offset regression probabilities (0..1) */
  regOffset: Float32Array;
  /** Flattened [numFrames * 88] active pitch frame probabilities (0..1) */
  frame: Float32Array;
  /** Flattened [numFrames * 88] neural MIDI velocities (0..1) */
  velocity: Float32Array;
  /** Optional [numFrames] sustain pedal active probabilities (0..1) */
  pedalFrame?: Float32Array;
  /** Optional [numFrames] sustain pedal release regression peaks (0..1) */
  pedalOffset?: Float32Array;
}

/**
 * Pure, unit-testable implementation of Qiuqiang Kong et al. (2020) ByteDance
 * High-Resolution Piano RegressionPostProcessor (onset + offset + frame + velocity + pedal).
 * Achieves sub-10ms onset accuracy via parabolic interpolation around regression peaks
 * and honors real acoustic sustain pedal extensions.
 */
export function decodeByteDanceOutputsToPiece(
  outputs: ByteDanceRawOutputs,
  durationSec: number,
  title: string,
  thresholds?: {
    onsetThresh?: number;
    offsetThresh?: number;
    frameThresh?: number;
    pedalThresh?: number;
  },
): TranscribedPiece {
  const T = outputs.numFrames;
  const onsetThresh = thresholds?.onsetThresh ?? 0.3;
  const offsetThresh = thresholds?.offsetThresh ?? 0.3;
  const frameThresh = thresholds?.frameThresh ?? 0.12;
  const pedalThresh = thresholds?.pedalThresh ?? 0.5;
  const fps = BYTEDANCE_FRAMES_PER_SECOND;

  interface DecodedNote {
    midi: number;
    startTime: number;
    duration: number;
    velocity: number;
    confidence: number;
  }

  const decoded: DecodedNote[] = [];

  for (let k = 0; k < NUM_PIANO_KEYS; k++) {
    // 1. Find local regression peaks for key `k`
    const onsets: Array<{ frame: number; exactTime: number; prob: number; vel: number }> = [];
    let lastOnsetFrame = -10;

    for (let t = 1; t < T - 1; t++) {
      const idx = t * NUM_PIANO_KEYS + k;
      const cur = outputs.regOnset[idx];
      if (cur < onsetThresh) continue;
      const prev = outputs.regOnset[(t - 1) * NUM_PIANO_KEYS + k];
      const next = outputs.regOnset[(t + 1) * NUM_PIANO_KEYS + k];
      if (cur >= prev && cur > next && t - lastOnsetFrame >= 4) {
        // Parabolic sub-frame shift (Kong et al. regression refinement)
        const denom = 2 * (2 * cur - prev - next);
        const shift = denom > 1e-6 ? Math.max(-0.5, Math.min(0.5, (next - prev) / denom)) : 0;
        const exactTime = Math.max(0, (t + shift) / fps);
        const rawVel = outputs.velocity[idx];
        onsets.push({ frame: t, exactTime, prob: cur, vel: rawVel });
        lastOnsetFrame = t;
      }
    }

    // 2. For each onset, find exact offset via regOffset peak, frame drop, sustain pedal, or next onset
    for (let oi = 0; oi < onsets.length; oi++) {
      const on = onsets[oi];
      const nextOnsetFrame = oi + 1 < onsets.length ? onsets[oi + 1].frame : T - 1;
      const maxEndFrame = Math.min(nextOnsetFrame, on.frame + Math.round(6.5 * fps));

      let endFrame = on.frame + 3;
      let exactEndTime = Math.min(durationSec, endFrame / fps);
      let peakFrameProb = outputs.frame[on.frame * NUM_PIANO_KEYS + k];
      let lowFrameStreak = 0;
      let heldByPedal = false;

      for (let t = on.frame + 1; t < maxEndFrame; t++) {
        const idx = t * NUM_PIANO_KEYS + k;
        const fProb = outputs.frame[idx];
        if (fProb > peakFrameProb) peakFrameProb = fProb;

        const pedalDown = outputs.pedalFrame ? outputs.pedalFrame[t] >= pedalThresh : false;
        const pedalReleased =
          heldByPedal &&
          outputs.pedalFrame !== undefined &&
          (outputs.pedalFrame[t] < 0.35 ||
            (outputs.pedalOffset !== undefined && outputs.pedalOffset[t] >= 0.3));

        if (pedalReleased && fProb < frameThresh) {
          endFrame = t;
          exactEndTime = t / fps;
          break;
        }

        // Check regOffset peak when sustain pedal isn't holding the string open
        if (!pedalDown && t > on.frame + 2 && t < T - 1) {
          const offCur = outputs.regOffset[idx];
          if (offCur >= offsetThresh) {
            const offPrev = outputs.regOffset[(t - 1) * NUM_PIANO_KEYS + k];
            const offNext = outputs.regOffset[(t + 1) * NUM_PIANO_KEYS + k];
            if (offCur >= offPrev && offCur > offNext) {
              const denom = 2 * (2 * offCur - offPrev - offNext);
              const shift =
                denom > 1e-6 ? Math.max(-0.5, Math.min(0.5, (offNext - offPrev) / denom)) : 0;
              endFrame = t;
              exactEndTime = Math.max(on.exactTime + 0.04, (t + shift) / fps);
              break;
            }
          }
        }

        if (fProb < frameThresh) {
          lowFrameStreak++;
          if (lowFrameStreak >= 2) {
            if (pedalDown) {
              heldByPedal = true;
            } else {
              endFrame = Math.max(on.frame + 3, t - 1);
              exactEndTime = endFrame / fps;
              break;
            }
          }
        } else {
          lowFrameStreak = 0;
        }

        endFrame = t;
        exactEndTime = t / fps;
      }

      // Trim cleanly 10ms before a re-strike of the exact same key
      if (oi + 1 < onsets.length) {
        const reStrikeLimit = Math.max(on.exactTime + 0.035, onsets[oi + 1].exactTime - 0.01);
        if (exactEndTime > reStrikeLimit) exactEndTime = reStrikeLimit;
      }

      const dur = exactEndTime - on.exactTime;
      if (dur < 0.035) continue;

      // Require minimum frame support so isolated sub-frame noise spikes are rejected
      if (peakFrameProb < 0.08 && on.prob < 0.55) continue;

      const midi = k + MIN_MIDI;
      if (midi < MIN_MIDI || midi > MAX_MIDI) continue;

      // Calibrated ByteDance MIDI velocity (typical model range ~0.10..0.75 -> 0.25..0.98)
      const normVel = Math.max(0.24, Math.min(0.98, 0.18 + on.vel * 1.15));
      const confidence = Math.max(
        0.18,
        Math.min(0.99, on.prob * 0.65 + peakFrameProb * 0.35),
      );

      decoded.push({
        midi,
        startTime: on.exactTime,
        duration: Math.min(6.5, dur),
        velocity: normVel,
        confidence,
      });
    }
  }

  decoded.sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);

  // Estimate global key signature & BPM for HUD metadata and MIDI file header
  const chroma = new Float64Array(12);
  for (const n of decoded) {
    chroma[n.midi % 12] += n.velocity * Math.min(2.0, n.duration);
  }
  const detectedKey = estimateKeyFromChroma(chroma);
  const bpm = estimateBpmFromNoteOnsets(
    decoded.map((n) => ({ startTimeSeconds: n.startTime })),
  );

  // Identify right-hand top melody voice (highest prominent note >= MIDI 58 per 65ms window)
  const melodyIdx = new Set<number>();
  const WIN_SEC = 0.065;
  let i = 0;
  while (i < decoded.length) {
    const winStart = decoded[i].startTime;
    let j = i;
    let bestMel = -1;
    let bestScore = -Infinity;
    while (j < decoded.length && decoded[j].startTime - winStart <= WIN_SEC) {
      const n = decoded[j];
      if (n.midi >= 58) {
        const score = n.velocity * 1.1 + (n.midi - 58) * 0.025;
        if (score > bestScore) {
          bestScore = score;
          bestMel = j;
        }
      }
      j++;
    }
    if (bestMel >= 0) melodyIdx.add(bestMel);
    i = j;
  }

  // Assign every decoded piano note to one of Aurelia's 4 playable tracks (zero dropped voices!)
  const countsByTrack: Record<TrackRole, number> = {
    vocal: 0,
    chords: 0,
    bass: 0,
    lead: 0,
  };
  const events: PianoNoteEvent[] = [];
  let maxEnd = durationSec;

  for (let idx = 0; idx < decoded.length; idx++) {
    const n = decoded[idx];
    const inKey = isPitchInKey(n.midi, detectedKey.root, detectedKey.mode);
    let role: TrackRole;
    if (n.midi <= 47) {
      role = 'bass';
    } else if (melodyIdx.has(idx)) {
      role = 'vocal';
    } else if (n.midi >= 77) {
      role = 'lead';
    } else {
      role = 'chords';
    }

    const startTime = Number(n.startTime.toFixed(3));
    const duration = Number(n.duration.toFixed(3));
    if (startTime + duration > maxEnd) maxEnd = startTime + duration;

    countsByTrack[role]++;
    events.push({
      id: `bd-piano-${idx + 1}`,
      midi: n.midi,
      startTime,
      duration,
      velocity: Number(n.velocity.toFixed(2)),
      track: role,
      confidence: Number(n.confidence.toFixed(2)),
      inKey,
    });
  }

  return {
    id: `upload-${Date.now()}`,
    title,
    subtitle: 'IA ByteDance High-Res Piano CRNN (96.7% MAESTRO · 88 teclas + pedal)',
    source: 'upload',
    transcriptionMode: 'piano',
    duration: Number(Math.max(1, maxEnd).toFixed(2)),
    bpm,
    keySignature: detectedKey.label,
    keyRoot: detectedKey.root,
    keyMode: detectedKey.mode,
    notes: events,
    countsByTrack,
  };
}

/**
 * Full 100% client-side ByteDance High-Resolution Piano Transcription pipeline
 * running over ONNX Runtime Web (WebGPU / SIMD WASM), cached in browser Cache Storage.
 */
export async function transcribeAudioBufferWithByteDancePiano(
  audioBuffer: AudioBuffer,
  title: string,
  onProgress?: (progress: number, stage: string) => void,
): Promise<TranscribedPiece> {
  onProgress?.(0.04, 'Remuestreando grabación de piano a 16 kHz...');
  const mono16k = await resampleBufferToMono16k(audioBuffer, 240);
  const actualDurationSec = mono16k.length / BYTEDANCE_SAMPLE_RATE;

  const session = await getByteDanceSession(onProgress);

  const totalFrames = Math.max(
    1,
    Math.ceil(actualDurationSec * BYTEDANCE_FRAMES_PER_SECOND) + 1,
  );
  const fullRegOnset = new Float32Array(totalFrames * NUM_PIANO_KEYS);
  const fullRegOffset = new Float32Array(totalFrames * NUM_PIANO_KEYS);
  const fullFrame = new Float32Array(totalFrames * NUM_PIANO_KEYS);
  const fullVelocity = new Float32Array(totalFrames * NUM_PIANO_KEYS);
  const fullPedalFrame = new Float32Array(totalFrames);
  const fullPedalOffset = new Float32Array(totalFrames);

  // Build overlapping 10-second segments (8s hop -> 1s left/right acoustic context margin)
  const segStarts: number[] = [];
  if (mono16k.length <= BYTEDANCE_SEGMENT_SAMPLES) {
    segStarts.push(0);
  } else {
    for (let s = 0; s < mono16k.length; s += BYTEDANCE_HOP_SAMPLES) {
      segStarts.push(s);
      if (s + BYTEDANCE_SEGMENT_SAMPLES >= mono16k.length) break;
    }
  }

  const inputName = session.inputNames[0] || 'audio';
  const segBuf = new Float32Array(BYTEDANCE_SEGMENT_SAMPLES);

  for (let segIdx = 0; segIdx < segStarts.length; segIdx++) {
    const startSample = segStarts[segIdx];
    segBuf.fill(0);
    const copyLen = Math.min(BYTEDANCE_SEGMENT_SAMPLES, mono16k.length - startSample);
    segBuf.set(mono16k.subarray(startSample, startSample + copyLen), 0);

    onProgress?.(
      0.34 + (segIdx / segStarts.length) * 0.56,
      `Transcribiendo con IA ByteDance Piano (bloque ${segIdx + 1}/${segStarts.length})...`,
    );
    // Yield a frame so React paints the updated progress bar before heavy inference
    await new Promise((resolve) => setTimeout(resolve, 16));

    const inputTensor = new ort.Tensor('float32', segBuf, [1, BYTEDANCE_SEGMENT_SAMPLES]);
    const results = await session.run({ [inputName]: inputTensor });

    const regOnsetData = results.reg_onset.data as Float32Array;
    const regOffsetData = results.reg_offset.data as Float32Array;
    const frameData = results.frame.data as Float32Array;
    const velocityData = results.velocity.data as Float32Array;
    const pedalFrameData = results.pedal_frame?.data as Float32Array | undefined;
    const pedalOffsetData = results.pedal_offset?.data as Float32Array | undefined;

    const segStartFrame = Math.round(startSample / (BYTEDANCE_SAMPLE_RATE / BYTEDANCE_FRAMES_PER_SECOND));
    // Keep non-edge center region [100..900] on interior segments to eliminate boundary cuts
    const localStart = segIdx === 0 ? 0 : 100;
    const localEnd =
      segIdx === segStarts.length - 1
        ? Math.min(BYTEDANCE_SEGMENT_FRAMES, totalFrames - segStartFrame)
        : Math.min(900, BYTEDANCE_SEGMENT_FRAMES, totalFrames - segStartFrame);

    for (let lf = localStart; lf < localEnd; lf++) {
      const gf = segStartFrame + lf;
      if (gf < 0 || gf >= totalFrames) continue;
      const srcBase = lf * NUM_PIANO_KEYS;
      const dstBase = gf * NUM_PIANO_KEYS;
      fullRegOnset.set(regOnsetData.subarray(srcBase, srcBase + NUM_PIANO_KEYS), dstBase);
      fullRegOffset.set(regOffsetData.subarray(srcBase, srcBase + NUM_PIANO_KEYS), dstBase);
      fullFrame.set(frameData.subarray(srcBase, srcBase + NUM_PIANO_KEYS), dstBase);
      fullVelocity.set(velocityData.subarray(srcBase, srcBase + NUM_PIANO_KEYS), dstBase);
      if (pedalFrameData) fullPedalFrame[gf] = pedalFrameData[lf];
      if (pedalOffsetData) fullPedalOffset[gf] = pedalOffsetData[lf];
    }

    // Explicitly dispose WASM/WebGPU tensors to keep memory usage flat across long pieces
    inputTensor.dispose();
    for (const key of Object.keys(results)) {
      results[key].dispose();
    }
  }

  onProgress?.(0.94, 'Decodificando notas, velocidades y pedal de resonancia...');
  const piece = decodeByteDanceOutputsToPiece(
    {
      numFrames: totalFrames,
      regOnset: fullRegOnset,
      regOffset: fullRegOffset,
      frame: fullFrame,
      velocity: fullVelocity,
      pedalFrame: fullPedalFrame,
      pedalOffset: fullPedalOffset,
    },
    actualDurationSec,
    title,
  );

  onProgress?.(1, 'Transcripción ByteDance Piano lista');
  return piece;
}

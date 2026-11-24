import * as tf from '@tensorflow/tfjs';
import {
  addPitchBendsToNoteEvents,
  BasicPitch,
  NoteEventTime,
  noteFramesToTime,
  outputToNotesPoly,
} from '@spotify/basic-pitch';
import { MAX_MIDI, MIN_MIDI } from './pianoTheory';
import {
  downsampleMono,
  estimateKeyFromChroma,
  isPitchInKey,
  PianoNoteEvent,
  TrackRole,
  TranscribedPiece,
} from './pitchDsp';

const BASIC_PITCH_SAMPLE_RATE = 22050;

let cachedModelPromise: Promise<tf.GraphModel> | null = null;
let backendReadyPromise: Promise<string> | null = null;

function isSoftwareWebGlEnvironment(): boolean {
  if (typeof document === 'undefined') return true;
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    if (!ext) return false;
    const renderer = String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '');
    return /swiftshader|llvmpipe|software|mesa/i.test(renderer);
  } catch {
    return true;
  }
}

async function ensureTfBackend(): Promise<string> {
  if (!backendReadyPromise) {
    backendReadyPromise = (async () => {
      const preferCpu = isSoftwareWebGlEnvironment();
      if (!preferCpu) {
        try {
          await tf.setBackend('webgl');
          await tf.ready();
          if (tf.getBackend() === 'webgl') return 'webgl';
        } catch {
          // Fallback to CPU backend below
        }
      }
      await tf.setBackend('cpu');
      await tf.ready();
      return tf.getBackend();
    })();
  }
  return backendReadyPromise;
}

function getModelUrl(): string {
  if (typeof window === 'undefined') {
    return './basic-pitch-model/model.json';
  }
  return new URL('./basic-pitch-model/model.json', window.location.href).href;
}

function loadBasicPitchModel(): Promise<tf.GraphModel> {
  if (!cachedModelPromise) {
    cachedModelPromise = (async () => {
      await ensureTfBackend();
      return tf.loadGraphModel(getModelUrl());
    })().catch((err) => {
      cachedModelPromise = null;
      throw err;
    });
  }
  return cachedModelPromise;
}

/**
 * Resamples a decoded Web Audio AudioBuffer into mono 22,050 Hz Float32Array
 * as required by Spotify's Basic Pitch neural model.
 */
async function resampleBufferToMono22050(
  audioBuffer: AudioBuffer,
  maxDurationSec = 210,
): Promise<Float32Array> {
  const dur = Math.min(audioBuffer.duration, maxDurationSec);
  const targetLen = Math.max(1, Math.ceil(dur * BASIC_PITCH_SAMPLE_RATE));

  // Prefer native OfflineAudioContext high-quality sinc resampling when available
  const OfflineCtx =
    typeof window !== 'undefined'
      ? window.OfflineAudioContext ||
        (window as unknown as { webkitOfflineAudioContext?: typeof OfflineAudioContext })
          .webkitOfflineAudioContext
      : undefined;

  if (OfflineCtx) {
    try {
      const offline = new OfflineCtx(1, targetLen, BASIC_PITCH_SAMPLE_RATE);
      const src = offline.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(offline.destination);
      src.start(0, 0, dur);
      const rendered = await offline.startRendering();
      return rendered.getChannelData(0);
    } catch {
      // Fallback to pure-JS resampler below
    }
  }

  const maxSamples = Math.min(audioBuffer.length, Math.floor(dur * audioBuffer.sampleRate));
  const left = audioBuffer.getChannelData(0).subarray(0, maxSamples);
  const right =
    audioBuffer.numberOfChannels > 1
      ? audioBuffer.getChannelData(1).subarray(0, maxSamples)
      : null;

  const mono = new Float32Array(maxSamples);
  if (right) {
    for (let i = 0; i < maxSamples; i++) {
      mono[i] = 0.5 * (left[i] + right[i]);
    }
  } else {
    mono.set(left);
  }
  return downsampleMono(mono, audioBuffer.sampleRate, BASIC_PITCH_SAMPLE_RATE);
}

/**
 * Estimates musical tempo (BPM) from neural note onset inter-onset intervals (IOI).
 */
export function estimateBpmFromNoteOnsets(notes: Array<{ startTimeSeconds: number }>): number {
  if (notes.length < 4) return 110;
  const sorted = [...notes].sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);

  // Deduplicate simultaneous chord onsets within 45 ms
  const onsets: number[] = [];
  for (const n of sorted) {
    if (onsets.length === 0 || n.startTimeSeconds - onsets[onsets.length - 1] > 0.045) {
      onsets.push(n.startTimeSeconds);
    }
  }
  if (onsets.length < 4) return 110;

  // Histogram BPM candidates in 68..164 BPM range
  const hist = new Float64Array(180);
  const maxPairs = Math.min(onsets.length, 240);
  for (let i = 0; i < maxPairs; i++) {
    for (let j = i + 1; j < Math.min(maxPairs, i + 6); j++) {
      const dt = onsets[j] - onsets[i];
      if (dt < 0.16 || dt > 1.8) continue;
      let bpm = 60 / dt;
      while (bpm < 72) bpm *= 2;
      while (bpm > 158) bpm *= 0.5;
      const bin = Math.round(bpm);
      if (bin >= 68 && bin <= 164) {
        hist[bin] += 1;
        if (bin - 1 >= 68) hist[bin - 1] += 0.45;
        if (bin + 1 <= 164) hist[bin + 1] += 0.45;
      }
    }
  }

  let bestBpm = 112;
  let bestScore = 0;
  for (let bpm = 68; bpm <= 164; bpm++) {
    const prior = 0.7 + 0.3 * Math.exp(-0.5 * Math.pow((bpm - 114) / 34, 2));
    const score = hist[bpm] * prior;
    if (score > bestScore) {
      bestScore = score;
      bestBpm = bpm;
    }
  }
  return bestBpm;
}

/**
 * Pure, unit-testable post-processor that transforms Spotify Basic Pitch neural
 * NoteEventTime[] detections into Aurelia's 4 clean, separated MIDI instrument tracks:
 *   - `vocal`:  Dominant upper-middle singing melody line (MIDI 53..84)
 *   - `bass`:   Clean monophonic bass foundation (MIDI 28..50)
 *   - `chords`: Harmonic piano accompaniment (MIDI 48..76, semitone-clash pruned)
 *   - `lead`:   High counter-melodies & sparkling upper arrangements (MIDI 70..96)
 */
export function classifyBasicPitchNotesToPiece(
  bpNotes: NoteEventTime[],
  durationSec: number,
  title: string,
): TranscribedPiece {
  // 1. Clean raw Basic Pitch detections: clamp to 88 keys, drop sub-60ms micro-blips
  const cleaned = bpNotes
    .filter(
      (n) =>
        Number.isFinite(n.startTimeSeconds) &&
        Number.isFinite(n.durationSeconds) &&
        n.durationSeconds >= 0.065 &&
        n.pitchMidi >= MIN_MIDI &&
        n.pitchMidi <= MAX_MIDI,
    )
    .map((n) => {
      const bends = n.pitchBends ?? [];
      let bendVariance = 0;
      if (bends.length > 2) {
        const mean = bends.reduce((s, v) => s + v, 0) / bends.length;
        bendVariance = bends.reduce((s, v) => s + Math.abs(v - mean), 0) / bends.length;
      }
      return {
        midi: Math.round(n.pitchMidi),
        startTime: Math.max(0, n.startTimeSeconds),
        duration: Math.max(0.07, Math.min(4.8, n.durationSeconds)),
        amplitude: Math.max(0.05, Math.min(1, n.amplitude)),
        hasVibrato: bendVariance > 0.22,
      };
    })
    .sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);

  // 2. Estimate global key signature & BPM
  const chroma = new Float64Array(12);
  for (const n of cleaned) {
    chroma[n.midi % 12] += n.amplitude * Math.min(1.5, n.duration);
  }
  const detectedKey = estimateKeyFromChroma(chroma);
  const bpm = estimateBpmFromNoteOnsets(
    cleaned.map((n) => ({ startTimeSeconds: n.startTime })),
  );

  // 3. Bucket notes into 90ms onset windows to identify top melody & bottom bass cleanly
  const WINDOW_SEC = 0.09;
  const maxAmp = cleaned.reduce((m, n) => (n.amplitude > m ? n.amplitude : m), 0.25);

  const vocalIdx = new Set<number>();
  const bassIdx = new Set<number>();

  // First pass: pick strongest melodic top voice per 90ms window in vocal register (MIDI 53..86)
  let i = 0;
  let lastVocalMidi = -1;
  while (i < cleaned.length) {
    const winStart = cleaned[i].startTime;
    let j = i;
    while (j < cleaned.length && cleaned[j].startTime - winStart <= WINDOW_SEC) {
      j++;
    }

    // Bass candidate: lowest strong note <= MIDI 50
    let bestBass = -1;
    let bestBassScore = -Infinity;
    // Vocal candidate: expressive / prominent top voice in MIDI 53..86
    let bestVocal = -1;
    let bestVocalScore = -Infinity;

    for (let k = i; k < j; k++) {
      const n = cleaned[k];
      if (n.midi <= 50 && n.midi >= 28) {
        const score = n.amplitude * 1.4 - (n.midi - 28) * 0.015;
        if (score > bestBassScore) {
          bestBassScore = score;
          bestBass = k;
        }
      }
      if (n.midi >= 53 && n.midi <= 86) {
        const tessitura = n.midi >= 58 && n.midi <= 80 ? 1.18 : 0.94;
        const vibratoBoost = n.hasVibrato ? 1.25 : 1.0;
        const continuity =
          lastVocalMidi >= 0 ? Math.max(0.75, 1.15 - Math.abs(n.midi - lastVocalMidi) * 0.03) : 1.0;
        const heightBoost = 0.85 + ((n.midi - 53) / 33) * 0.35;
        const score = n.amplitude * tessitura * vibratoBoost * continuity * heightBoost;
        if (score > bestVocalScore) {
          bestVocalScore = score;
          bestVocal = k;
        }
      }
    }

    if (bestBass >= 0) bassIdx.add(bestBass);
    if (bestVocal >= 0 && bestVocal !== bestBass) {
      vocalIdx.add(bestVocal);
      lastVocalMidi = cleaned[bestVocal].midi;
    }

    i = j;
  }

  // 4. Build final PianoNoteEvent list with calibrated confidence & role assignment
  const countsByTrack: Record<TrackRole, number> = {
    vocal: 0,
    chords: 0,
    bass: 0,
    lead: 0,
  };

  const events: PianoNoteEvent[] = [];
  let maxEnd = durationSec;

  for (let idx = 0; idx < cleaned.length; idx++) {
    const n = cleaned[idx];
    const inKey = isPitchInKey(n.midi, detectedKey.root, detectedKey.mode);
    const ampNorm = Math.min(1, n.amplitude / maxAmp);
    const durBonus = Math.min(0.22, n.duration * 0.28);
    const keyBonus = inKey ? 0.08 : -0.06;
    const confidence = Math.max(
      0.15,
      Math.min(0.99, ampNorm * 0.74 + durBonus + keyBonus),
    );

    let role: TrackRole;
    if (bassIdx.has(idx)) {
      role = 'bass';
    } else if (vocalIdx.has(idx)) {
      role = 'vocal';
    } else if (n.midi >= 74) {
      role = 'lead';
    } else if (n.midi < 46) {
      role = 'bass';
    } else {
      role = 'chords';
    }

    const startTime = Number(n.startTime.toFixed(3));
    const duration = Number(n.duration.toFixed(3));
    if (startTime + duration > maxEnd) maxEnd = startTime + duration;

    const baseVel =
      role === 'vocal'
        ? 0.56 + ampNorm * 0.42
        : role === 'bass'
          ? 0.44 + ampNorm * 0.42
          : role === 'lead'
            ? 0.36 + ampNorm * 0.42
            : 0.38 + ampNorm * 0.42;

    countsByTrack[role]++;
    events.push({
      id: `bp-${idx + 1}`,
      midi: n.midi,
      startTime,
      duration,
      velocity: Number(Math.max(0.25, Math.min(0.98, baseVel)).toFixed(2)),
      track: role,
      confidence: Number(confidence.toFixed(2)),
      inKey,
    });
  }

  return {
    id: `upload-${Date.now()}`,
    title,
    subtitle: 'MP3 transcrito con Spotify Basic Pitch (TensorFlow.js)',
    source: 'upload',
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
 * Full browser MP3 -> MIDI neural transcription pipeline powered by
 * Spotify's @spotify/basic-pitch running on TensorFlow.js.
 */
export async function transcribeAudioBufferWithBasicPitch(
  audioBuffer: AudioBuffer,
  title: string,
  onProgress?: (progress: number, stage: string) => void,
): Promise<TranscribedPiece> {
  onProgress?.(0.1, 'Remuestreando audio a 22.05 kHz para Basic Pitch...');
  const mono22050 = await resampleBufferToMono22050(audioBuffer, 210);

  onProgress?.(0.16, 'Cargando red neuronal Basic Pitch (TensorFlow.js)...');
  const modelPromise = loadBasicPitchModel();
  await modelPromise;

  const basicPitch = new BasicPitch(
    modelPromise as unknown as ConstructorParameters<typeof BasicPitch>[0],
  );
  const frames: number[][] = [];
  const onsets: number[][] = [];
  const contours: number[][] = [];

  onProgress?.(0.24, 'Analizando notas polifónicas y voz con Basic Pitch...');

  // Wrap TFJS inference inside an explicit tensor scope so intermediate WebGL/CPU tensors are freed
  tf.engine().startScope();
  try {
    await basicPitch.evaluateModel(
      mono22050,
      (f: number[][], o: number[][], c: number[][]) => {
        for (let k = 0; k < f.length; k++) {
          frames.push(f[k]);
          onsets.push(o[k]);
          contours.push(c[k]);
        }
      },
      (percent: number) => {
        const clamped = Math.max(0, Math.min(1, percent));
        onProgress?.(
          0.24 + clamped * 0.64,
          `Transcribiendo con Basic Pitch (${Math.round(clamped * 100)}%)...`,
        );
      },
    );
  } finally {
    tf.engine().endScope();
  }

  onProgress?.(0.91, 'Separando pistas MIDI (Voz, Acordes, Bajo y Arreglos)...');
  // Tuned thresholds for clean concert-piano + vocal extraction with minimal false noise
  const rawPoly = outputToNotesPoly(
    frames,
    onsets,
    0.52, // onsetThresh (clean note attacks)
    0.34, // frameThresh (reduces noisy ghost tails)
    6,    // minNoteLen (~70ms minimum note duration)
    true, // inferOnsets
    2200, // maxFreq (C7 upper concert limit — rejects hi-hat/cymbal hiss)
    38,   // minFreq (D1 lower bass limit — rejects sub-rumble)
    true, // melodiaTrick (tracks vocal/melodic contours accurately)
    11,   // energyTolerance
  );
  const withBends = addPitchBendsToNoteEvents(contours, rawPoly);
  const noteEvents = noteFramesToTime(withBends);

  const piece = classifyBasicPitchNotesToPiece(
    noteEvents,
    Math.min(audioBuffer.duration, 210),
    title,
  );

  onProgress?.(1, 'MIDI generado con Basic Pitch listo para reproducir');
  return piece;
}

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
  TranscriptionMode,
} from './pitchDsp';
import {
  PREPROCESS_SAMPLE_RATE,
  preprocessAudioForBasicPitchAsync,
  preprocessSoloPianoForBasicPitch,
  queryCenterScore,
  StereoCenterProfile,
} from './harmonicPreprocessor';

const BASIC_PITCH_SAMPLE_RATE = PREPROCESS_SAMPLE_RATE;

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
 * Resamples a decoded Web Audio AudioBuffer to 22,050 Hz while preserving stereo
 * Left & Right channels so center-panned vocals can be distinguished from wide accompaniment.
 */
async function resampleBufferToStereo22050(
  audioBuffer: AudioBuffer,
  maxDurationSec = 210,
): Promise<{ left: Float32Array; right: Float32Array | null }> {
  const dur = Math.min(audioBuffer.duration, maxDurationSec);
  const targetLen = Math.max(1, Math.ceil(dur * BASIC_PITCH_SAMPLE_RATE));
  const channels = audioBuffer.numberOfChannels >= 2 ? 2 : 1;

  const OfflineCtx =
    typeof window !== 'undefined'
      ? window.OfflineAudioContext ||
        (window as unknown as { webkitOfflineAudioContext?: typeof OfflineAudioContext })
          .webkitOfflineAudioContext
      : undefined;

  if (OfflineCtx) {
    try {
      const offline = new OfflineCtx(channels, targetLen, BASIC_PITCH_SAMPLE_RATE);
      const src = offline.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(offline.destination);
      src.start(0, 0, dur);
      const rendered = await offline.startRendering();
      return {
        left: rendered.getChannelData(0),
        right: channels >= 2 ? rendered.getChannelData(1) : null,
      };
    } catch {
      // Fallback to pure-JS anti-aliased resampler below
    }
  }

  const maxSamples = Math.min(audioBuffer.length, Math.floor(dur * audioBuffer.sampleRate));
  const leftRaw = audioBuffer.getChannelData(0).subarray(0, maxSamples);
  const left = downsampleMono(leftRaw, audioBuffer.sampleRate, BASIC_PITCH_SAMPLE_RATE);

  if (channels >= 2) {
    const rightRaw = audioBuffer.getChannelData(1).subarray(0, maxSamples);
    const right = downsampleMono(rightRaw, audioBuffer.sampleRate, BASIC_PITCH_SAMPLE_RATE);
    return { left, right };
  }

  return { left, right: null };
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

interface EnrichedBpNote {
  midi: number;
  startTime: number;
  duration: number;
  amplitude: number;
  hasVibrato: boolean;
  centerScore: number;
}

/**
 * Merges consecutive detections of the exact same MIDI pitch separated by a brief gap
 * (<= 65 ms) caused by vocal vibrato dips or soft re-articulation artifacts.
 */
export function stitchFragmentedNotes(notes: EnrichedBpNote[]): EnrichedBpNote[] {
  if (notes.length <= 1) return notes;
  const byPitch = new Map<number, EnrichedBpNote[]>();
  for (const n of notes) {
    const list = byPitch.get(n.midi);
    if (list) list.push(n);
    else byPitch.set(n.midi, [n]);
  }

  const merged: EnrichedBpNote[] = [];
  for (const list of byPitch.values()) {
    list.sort((a, b) => a.startTime - b.startTime);
    let curr = { ...list[0] };
    for (let i = 1; i < list.length; i++) {
      const next = list[i];
      const currEnd = curr.startTime + curr.duration;
      const gap = next.startTime - currEnd;
      // Merge micro-gaps (<= 65ms) unless the second note is a clearly stronger fresh re-strike
      if (gap <= 0.065 && next.amplitude < curr.amplitude * 1.45 && curr.duration + next.duration < 4.5) {
        const newEnd = Math.max(currEnd, next.startTime + next.duration);
        curr.duration = newEnd - curr.startTime;
        curr.amplitude = Math.max(curr.amplitude, next.amplitude);
        curr.hasVibrato = curr.hasVibrato || next.hasVibrato;
        curr.centerScore = Math.max(curr.centerScore, next.centerScore);
      } else {
        merged.push(curr);
        curr = { ...next };
      }
    }
    merged.push(curr);
  }

  return merged.sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);
}

/**
 * Suppresses false +12 (2nd harmonic / octave) and +19 (3rd harmonic / octave+5th)
 * overtone ghosts triggered simultaneously with strong lower fundamentals.
 */
export function pruneHarmonicGhosts(notes: EnrichedBpNote[]): EnrichedBpNote[] {
  const kept: EnrichedBpNote[] = [];
  for (let i = 0; i < notes.length; i++) {
    const cand = notes[i];
    const candEnd = cand.startTime + cand.duration;
    let isGhost = false;

    for (let j = 0; j < notes.length; j++) {
      if (i === j) continue;
      const fund = notes[j];
      const interval = cand.midi - fund.midi;
      if (interval !== 12 && interval !== 19) continue;

      const fundEnd = fund.startTime + fund.duration;
      const startsTogether = Math.abs(cand.startTime - fund.startTime) <= 0.055;
      const containedInside = cand.startTime >= fund.startTime - 0.03 && candEnd <= fundEnd + 0.09;
      if (!startsTogether && !containedInside) continue;

      // Preserve genuine center-panned vocal melody singing an octave above a wide bass/chord
      const clearlyIndependentVocal =
        cand.hasVibrato && !fund.hasVibrato && cand.centerScore > fund.centerScore + 0.22;
      if (clearlyIndependentVocal) continue;

      const ampLimit = interval === 12 ? 0.54 : 0.45;
      if (cand.amplitude < fund.amplitude * ampLimit) {
        isGhost = true;
        break;
      }
    }

    if (!isGhost) kept.push(cand);
  }
  return kept;
}

/**
 * Viterbi Dynamic Programming vocal melody tracker.
 * Finds the globally optimal monophonic vocal path across onset windows using stereo
 * center-panning dominance, pitch-bend vibrato, tessitura, amplitude, and smooth step motion.
 * Includes an explicit "rest" state so instrumental-only bars are not mislabeled as vocal.
 */
export function extractVocalPathViterbi(
  notes: EnrichedBpNote[],
  detectedKey: { root: number; mode: 'major' | 'minor' },
  isStereo: boolean,
): Set<number> {
  const WINDOW_SEC = 0.095;
  const maxAmp = notes.reduce((m, n) => (n.amplitude > m ? n.amplitude : m), 0.25);

  interface WinGroup {
    time: number;
    candidates: number[]; // indices into `notes`
  }

  const windows: WinGroup[] = [];
  let i = 0;
  while (i < notes.length) {
    const winStart = notes[i].startTime;
    let j = i;
    const cands: number[] = [];
    while (j < notes.length && notes[j].startTime - winStart <= WINDOW_SEC) {
      const n = notes[j];
      if (n.midi >= 52 && n.midi <= 86 && n.duration >= 0.09) {
        cands.push(j);
      }
      j++;
    }
    if (cands.length > 0) {
      windows.push({ time: winStart, candidates: cands });
    }
    i = j;
  }

  if (windows.length === 0) return new Set();

  // State 0 in each window = REST (no vocal in this window); States 1..K = candidates[s-1]
  const dpScores: number[][] = [];
  const backPtr: number[][] = [];

  const emissionScore = (idx: number): number => {
    const n = notes[idx];
    const ampNorm = Math.min(1, n.amplitude / maxAmp);
    const inKey = isPitchInKey(n.midi, detectedKey.root, detectedKey.mode);

    // Vocal tessitura prior: strongest in A3..G#5 (MIDI 57..80)
    let tessitura = 0.12;
    if (n.midi >= 57 && n.midi <= 80) tessitura = 0.34;
    else if (n.midi >= 54 && n.midi <= 83) tessitura = 0.22;

    // Higher pitch within window tends to carry top melody
    const heightBonus = ((n.midi - 52) / 34) * 0.22;

    // Stereo center panning is a major cue for lead vocals in commercial mixes
    let stereoBonus = 0;
    if (isStereo) {
      if (n.centerScore >= 0.58) stereoBonus = 0.22 + (n.centerScore - 0.58) * 0.65;
      else if (n.centerScore <= 0.32) stereoBonus = -0.28;
    }

    const vibratoBonus = n.hasVibrato ? 0.28 : 0;
    const durBonus = Math.min(0.18, n.duration * 0.22);
    const keyBonus = inKey ? 0.08 : -0.05;

    return ampNorm * 0.72 + tessitura + heightBonus + stereoBonus + vibratoBonus + durBonus + keyBonus;
  };

  // Threshold baseline for REST state — candidates must beat this to claim the vocal track
  const REST_EMISSION = isStereo ? 0.86 : 0.78;

  for (let w = 0; w < windows.length; w++) {
    const cands = windows[w].candidates;
    const numStates = cands.length + 1;
    const curScores = new Array<number>(numStates).fill(-Infinity);
    const curPtr = new Array<number>(numStates).fill(0);

    if (w === 0) {
      curScores[0] = REST_EMISSION;
      for (let s = 1; s < numStates; s++) {
        curScores[s] = emissionScore(cands[s - 1]);
      }
    } else {
      const prevCands = windows[w - 1].candidates;
      const prevScores = dpScores[w - 1];
      const dt = Math.max(0.05, windows[w].time - windows[w - 1].time);

      for (let s = 0; s < numStates; s++) {
        const emit = s === 0 ? REST_EMISSION : emissionScore(cands[s - 1]);
        let bestPrevScore = -Infinity;
        let bestPrev = 0;

        for (let ps = 0; ps < prevScores.length; ps++) {
          let trans = 0;
          if (s === 0 && ps === 0) {
            trans = 0.04; // resting stays slightly coherent across instrumental bars
          } else if (s === 0 || ps === 0) {
            trans = -0.04; // small cost to start/end a vocal phrase
          } else {
            const prevMidi = notes[prevCands[ps - 1]].midi;
            const currMidi = notes[cands[s - 1]].midi;
            const jump = Math.abs(currMidi - prevMidi);
            if (jump <= 4) trans = 0.2;
            else if (jump <= 7) trans = 0.05;
            else if (jump <= 12) trans = -0.14;
            else trans = dt < 0.65 ? -0.52 : -0.22;
          }

          const total = prevScores[ps] + trans + emit;
          if (total > bestPrevScore) {
            bestPrevScore = total;
            bestPrev = ps;
          }
        }

        curScores[s] = bestPrevScore;
        curPtr[s] = bestPrev;
      }
    }

    dpScores.push(curScores);
    backPtr.push(curPtr);
  }

  // Backtrack best global path
  const lastScores = dpScores[windows.length - 1];
  let bestState = 0;
  for (let s = 1; s < lastScores.length; s++) {
    if (lastScores[s] > lastScores[bestState]) bestState = s;
  }

  const vocalIdx = new Set<number>();
  for (let w = windows.length - 1; w >= 0; w--) {
    if (bestState > 0) {
      vocalIdx.add(windows[w].candidates[bestState - 1]);
    }
    bestState = backPtr[w][bestState];
  }

  return vocalIdx;
}

/**
 * Phase-aligned gentle 1/16-note rhythmic micro-quantization.
 * Pulls minor frame-hop jitter (<= 26 ms) 55% toward the detected beat grid
 * while preserving natural rubato and expressive phrasing.
 */
export function softQuantizeOnsets(notes: EnrichedBpNote[], bpm: number): void {
  if (notes.length < 4 || bpm < 50 || bpm > 220) return;
  const sixteenth = 60 / bpm / 4;
  if (sixteenth < 0.06) return;

  // Estimate global beat phase offset that maximizes onset alignment
  const steps = 16;
  let bestPhase = 0;
  let bestScore = -1;
  for (let p = 0; p < steps; p++) {
    const phase = (p / steps) * sixteenth;
    let score = 0;
    for (let i = 0; i < Math.min(notes.length, 300); i++) {
      const shifted = notes[i].startTime - phase;
      const gridIdx = Math.round(shifted / sixteenth);
      const dist = Math.abs(shifted - gridIdx * sixteenth);
      score += Math.exp(-0.5 * Math.pow(dist / 0.018, 2));
    }
    if (score > bestScore) {
      bestScore = score;
      bestPhase = phase;
    }
  }

  const MAX_SNAP_DIST = 0.026;
  const PULL_STRENGTH = 0.55;
  for (const n of notes) {
    const shifted = n.startTime - bestPhase;
    const snapped = Math.round(shifted / sixteenth) * sixteenth + bestPhase;
    const diff = snapped - n.startTime;
    if (Math.abs(diff) <= MAX_SNAP_DIST) {
      n.startTime = Math.max(0, n.startTime + diff * PULL_STRENGTH);
    }
  }
  notes.sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);
}

/**
 * Pure, unit-testable precision post-processor that transforms Spotify Basic Pitch
 * neural detections (+ optional StereoCenterProfile) into Aurelia's 4 clean MIDI tracks:
 *   - `vocal`:  Viterbi-tracked center-panned singing melody (MIDI 52..86)
 *   - `bass`:   Clean monophonic bass foundation (MIDI 28..50)
 *   - `chords`: Semitone-clash-free piano harmony (MIDI 45..76, max 4 simultaneous voices)
 *   - `lead`:   Sparkling upper counter-melodies & arpeggios (MIDI 74..96)
 */
export function classifyBasicPitchNotesToPiece(
  bpNotes: NoteEventTime[],
  durationSec: number,
  title: string,
  centerProfile?: StereoCenterProfile | null,
): TranscribedPiece {
  // 1. Clean raw Basic Pitch detections: clamp to 88 keys, drop sub-65ms micro-blips
  const initial: EnrichedBpNote[] = bpNotes
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
      const midi = Math.round(n.pitchMidi);
      const startTime = Math.max(0, n.startTimeSeconds);
      const duration = Math.max(0.07, Math.min(4.8, n.durationSeconds));
      return {
        midi,
        startTime,
        duration,
        amplitude: Math.max(0.05, Math.min(1, n.amplitude)),
        hasVibrato: bendVariance > 0.2,
        centerScore: queryCenterScore(centerProfile, startTime, duration, midi),
      };
    })
    .sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);

  // 2. Stitch vibrato-split same-pitch segments & prune +12 / +19 harmonic overtone ghosts
  const stitched = stitchFragmentedNotes(initial);
  const cleaned = pruneHarmonicGhosts(stitched);

  // 3. Estimate global key signature & BPM, then apply gentle phase-aligned micro-quantization
  const chroma = new Float64Array(12);
  for (const n of cleaned) {
    chroma[n.midi % 12] += n.amplitude * Math.min(1.5, n.duration);
  }
  const detectedKey = estimateKeyFromChroma(chroma);
  const bpm = estimateBpmFromNoteOnsets(
    cleaned.map((n) => ({ startTimeSeconds: n.startTime })),
  );
  softQuantizeOnsets(cleaned, bpm);

  // 4. Identify clean monophonic Bass & Viterbi-tracked Vocal melody
  const isStereo = Boolean(centerProfile?.isStereo);
  const vocalIdx = extractVocalPathViterbi(cleaned, detectedKey, isStereo);
  const bassIdx = new Set<number>();

  const WINDOW_SEC = 0.09;
  let i = 0;
  while (i < cleaned.length) {
    const winStart = cleaned[i].startTime;
    let j = i;
    let bestBass = -1;
    let bestBassScore = -Infinity;
    while (j < cleaned.length && cleaned[j].startTime - winStart <= WINDOW_SEC) {
      const n = cleaned[j];
      if (!vocalIdx.has(j) && n.midi >= 28 && n.midi <= 50) {
        const keyBoost = isPitchInKey(n.midi, detectedKey.root, detectedKey.mode) ? 0.12 : 0;
        const score = n.amplitude * 1.45 - (n.midi - 28) * 0.015 + keyBoost;
        if (score > bestBassScore) {
          bestBassScore = score;
          bestBass = j;
        }
      }
      j++;
    }
    if (bestBass >= 0) bassIdx.add(bestBass);
    i = j;
  }

  // 5. Build calibrated events with chord clash-pruning & clean monophonic bass voice-leading
  const maxAmp = cleaned.reduce((m, n) => (n.amplitude > m ? n.amplitude : m), 0.25);
  const countsByTrack: Record<TrackRole, number> = {
    vocal: 0,
    chords: 0,
    bass: 0,
    lead: 0,
  };

  const rawEvents: PianoNoteEvent[] = [];
  for (let idx = 0; idx < cleaned.length; idx++) {
    const n = cleaned[idx];
    const inKey = isPitchInKey(n.midi, detectedKey.root, detectedKey.mode);
    const ampNorm = Math.min(1, n.amplitude / maxAmp);
    const durBonus = Math.min(0.22, n.duration * 0.28);
    const keyBonus = inKey ? 0.09 : -0.07;

    let role: TrackRole;
    if (bassIdx.has(idx)) {
      role = 'bass';
    } else if (vocalIdx.has(idx)) {
      role = 'vocal';
    } else if (n.midi >= 74) {
      role = 'lead';
    } else if (n.midi < 45) {
      // Skip secondary bass clutter notes that lost the monophonic bass window
      continue;
    } else {
      role = 'chords';
    }

    const vocalCenterBonus = role === 'vocal' && isStereo ? (n.centerScore - 0.5) * 0.16 : 0;
    const confidence = Math.max(
      0.16,
      Math.min(0.99, ampNorm * 0.72 + durBonus + keyBonus + vocalCenterBonus),
    );

    const baseVel =
      role === 'vocal'
        ? 0.58 + ampNorm * 0.4
        : role === 'bass'
          ? 0.46 + ampNorm * 0.4
          : role === 'lead'
            ? 0.36 + ampNorm * 0.4
            : 0.38 + ampNorm * 0.4;

    rawEvents.push({
      id: `bp-${idx + 1}`,
      midi: n.midi,
      startTime: Number(n.startTime.toFixed(3)),
      duration: Number(n.duration.toFixed(3)),
      velocity: Number(Math.max(0.25, Math.min(0.98, baseVel)).toFixed(2)),
      track: role,
      confidence: Number(confidence.toFixed(2)),
      inKey,
    });
  }

  // Enforce clean monophonic bass (trim overlapping previous bass tail before new bass attack)
  // and prune semitone clashes inside simultaneous chords (cap at 4 cleanest chord voices).
  const events: PianoNoteEvent[] = [];
  let lastBass: PianoNoteEvent | null = null;
  let maxEnd = durationSec;

  for (const ev of rawEvents) {
    if (ev.track === 'bass') {
      if (lastBass && lastBass.startTime + lastBass.duration > ev.startTime - 0.015) {
        const trimmed = Math.max(0.08, ev.startTime - lastBass.startTime - 0.015);
        lastBass.duration = Number(trimmed.toFixed(3));
      }
      lastBass = ev;
    } else if (ev.track === 'chords') {
      let activeChords = 0;
      let clashIndex = -1;
      for (let k = events.length - 1; k >= 0; k--) {
        const prev = events[k];
        if (prev.startTime < ev.startTime - 2.5) break;
        if (prev.track !== 'chords') continue;
        if (prev.startTime + prev.duration <= ev.startTime + 0.025) continue;
        activeChords++;
        if (Math.abs(prev.midi - ev.midi) === 1) {
          clashIndex = k;
        }
      }
      if (clashIndex >= 0) {
        const prev = events[clashIndex];
        const prevScore = prev.confidence + (prev.inKey ? 0.12 : 0);
        const curScore = ev.confidence + (ev.inKey ? 0.12 : 0);
        if (curScore <= prevScore) {
          continue;
        }
        // Replace weaker clashing chord tone
        countsByTrack.chords--;
        events.splice(clashIndex, 1);
      } else if (activeChords >= 4) {
        continue;
      }
    }

    countsByTrack[ev.track]++;
    events.push(ev);
    const end = ev.startTime + ev.duration;
    if (end > maxEnd) maxEnd = end;
  }

  return {
    id: `upload-${Date.now()}`,
    title,
    subtitle: isStereo
      ? 'Canción limpiada (HPSS + Estéreo) y transcrita con Basic Pitch'
      : 'Canción limpiada (HPSS) y transcrita con Basic Pitch',
    source: 'upload',
    transcriptionMode: 'song',
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
 * Pure, unit-testable post-processor specifically optimized for SOLO PIANO MP3 recordings.
 * Preserves:
 *   - Full 88-key pitch range (A0 = MIDI 21 .. C8 = MIDI 108)
 *   - True two-handed polyphony (chords + left-hand bass octaves + inner voices)
 *   - Fast 16th/32nd piano runs & grace notes (>= 45 ms)
 *   - Natural sustain pedal decay (up to 6.5 s) and expressive pianist rubato (no rigid grid snap)
 *   - Rich classical/jazz chromaticism while gently pruning faint sympathetic string overtones
 */
export function classifyPianoRecordingToPiece(
  bpNotes: NoteEventTime[],
  durationSec: number,
  title: string,
): TranscribedPiece {
  // 1. Keep full 88-key range and fast piano runs (>= 45 ms)
  const raw = bpNotes
    .filter(
      (n) =>
        Number.isFinite(n.startTimeSeconds) &&
        Number.isFinite(n.durationSeconds) &&
        n.durationSeconds >= 0.045 &&
        n.pitchMidi >= MIN_MIDI &&
        n.pitchMidi <= MAX_MIDI,
    )
    .map((n) => ({
      midi: Math.round(n.pitchMidi),
      startTime: Math.max(0, n.startTimeSeconds),
      duration: Math.max(0.045, Math.min(6.5, n.durationSeconds)),
      amplitude: Math.max(0.05, Math.min(1, n.amplitude)),
    }))
    .sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);

  // 2. Deduplicate ONLY sub-32ms accidental double-triggers on the exact same piano key
  //    (never merges intentional fast repeated piano notes!)
  const deduped: typeof raw = [];
  const byKey = new Map<number, typeof raw>();
  for (const n of raw) {
    const arr = byKey.get(n.midi);
    if (arr) arr.push(n);
    else byKey.set(n.midi, [n]);
  }
  for (const arr of byKey.values()) {
    arr.sort((a, b) => a.startTime - b.startTime);
    let curr = { ...arr[0] };
    for (let i = 1; i < arr.length; i++) {
      const next = arr[i];
      const currEnd = curr.startTime + curr.duration;
      const gap = next.startTime - currEnd;
      if (gap <= 0.032 && next.amplitude < curr.amplitude * 0.9 && curr.duration + next.duration < 6.5) {
        curr.duration = Math.max(currEnd, next.startTime + next.duration) - curr.startTime;
        curr.amplitude = Math.max(curr.amplitude, next.amplitude);
      } else {
        // Trim micro-overlap if same key re-strikes cleanly
        if (next.startTime < currEnd) {
          curr.duration = Math.max(0.04, next.startTime - curr.startTime - 0.008);
        }
        deduped.push(curr);
        curr = { ...next };
      }
    }
    deduped.push(curr);
  }
  deduped.sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);

  // 3. Gentle piano-acoustic resonance pruning:
  //    - Prune ONLY very faint sympathetic +12 (< 34%) or +19 (< 28%) string overtones
  //      (real piano octave doublings are strong and preserved 100%!)
  //    - Prune ONLY faint short semitone bleed (< 34% amplitude & < 140 ms duration)
  //      (real classical/jazz minor-2nd & major-7th voicings are preserved 100%!)
  const cleaned: typeof raw = [];
  for (let i = 0; i < deduped.length; i++) {
    const cand = deduped[i];
    const candEnd = cand.startTime + cand.duration;
    let isAcousticGhost = false;

    for (let j = 0; j < deduped.length; j++) {
      if (i === j) continue;
      const other = deduped[j];
      const otherEnd = other.startTime + other.duration;

      // Sympathetic overtone (+12 or +19) inside a much louder lower piano note
      const interval = cand.midi - other.midi;
      if (interval === 12 || interval === 19) {
        const synced = Math.abs(cand.startTime - other.startTime) <= 0.045;
        const inside = cand.startTime >= other.startTime - 0.025 && candEnd <= otherEnd + 0.08;
        if (synced || inside) {
          const limit = interval === 12 ? 0.34 : 0.28;
          if (cand.amplitude < other.amplitude * limit) {
            isAcousticGhost = true;
            break;
          }
        }
      }

      // Faint, brief semitone neighbor bleed from piano hammer inharmonicity
      if (
        Math.abs(cand.midi - other.midi) === 1 &&
        cand.duration < 0.14 &&
        Math.abs(cand.startTime - other.startTime) <= 0.045 &&
        cand.amplitude < other.amplitude * 0.34
      ) {
        isAcousticGhost = true;
        break;
      }
    }

    if (!isAcousticGhost) cleaned.push(cand);
  }

  // 4. Estimate global key & tempo (for HUD metadata & MIDI header — NO rigid grid snapping!)
  const chroma = new Float64Array(12);
  for (const n of cleaned) {
    chroma[n.midi % 12] += n.amplitude * Math.min(2.0, n.duration);
  }
  const detectedKey = estimateKeyFromChroma(chroma);
  const bpm = estimateBpmFromNoteOnsets(
    cleaned.map((n) => ({ startTimeSeconds: n.startTime })),
  );

  // 5. Identify singing right-hand top melody voice (highest prominent note >= MIDI 58 per 65ms window)
  const melodyIdx = new Set<number>();
  const WIN_SEC = 0.065;
  let i = 0;
  while (i < cleaned.length) {
    const winStart = cleaned[i].startTime;
    let j = i;
    let bestMel = -1;
    let bestMelScore = -Infinity;
    while (j < cleaned.length && cleaned[j].startTime - winStart <= WIN_SEC) {
      const n = cleaned[j];
      if (n.midi >= 58) {
        // Favor top soprano piano note while requiring musical prominence
        const heightScore = (n.midi - 58) * 0.025;
        const score = n.amplitude * 1.15 + heightScore;
        if (score > bestMelScore) {
          bestMelScore = score;
          bestMel = j;
        }
      }
      j++;
    }
    if (bestMel >= 0) melodyIdx.add(bestMel);
    i = j;
  }

  // 6. Map full 88-key polyphonic piano into Aurelia's 4 playable tracks without dropping voices
  const maxAmp = cleaned.reduce((m, n) => (n.amplitude > m ? n.amplitude : m), 0.25);
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
    const durBonus = Math.min(0.24, n.duration * 0.35);
    // Gentle key hint only (+0.03) so chromatic classical/jazz notes never lose confidence
    const confidence = Math.max(
      0.16,
      Math.min(0.99, ampNorm * 0.74 + durBonus + (inKey ? 0.03 : 0)),
    );

    let role: TrackRole;
    if (n.midi <= 47) {
      role = 'bass'; // Polyphonic left-hand bass (preserves octave bass!)
    } else if (melodyIdx.has(idx)) {
      role = 'vocal'; // Right-hand top soprano melody line
    } else if (n.midi >= 77) {
      role = 'lead'; // Sparkling upper treble register
    } else {
      role = 'chords'; // Inner piano harmony voices
    }

    const startTime = Number(n.startTime.toFixed(3));
    const duration = Number(n.duration.toFixed(3));
    if (startTime + duration > maxEnd) maxEnd = startTime + duration;

    // Preserve natural expressive piano velocity contour (pp -> ff)
    const velocity = Number(Math.max(0.24, Math.min(0.98, 0.28 + ampNorm * 0.68)).toFixed(2));

    countsByTrack[role]++;
    events.push({
      id: `bp-piano-${idx + 1}`,
      midi: n.midi,
      startTime,
      duration,
      velocity,
      track: role,
      confidence: Number(confidence.toFixed(2)),
      inKey,
    });
  }

  return {
    id: `upload-${Date.now()}`,
    title,
    subtitle: 'Grabación de piano optimizada con Basic Pitch (88 teclas)',
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
 * Full 100% client-side precision MP3 -> MIDI neural transcription pipeline:
 *   - `mode === 'piano'` (default): Preserves crisp piano hammer attacks, full 88-key bandwidth
 *     (27.5 Hz - 4186 Hz), disables monophonic melodiaTrick for true two-handed polyphony,
 *     keeps fast 16th/32nd runs & pianist rubato, and prunes only faint sympathetic overtones.
 *   - `mode === 'song'`: Runs STFT 2D HPSS drum suppression + Stereo Center-Vocal profiling
 *     + Viterbi vocal melody tracking + gentle rhythmic micro-quantization.
 */
export async function transcribeAudioBufferWithBasicPitch(
  audioBuffer: AudioBuffer,
  title: string,
  onProgress?: (progress: number, stage: string) => void,
  mode: TranscriptionMode = 'piano',
): Promise<TranscribedPiece> {
  onProgress?.(0.06, 'Remuestreando audio a 22.05 kHz...');
  const { left, right } = await resampleBufferToStereo22050(audioBuffer, 210);

  let modelInputPcm: Float32Array;
  let centerProfile: StereoCenterProfile | null = null;

  if (mode === 'piano') {
    onProgress?.(0.14, 'Optimizando ataque de martillos y rango de 88 teclas (modo Piano)...');
    modelInputPcm = preprocessSoloPianoForBasicPitch(left, right, BASIC_PITCH_SAMPLE_RATE);
  } else {
    const preprocessed = await preprocessAudioForBasicPitchAsync(
      left,
      right,
      BASIC_PITCH_SAMPLE_RATE,
      onProgress,
    );
    modelInputPcm = preprocessed.harmonicPcm;
    centerProfile = preprocessed.centerProfile;
  }

  onProgress?.(0.24, 'Cargando red neuronal Basic Pitch (TensorFlow.js)...');
  const modelPromise = loadBasicPitchModel();
  await modelPromise;

  const basicPitch = new BasicPitch(
    modelPromise as unknown as ConstructorParameters<typeof BasicPitch>[0],
  );
  const frames: number[][] = [];
  const onsets: number[][] = [];
  const contours: number[][] = [];

  onProgress?.(
    0.28,
    mode === 'piano'
      ? 'Transcribiendo polifonía completa de piano con Basic Pitch...'
      : 'Transcribiendo señal armónica limpia con Basic Pitch...',
  );

  tf.engine().startScope();
  try {
    await basicPitch.evaluateModel(
      modelInputPcm,
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
          0.28 + clamped * 0.62,
          mode === 'piano'
            ? `Transcribiendo piano con Basic Pitch (${Math.round(clamped * 100)}%)...`
            : `Transcribiendo canción con Basic Pitch (${Math.round(clamped * 100)}%)...`,
        );
      },
    );
  } finally {
    tf.engine().endScope();
  }

  if (mode === 'piano') {
    onProgress?.(0.92, 'Reconstruyendo acordes, pedal y melodía de piano...');
    // Piano-calibrated Basic Pitch decoding:
    //   - melodiaTrick = FALSE: critical so inner voices in 4-8 note piano chords are NOT stripped!
    //   - 27 Hz .. 4200 Hz: full 88-key piano coverage (A0 .. C8)
    //   - minNoteLen = 4 (~46 ms): captures fast 16th/32nd piano passages & grace notes
    //   - onsetThresh = 0.44 / frameThresh = 0.28: sensitive to delicate pianissimo notes & pedal tails
    const rawPoly = outputToNotesPoly(
      frames,
      onsets,
      0.44,  // onsetThresh
      0.28,  // frameThresh
      4,     // minNoteLen (~46 ms)
      true,  // inferOnsets (detects repeated strikes under sustain pedal)
      4200,  // maxFreq (C8 = 4186 Hz)
      27,    // minFreq (A0 = 27.5 Hz)
      false, // melodiaTrick = false -> preserves full polyphonic piano voicings!
      11,    // energyTolerance
    );
    const noteEvents = noteFramesToTime(rawPoly);
    const piece = classifyPianoRecordingToPiece(
      noteEvents,
      Math.min(audioBuffer.duration, 210),
      title,
    );
    onProgress?.(1, 'MIDI de piano listo para reproducir');
    return piece;
  }

  onProgress?.(0.92, 'Rastreando melodía vocal (Viterbi) y limpiando armónicos...');
  const rawPoly = outputToNotesPoly(
    frames,
    onsets,
    0.48,
    0.3,
    6,
    true,
    2200,
    38,
    true,
    11,
  );
  const withBends = addPitchBendsToNoteEvents(contours, rawPoly);
  const noteEvents = noteFramesToTime(withBends);

  const piece = classifyBasicPitchNotesToPiece(
    noteEvents,
    Math.min(audioBuffer.duration, 210),
    title,
    centerProfile,
  );

  onProgress?.(1, 'MIDI de alta precisión listo para reproducir');
  return piece;
}

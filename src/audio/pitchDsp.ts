import { midiToFreq } from './pianoTheory';

export type TrackRole = 'vocal' | 'chords' | 'bass' | 'lead';

export type CleanlinessLevel = 'soft' | 'balanced' | 'strict';

export type TranscriptionMode = 'piano-hd' | 'piano' | 'song';

export interface PianoNoteEvent {
  id: string;
  midi: number;
  startTime: number;
  duration: number;
  velocity: number;
  track: TrackRole;
  /** Musical & spectral confidence (0..1) used for live cleanliness filtering */
  confidence: number;
  /** Whether this pitch belongs to the estimated global scale */
  inKey: boolean;
}

export interface TranscribedPiece {
  id: string;
  title: string;
  subtitle: string;
  source: 'demo' | 'upload' | 'midi';
  transcriptionMode?: TranscriptionMode;
  duration: number;
  bpm: number;
  keySignature: string;
  keyRoot: number; // 0..11
  keyMode: 'major' | 'minor';
  notes: PianoNoteEvent[];
  countsByTrack: Record<TrackRole, number>;
}

export interface TranscriptionOptions {
  maxDurationSec?: number;
  targetSampleRate?: number;
  onProgress?: (progress: number, stage: string) => void;
}

const ANALYSIS_MIN_MIDI = 28; // E1 (41.2 Hz)
const ANALYSIS_MAX_MIDI = 96; // C7 (2093 Hz)
const ANALYSIS_NUM_NOTES = ANALYSIS_MAX_MIDI - ANALYSIS_MIN_MIDI + 1; // 69 notes

/**
 * In-place Radix-2 Cooley-Tukey FFT.
 */
export function fftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (n <= 1) return;

  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      const ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wLenR = Math.cos(angle);
    const wLenI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wR = 1;
      let wI = 0;
      for (let k = 0; k < half; k++) {
        const uR = real[i + k];
        const uI = imag[i + k];
        const vR = real[i + k + half] * wR - imag[i + k + half] * wI;
        const vI = real[i + k + half] * wI + imag[i + k + half] * wR;

        real[i + k] = uR + vR;
        imag[i + k] = uI + vI;
        real[i + k + half] = uR - vR;
        imag[i + k + half] = uI - vI;

        const nextWR = wR * wLenR - wI * wLenI;
        wI = wR * wLenI + wI * wLenR;
        wR = nextWR;
      }
    }
  }
}

/**
 * Anti-aliased 2:1 decimation using a symmetric 5-tap binomial low-pass kernel
 * [1, 4, 6, 4, 1] / 16 before halving sample rate. Zero aliasing artifacts.
 */
export function decimateBy2(input: Float32Array): Float32Array {
  const n = input.length;
  const outLen = Math.floor(n / 2);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const c = i * 2;
    const s0 = input[Math.max(0, c - 2)];
    const s1 = input[Math.max(0, c - 1)];
    const s2 = input[c];
    const s3 = input[Math.min(n - 1, c + 1)];
    const s4 = input[Math.min(n - 1, c + 2)];
    out[i] = 0.0625 * s0 + 0.25 * s1 + 0.375 * s2 + 0.25 * s3 + 0.0625 * s4;
  }
  return out;
}

/**
 * Clean resampling to target sample rate using anti-aliased window averaging.
 */
export function downsampleMono(
  input: Float32Array,
  fromRate: number,
  toRate: number,
): Float32Array {
  if (Math.abs(fromRate - toRate) < 1) return input;
  if (toRate > fromRate) {
    const ratio = fromRate / toRate;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = input[idx] ?? 0;
      const b = input[Math.min(input.length - 1, idx + 1)] ?? a;
      out[i] = a + (b - a) * frac;
    }
    return out;
  }

  // Integer 2:1 fast path (e.g. 44100 -> 22050)
  if (Math.abs(fromRate / toRate - 2) < 0.01) {
    return decimateBy2(input);
  }

  const ratio = fromRate / toRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  const halfWin = Math.max(1, Math.ceil(ratio * 0.75));
  for (let i = 0; i < outLen; i++) {
    const center = i * ratio;
    const start = Math.max(0, Math.floor(center - halfWin));
    const end = Math.min(input.length - 1, Math.ceil(center + halfWin));
    let wSum = 0;
    let acc = 0;
    for (let k = start; k <= end; k++) {
      const d = (k - center) / halfWin;
      const w = 0.5 * (1 + Math.cos(Math.PI * Math.min(1, Math.abs(d))));
      acc += input[k] * w;
      wSum += w;
    }
    out[i] = wSum > 0 ? acc / wSum : 0;
  }
  return out;
}

function createHannWindow(size: number): Float32Array {
  const win = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return win;
}

function medianOfSmall(arr: Float32Array, len: number): number {
  for (let i = 1; i < len; i++) {
    const key = arr[i];
    let j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
  return arr[len >> 1];
}

function percentileOfPositive(values: Float32Array, p: number): number {
  const pos: number[] = [];
  for (let i = 0; i < values.length; i++) {
    if (values[i] > 1e-6) pos.push(values[i]);
  }
  if (pos.length === 0) return 0;
  pos.sort((a, b) => a - b);
  const idx = Math.min(pos.length - 1, Math.max(0, Math.floor(pos.length * p)));
  return pos[idx];
}

/**
 * 2D Harmonic-Percussive Source Separation (HPSS) on a Semitone Energy Matrix [frames x bins].
 * Strips broadband percussive transients (drums, snares, hi-hats, clicks) via horizontal
 * temporal median filtering vs vertical pitch median filtering + Wiener harmonic masking.
 */
export function applyHarmonicPercussiveMask(
  matrix: Float32Array,
  numFrames: number,
  numBins: number,
  timeHalfWin = 4,
  pitchHalfWin = 5,
): Float32Array {
  const cleaned = new Float32Array(matrix.length);
  const scratchT = new Float32Array(timeHalfWin * 2 + 1);
  const scratchF = new Float32Array(pitchHalfWin * 2 + 1);

  for (let f = 0; f < numFrames; f++) {
    const tStart = Math.max(0, f - timeHalfWin);
    const tEnd = Math.min(numFrames - 1, f + timeHalfWin);
    const tLen = tEnd - tStart + 1;
    const rowBase = f * numBins;

    for (let b = 0; b < numBins; b++) {
      const val = matrix[rowBase + b];
      if (val <= 1e-6) continue;

      let ti = 0;
      for (let t = tStart; t <= tEnd; t++) {
        scratchT[ti++] = matrix[t * numBins + b];
      }
      const hEst = medianOfSmall(scratchT, tLen);

      const bStart = Math.max(0, b - pitchHalfWin);
      const bEnd = Math.min(numBins - 1, b + pitchHalfWin);
      const bLen = bEnd - bStart + 1;
      let bi = 0;
      for (let k = bStart; k <= bEnd; k++) {
        scratchF[bi++] = matrix[rowBase + k];
      }
      const pEst = medianOfSmall(scratchF, bLen);

      const h2 = hEst * hEst;
      const p2 = (pEst * 0.85) * (pEst * 0.85);
      const harmonicMask = h2 / (h2 + p2 + 1e-10);
      const aboveFloor = Math.max(0, val - pEst * 0.85);

      cleaned[rowBase + b] = aboveFloor * harmonicMask;
    }
  }

  return cleaned;
}

const NOTE_NAMES_ES = [
  'Do',
  'Do#',
  'Re',
  'Mi♭',
  'Mi',
  'Fa',
  'Fa#',
  'Sol',
  'La♭',
  'La',
  'Si♭',
  'Si',
];

const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

const MAJOR_SCALE_INTERVALS = new Set([0, 2, 4, 5, 7, 9, 11]);
const MINOR_SCALE_INTERVALS = new Set([0, 2, 3, 5, 7, 8, 10, 11]);

export function isPitchInKey(midi: number, root: number, mode: 'major' | 'minor'): boolean {
  const interval = (((midi % 12) - root) % 12 + 12) % 12;
  return mode === 'major'
    ? MAJOR_SCALE_INTERVALS.has(interval)
    : MINOR_SCALE_INTERVALS.has(interval);
}

export function estimateKeyFromChroma(chroma: Float64Array): {
  label: string;
  root: number;
  mode: 'major' | 'minor';
} {
  let bestScore = -Infinity;
  let bestRoot = 0;
  let bestMode: 'major' | 'minor' = 'major';

  const correlate = (profile: number[], shift: number): number => {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < 12; i++) {
      const a = chroma[(i + shift) % 12];
      const b = profile[i];
      dot += a * b;
      normA += a * a;
      normB += b * b;
    }
    return normA > 0 && normB > 0 ? dot / Math.sqrt(normA * normB) : 0;
  };

  for (let root = 0; root < 12; root++) {
    const maj = correlate(MAJOR_PROFILE, root);
    if (maj > bestScore) {
      bestScore = maj;
      bestRoot = root;
      bestMode = 'major';
    }
    const min = correlate(MINOR_PROFILE, root);
    if (min > bestScore) {
      bestScore = min;
      bestRoot = root;
      bestMode = 'minor';
    }
  }

  return {
    label: `${NOTE_NAMES_ES[bestRoot]} ${bestMode === 'major' ? 'mayor' : 'menor'}`,
    root: bestRoot,
    mode: bestMode,
  };
}

export function estimateTempoBpm(onsetFlux: Float32Array, hopDurationSec: number): number {
  if (onsetFlux.length < 24) return 108;
  const minLag = Math.max(2, Math.round(60 / 164 / hopDurationSec));
  const maxLag = Math.min(Math.floor(onsetFlux.length / 2), Math.round(60 / 68 / hopDurationSec));

  let bestLag = Math.round(60 / 110 / hopDurationSec);
  let bestScore = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let corr = 0;
    const count = onsetFlux.length - lag;
    for (let i = 0; i < count; i++) {
      corr += onsetFlux[i] * onsetFlux[i + lag];
    }
    const normalized = corr / Math.max(1, count);
    const bpm = 60 / (lag * hopDurationSec);
    // Natural musical tempo prior centered around 112 BPM
    const prior = Math.exp(-0.5 * Math.pow((bpm - 112) / 36, 2));
    const score = normalized * (0.45 + 0.55 * prior);
    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }

  const estimated = Math.round(60 / (bestLag * hopDurationSec));
  return Math.max(68, Math.min(164, estimated));
}

/**
 * Estimates the downbeat phase offset (in frames) that maximizes alignment
 * of spectral onset flux with the beat grid.
 */
function estimateBeatPhaseFrames(onsetFlux: Float32Array, beatPeriodFrames: number): number {
  const period = Math.max(4, Math.round(beatPeriodFrames));
  let bestPhase = 0;
  let bestSum = -1;
  for (let p = 0; p < period; p++) {
    let sum = 0;
    for (let f = p; f < onsetFlux.length; f += period) {
      const local =
        onsetFlux[f] +
        0.5 * (f > 0 ? onsetFlux[f - 1] : 0) +
        0.5 * (f + 1 < onsetFlux.length ? onsetFlux[f + 1] : 0);
      sum += local;
    }
    if (sum > bestSum) {
      bestSum = sum;
      bestPhase = p;
    }
  }
  return bestPhase;
}

/**
 * Computes narrowband sinusoidal peak prominence at a target frequency inside an FFT magnitude
 * spectrum. Returns 0 for broadband drums/cymbals/noise and positive linear amplitude for
 * true pitched sinusoidal partials.
 */
function sampleSinusoidalProminence(
  mag: Float32Array,
  freqHz: number,
  sampleRate: number,
  fftSize: number,
): number {
  const halfN = mag.length - 1;
  const binExact = (freqHz * fftSize) / sampleRate;
  if (binExact < 3 || binExact > halfN - 4) return 0;

  // ±0.35 semitone search window around exact pitch
  const deltaSemitone = binExact * 0.059463; // 2^(1/12) - 1
  const searchRadius = Math.max(1, Math.min(3, Math.round(deltaSemitone * 0.42)));
  const c0 = Math.round(binExact);

  let peakVal = 0;
  let peakBin = c0;
  for (let k = Math.max(2, c0 - searchRadius); k <= Math.min(halfN - 2, c0 + searchRadius); k++) {
    if (mag[k] > peakVal) {
      peakVal = mag[k];
      peakBin = k;
    }
  }
  if (peakVal <= 1e-6) return 0;

  // Must be a local peak (or plateau) to reject spectral skirts from adjacent strong notes
  if (peakVal < mag[peakBin - 1] * 0.96 || peakVal < mag[peakBin + 1] * 0.96) {
    return 0;
  }

  // Local noise floor ±1.3 to ±2.4 semitones away
  const flankInner = Math.max(2, Math.round(deltaSemitone * 1.25));
  const flankOuter = Math.max(flankInner + 1, Math.round(deltaSemitone * 2.3));
  let floorSum = 0;
  let floorCnt = 0;
  for (let d = flankInner; d <= flankOuter; d++) {
    const lo = peakBin - d;
    const hi = peakBin + d;
    if (lo >= 1) {
      floorSum += mag[lo];
      floorCnt++;
    }
    if (hi <= halfN) {
      floorSum += mag[hi];
      floorCnt++;
    }
  }
  const floor = floorCnt > 0 ? floorSum / floorCnt : 0;
  const snr = peakVal / (floor + 1e-7);

  // Broadband noise / snares / hi-hats have SNR ~ 1.0..1.35 -> gated out cleanly
  if (snr < 1.38) return 0;
  const gate = Math.min(1, (snr - 1.38) / 1.45);
  return Math.max(0, peakVal - floor * 1.08) * gate;
}

/**
 * Estimates tuning deviation in cents (-45..+45) from A4 = 440 Hz using parabolic FFT peaks
 * across mid-register tonal energy.
 */
function estimateTuningRatio(
  midLevel1: Float32Array,
  sampleRate1: number,
  fftSize: number,
  hann: Float32Array,
): number {
  const n = midLevel1.length;
  if (n < fftSize) return 1.0;

  const real = new Float32Array(fftSize);
  const imag = new Float32Array(fftSize);
  const halfN = fftSize >> 1;
  const mag = new Float32Array(halfN);

  const hist = new Float64Array(21); // -50c to +50c in 5-cent bins
  const numSamples = Math.min(18, Math.max(4, Math.floor(n / fftSize)));
  const stride = Math.max(1, Math.floor((n - fftSize) / numSamples));

  for (let s = 0; s < numSamples; s++) {
    const off = s * stride;
    for (let i = 0; i < fftSize; i++) {
      real[i] = midLevel1[off + i] * hann[i];
      imag[i] = 0;
    }
    fftRadix2(real, imag);
    for (let k = 0; k < halfN; k++) {
      mag[k] = Math.hypot(real[k], imag[k]);
    }

    // Inspect 200 Hz .. 1200 Hz
    const kMin = Math.max(3, Math.floor((200 * fftSize) / sampleRate1));
    const kMax = Math.min(halfN - 3, Math.ceil((1200 * fftSize) / sampleRate1));
    for (let k = kMin; k <= kMax; k++) {
      const m1 = mag[k];
      if (m1 > mag[k - 1] && m1 > mag[k + 1] && m1 > mag[k - 2] * 1.8 && m1 > mag[k + 2] * 1.8) {
        // Parabolic peak interpolation
        const denom = mag[k - 1] - 2 * m1 + mag[k + 1];
        const delta = Math.abs(denom) > 1e-8 ? 0.5 * (mag[k - 1] - mag[k + 1]) / denom : 0;
        const exactFreq = ((k + Math.max(-0.5, Math.min(0.5, delta))) * sampleRate1) / fftSize;
        const midiFloat = 69 + 12 * Math.log2(exactFreq / 440);
        const cents = (midiFloat - Math.round(midiFloat)) * 100;
        const bin = Math.max(0, Math.min(20, Math.round((cents + 50) / 5)));
        hist[bin] += m1;
      }
    }
  }

  let bestBin = 10; // 0 cents default
  let bestWeight = 0;
  let totalWeight = 0;
  for (let b = 0; b < 21; b++) {
    totalWeight += hist[b];
    if (hist[b] > bestWeight) {
      bestWeight = hist[b];
      bestBin = b;
    }
  }
  if (totalWeight <= 1e-5 || bestWeight < totalWeight * 0.14) return 1.0;
  const estCents = Math.max(-40, Math.min(40, (bestBin - 10) * 5));
  return Math.pow(2, estCents / 1200);
}

/**
 * Adaptive rolling reference envelope: blends local ±2.2s energy with global 80th percentile
 * so both brickwall-mastered MP3s and dynamic piano ballads threshold accurately.
 */
function buildAdaptiveReference(frameMax: Float32Array, halfWin = 48): Float32Array {
  const n = frameMax.length;
  const globalRef = Math.max(1e-5, percentileOfPositive(frameMax, 0.8));
  const out = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    const a = Math.max(0, f - halfWin);
    const b = Math.min(n - 1, f + halfWin);
    let localPeak = 0;
    for (let k = a; k <= b; k++) {
      if (frameMax[k] > localPeak) localPeak = frameMax[k];
    }
    // Never drop below 32% of global reference so pure background hiss stays silent
    out[f] = Math.max(globalRef * 0.32, 0.52 * localPeak + 0.48 * globalRef);
  }
  return out;
}

/**
 * Monophonic Viterbi pitch contour tracker with adaptive per-frame silence gate,
 * natural melodic transition costs, and vibrato spike stabilization.
 */
function runAdaptiveViterbi(
  salience: Float32Array, // [numFrames * span]
  numFrames: number,
  span: number,
  silenceCurve: Float32Array,
  scaleCurve: Float32Array,
  stayBonusRatio: number,
  jumpPenaltyRatio: number,
  maxJumpSemitones = 10,
): Int16Array {
  const states = span + 1; // 0 = silent, 1..span = pitch index
  const dp = new Float32Array(numFrames * states);
  const backptr = new Int16Array(numFrames * states);

  for (let s = 0; s < states; s++) {
    dp[s] = s === 0 ? silenceCurve[0] : salience[s - 1];
  }

  for (let f = 1; f < numFrames; f++) {
    const prevBase = (f - 1) * states;
    const currBase = f * states;
    const salBase = f * span;
    const ref = scaleCurve[f];
    const stayBonus = stayBonusRatio * ref;
    const silentSwitchPenalty = -0.09 * ref;
    const jumpUnit = jumpPenaltyRatio * ref;

    for (let s = 0; s < states; s++) {
      const emission = s === 0 ? silenceCurve[f] : salience[salBase + (s - 1)];
      let bestPrev = 0;
      let bestScore = -Infinity;

      for (let p = 0; p < states; p++) {
        let trans = 0;
        if (s === p) {
          trans = stayBonus;
        } else if (s === 0 || p === 0) {
          trans = silentSwitchPenalty;
        } else {
          const jump = Math.abs(s - p);
          trans = -Math.min(jump, maxJumpSemitones) * jumpUnit;
        }
        const cand = dp[prevBase + p] + trans;
        if (cand > bestScore) {
          bestScore = cand;
          bestPrev = p;
        }
      }

      dp[currBase + s] = bestScore + emission;
      backptr[currBase + s] = bestPrev;
    }
  }

  const path = new Int16Array(numFrames);
  let lastState = 0;
  let lastBest = -Infinity;
  const lastBase = (numFrames - 1) * states;
  for (let s = 0; s < states; s++) {
    if (dp[lastBase + s] > lastBest) {
      lastBest = dp[lastBase + s];
      lastState = s;
    }
  }
  path[numFrames - 1] = lastState;
  for (let f = numFrames - 2; f >= 0; f--) {
    path[f] = backptr[(f + 1) * states + path[f + 1]];
  }

  // Two-pass median/spike cleanup to eliminate 1-frame vibrato blips
  const smoothed = new Int16Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    if (f > 0 && f < numFrames - 1 && path[f - 1] === path[f + 1] && path[f] !== path[f - 1]) {
      smoothed[f] = path[f - 1];
    } else {
      smoothed[f] = path[f];
    }
  }
  return smoothed;
}

interface ChordTemplate {
  id: number;
  root: number; // -1 for No-Chord
  quality: 'none' | 'major' | 'minor';
  pcs: number[]; // pitch classes (0..11)
  weights: Float32Array; // length 12 normalized vector
}

function buildChordTemplates(): ChordTemplate[] {
  const templates: ChordTemplate[] = [
    {
      id: 0,
      root: -1,
      quality: 'none',
      pcs: [],
      weights: new Float32Array(12),
    },
  ];
  let id = 1;
  for (let root = 0; root < 12; root++) {
    for (const quality of ['major', 'minor'] as const) {
      const third = (root + (quality === 'major' ? 4 : 3)) % 12;
      const fifth = (root + 7) % 12;
      const w = new Float32Array(12);
      w[root] = 1.0;
      w[third] = 0.85;
      w[fifth] = 0.78;
      // Mild negative penalty on clashing semitones to sharpen chord discrimination
      w[(root + 1) % 12] = -0.18;
      w[(root + 6) % 12] = -0.14;
      const norm = Math.sqrt(1.0 * 1.0 + 0.85 * 0.85 + 0.78 * 0.78);
      for (let i = 0; i < 12; i++) w[i] /= norm;
      templates.push({
        id: id++,
        root,
        quality,
        pcs: [root, third, fifth],
        weights: w,
      });
    }
  }
  return templates;
}

const CHORD_TEMPLATES = buildChordTemplates();

/**
 * Chooses the smoothest concert-piano voicing (3-note triad in C3..D5, MIDI 48..74)
 * using nearest voice-leading relative to the previous chord's pitch centroid.
 */
function voiceChordSmoothly(
  pcs: number[],
  prevMidis: number[] | null,
  avoidMidi: number,
  bassMidi: number,
): number[] {
  if (pcs.length === 0) return [];
  const targetCenter = prevMidis && prevMidis.length > 0
    ? prevMidis.reduce((s, m) => s + m, 0) / prevMidis.length
    : 58; // Bb3 — warm central piano accompaniment register

  // Generate candidate close-position inversions around C3..D5
  const candidates: number[][] = [];
  for (let oct = 3; oct <= 4; oct++) {
    for (let inv = 0; inv < pcs.length; inv++) {
      const notes: number[] = [];
      let prev = -1;
      for (let k = 0; k < pcs.length; k++) {
        const pc = pcs[(inv + k) % pcs.length];
        let m = 12 * (oct + 1) + pc;
        while (m <= prev) m += 12;
        if (m < 48) m += 12;
        if (m > 74) m -= 12;
        notes.push(m);
        prev = m;
      }
      notes.sort((a, b) => a - b);
      // Verify clean voicing (no semitone clusters, in range)
      if (notes[0] >= 48 && notes[notes.length - 1] <= 74) {
        candidates.push(notes);
      }
    }
  }

  if (candidates.length === 0) {
    return pcs.map((pc) => 60 + pc).filter((m) => m <= 74);
  }

  let best = candidates[0];
  let bestCost = Infinity;
  for (const cand of candidates) {
    const center = cand.reduce((s, m) => s + m, 0) / cand.length;
    let cost = Math.abs(center - targetCenter);
    if (prevMidis && prevMidis.length === cand.length) {
      let voiceMove = 0;
      for (let i = 0; i < cand.length; i++) {
        voiceMove += Math.abs(cand[i] - prevMidis[i]);
      }
      cost = voiceMove * 0.65 + Math.abs(center - 58) * 0.28;
    }
    // Prefer voicings that sit cleanly below/away from active vocal note
    if (avoidMidi >= 0) {
      for (const m of cand) {
        if (Math.abs(m - avoidMidi) <= 1) cost += 7;
        if (m > avoidMidi) cost += 2.5;
      }
    }
    if (cost < bestCost) {
      bestCost = cost;
      best = cand;
    }
  }

  // Final safeguard: drop any note clashing within ±1 semitone of vocal or duplicating bass
  return best.filter(
    (m) => (avoidMidi < 0 || Math.abs(m - avoidMidi) > 1) && m !== bassMidi,
  );
}

interface PyramidMatrices {
  numFrames: number;
  hopSec: number;
  duration: number;
  mixMat: Float32Array;
  centerMat: Float32Array;
  sideMat: Float32Array;
  onsetFlux: Float32Array;
  frameRms: Float32Array;
}

/**
 * Multi-Rate Constant-Q Pyramid + Linear Mid/Side Spectral Decomposition.
 * Uses 4 octave-matched 2048-point FFT levels so every piano note from E1 (41 Hz)
 * to C7 (2093 Hz) gets 2–6 FFT bins per semitone with zero nonlinear distortion.
 */
function* buildMultiRateTonalMatricesGen(
  leftChannel: Float32Array,
  rightChannel: Float32Array | null,
  sourceSampleRate: number,
  options: TranscriptionOptions,
): Generator<void, PyramidMatrices, void> {
  const targetSampleRate = options.targetSampleRate ?? 22050;
  const maxDurationSec = options.maxDurationSec ?? 210;
  const fftSize = 2048;
  const halfN = fftSize >> 1;
  const baseHop = 1024; // 46.44 ms @ 22050 Hz

  options.onProgress?.(0.05, 'Acondicionando audio estéreo y calibrando afinación...');

  const maxSamples = Math.min(leftChannel.length, Math.floor(maxDurationSec * sourceSampleRate));
  const leftTrim = leftChannel.subarray(0, maxSamples);
  const rightTrim = rightChannel ? rightChannel.subarray(0, maxSamples) : null;

  const midFull = new Float32Array(maxSamples);
  const sideFull = new Float32Array(maxSamples);
  const hasStereo = Boolean(rightTrim && rightTrim.length === maxSamples);

  if (hasStereo && rightTrim) {
    for (let i = 0; i < maxSamples; i++) {
      midFull[i] = 0.5 * (leftTrim[i] + rightTrim[i]);
      sideFull[i] = 0.5 * (leftTrim[i] - rightTrim[i]);
    }
  } else {
    midFull.set(leftTrim);
  }

  // Level 0: 22050 Hz (Octaves 5..7, MIDI >= 72)
  const midL0 = downsampleMono(midFull, sourceSampleRate, targetSampleRate);
  const sideL0 = hasStereo
    ? downsampleMono(sideFull, sourceSampleRate, targetSampleRate)
    : new Float32Array(midL0.length);

  yield;

  // Level 1: 11025 Hz (Octave 4, MIDI 60..71)
  const midL1 = decimateBy2(midL0);
  const sideL1 = hasStereo ? decimateBy2(sideL0) : new Float32Array(midL1.length);

  // Level 2: 5512.5 Hz (Octave 3, MIDI 48..59)
  const midL2 = decimateBy2(midL1);
  const sideL2 = hasStereo ? decimateBy2(sideL1) : new Float32Array(midL2.length);

  // Level 3: 2756.25 Hz (Octaves 1..2, MIDI 28..47) — 1.34 Hz bin width!
  const midL3 = decimateBy2(midL2);
  const sideL3 = hasStereo ? decimateBy2(sideL2) : new Float32Array(midL3.length);

  const levels = [
    { sr: targetSampleRate, hop: baseHop, mid: midL0, side: sideL0 },
    { sr: targetSampleRate / 2, hop: baseHop >> 1, mid: midL1, side: sideL1 },
    { sr: targetSampleRate / 4, hop: baseHop >> 2, mid: midL2, side: sideL2 },
    { sr: targetSampleRate / 8, hop: baseHop >> 3, mid: midL3, side: sideL3 },
  ];

  const duration = midL0.length / targetSampleRate;
  const hopSec = baseHop / targetSampleRate;
  const numFrames = Math.max(1, Math.floor((midL0.length - fftSize) / baseHop) + 1);
  const hann = createHannWindow(fftSize);

  const tuningRatio = estimateTuningRatio(midL1, levels[1].sr, fftSize, hann);

  // Precompute tuned frequencies and assigned octave level for each MIDI 28..96
  const noteFreqs = new Float32Array(ANALYSIS_NUM_NOTES);
  const noteLevel = new Uint8Array(ANALYSIS_NUM_NOTES);
  for (let i = 0; i < ANALYSIS_NUM_NOTES; i++) {
    const midi = ANALYSIS_MIN_MIDI + i;
    noteFreqs[i] = midiToFreq(midi) * tuningRatio;
    if (midi >= 72) noteLevel[i] = 0;
    else if (midi >= 60) noteLevel[i] = 1;
    else if (midi >= 48) noteLevel[i] = 2;
    else noteLevel[i] = 3;
  }

  const mixMat = new Float32Array(numFrames * ANALYSIS_NUM_NOTES);
  const centerMat = new Float32Array(numFrames * ANALYSIS_NUM_NOTES);
  const sideMat = new Float32Array(numFrames * ANALYSIS_NUM_NOTES);
  const onsetFlux = new Float32Array(numFrames);
  const frameRms = new Float32Array(numFrames);

  // Scratch FFT buffers (zero allocations inside frame loop)
  const realBuf = new Float32Array(fftSize);
  const imagBuf = new Float32Array(fftSize);
  const midMags = [
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
  ];
  const sideMags = [
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
  ];
  const mixMags = [
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
  ];
  const centerMags = [
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
  ];
  const sideOnlyMags = [
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
    new Float32Array(halfN),
  ];
  const prevFluxMag = new Float32Array(halfN);

  const computeFftMag = (sig: Float32Array, centerSample: number, outMag: Float32Array): number => {
    const start = centerSample - (fftSize >> 1);
    let sq = 0;
    for (let i = 0; i < fftSize; i++) {
      const idx = start + i;
      const s = idx >= 0 && idx < sig.length ? sig[idx] : 0;
      sq += s * s;
      realBuf[i] = s * hann[i];
      imagBuf[i] = 0;
    }
    fftRadix2(realBuf, imagBuf);
    const invNorm = 2 / fftSize;
    for (let k = 0; k < halfN; k++) {
      outMag[k] = Math.hypot(realBuf[k], imagBuf[k]) * invNorm;
    }
    return Math.sqrt(sq / fftSize);
  };

  for (let f = 0; f < numFrames; f++) {
    if (f % 48 === 0) {
      options.onProgress?.(
        0.08 + 0.44 * (f / numFrames),
        'Calculando pirámide armónica Constant-Q (4 octavas)...',
      );
      yield;
    }

    const centerTimeSec = (f * baseHop + (fftSize >> 1)) / targetSampleRate;

    // Level 0 RMS & spectral onset flux
    const c0 = Math.round(centerTimeSec * levels[0].sr);
    const rms = computeFftMag(levels[0].mid, c0, midMags[0]);
    frameRms[f] = rms;

    let flux = 0;
    for (let k = 4; k < halfN; k++) {
      const cur = Math.log1p(midMags[0][k] * 40);
      const diff = cur - prevFluxMag[k];
      if (diff > 0) flux += diff;
      prevFluxMag[k] = cur;
    }
    onsetFlux[f] = flux;

    if (rms < 0.004) continue;

    // Run remaining levels + stereo side spectra linearly (zero waveshaping distortion!)
    for (let lv = 0; lv < 4; lv++) {
      const cSample = Math.round(centerTimeSec * levels[lv].sr);
      if (lv > 0) {
        computeFftMag(levels[lv].mid, cSample, midMags[lv]);
      }
      if (hasStereo) {
        computeFftMag(levels[lv].side, cSample, sideMags[lv]);
      } else {
        sideMags[lv].fill(0);
      }

      const mm = midMags[lv];
      const sm = sideMags[lv];
      const mx = mixMags[lv];
      const cm = centerMags[lv];
      const so = sideOnlyMags[lv];

      for (let k = 0; k < halfN; k++) {
        const mVal = mm[k];
        const sVal = sm[k];
        mx[k] = hasStereo ? Math.hypot(mVal, sVal) : mVal;
        // Linear frequency-domain center isolation removes stereo guitars/reverb cleanly
        cm[k] = hasStereo ? Math.max(0, mVal - 0.8 * sVal) : mVal;
        so[k] = hasStereo ? Math.max(0, sVal - 0.35 * mVal) : mVal * 0.55;
      }
    }

    const rowBase = f * ANALYSIS_NUM_NOTES;
    for (let i = 0; i < ANALYSIS_NUM_NOTES; i++) {
      const lv = noteLevel[i];
      const freq = noteFreqs[i];
      const sr = levels[lv].sr;

      mixMat[rowBase + i] = sampleSinusoidalProminence(mixMags[lv], freq, sr, fftSize);
      const midi = ANALYSIS_MIN_MIDI + i;
      if (midi >= 50 && midi <= 86) {
        centerMat[rowBase + i] = sampleSinusoidalProminence(centerMags[lv], freq, sr, fftSize);
      }
      if (midi >= 60 && midi <= 94) {
        sideMat[rowBase + i] = sampleSinusoidalProminence(sideOnlyMags[lv], freq, sr, fftSize);
      }
    }
  }

  return {
    numFrames,
    hopSec,
    duration,
    mixMat,
    centerMat,
    sideMat,
    onsetFlux,
    frameRms,
  };
}

/**
 * Shared post-pyramid musical extraction pipeline:
 * HPSS -> Key & Tempo/Beat Grid -> Beat-Synced Chord HMM -> Bass -> Vocal -> Voice-Led Chords -> Lead.
 */
function* extractFourTracksFromMatricesGen(
  pyr: PyramidMatrices,
  title: string,
  options: TranscriptionOptions,
): Generator<void, TranscribedPiece, void> {
  const { numFrames, hopSec, duration, onsetFlux } = pyr;

  options.onProgress?.(0.56, 'Eliminando batería, platillos y ruido de fondo (HPSS)...');
  yield;

  const cleanMix = applyHarmonicPercussiveMask(pyr.mixMat, numFrames, ANALYSIS_NUM_NOTES, 4, 5);
  const cleanCenter = applyHarmonicPercussiveMask(pyr.centerMat, numFrames, ANALYSIS_NUM_NOTES, 3, 5);
  const cleanSide = applyHarmonicPercussiveMask(pyr.sideMat, numFrames, ANALYSIS_NUM_NOTES, 3, 5);

  yield;

  // Tempo, Beat Grid & Global Musical Key
  const bpm = estimateTempoBpm(onsetFlux, hopSec);
  const beatSec = 60 / bpm;
  const beatPeriodFrames = Math.max(4, Math.round(beatSec / hopSec));
  const beatPhase = estimateBeatPhaseFrames(onsetFlux, beatPeriodFrames);

  const globalChroma = new Float64Array(12);
  for (let f = 0; f < numFrames; f++) {
    const rowBase = f * ANALYSIS_NUM_NOTES;
    // Focus chroma on MIDI 36 (C2) .. 84 (C6) where tonal harmony lives
    for (let m = 36; m <= 84; m++) {
      const e = cleanMix[rowBase + (m - ANALYSIS_MIN_MIDI)];
      if (e > 0) globalChroma[m % 12] += e;
    }
  }
  const detectedKey = estimateKeyFromChroma(globalChroma);

  options.onProgress?.(0.68, 'Reconociendo progresión de acordes y línea de bajo...');
  yield;

  // =========================================================================
  // STEP A: BEAT-SYNCHRONIZED CHORD PROGRESSION HMM (24 Major/Minor + N)
  // =========================================================================
  const halfBeatFrames = Math.max(3, Math.round(beatPeriodFrames / 2));
  const segStarts: number[] = [];
  if (beatPhase > 2) segStarts.push(0);
  for (let f = beatPhase; f < numFrames; f += halfBeatFrames) {
    segStarts.push(f);
  }
  if (segStarts.length === 0) segStarts.push(0);

  const numSegs = segStarts.length;
  const segEnergies = new Float32Array(numSegs);
  const segChromas: Float32Array[] = new Array(numSegs);

  for (let s = 0; s < numSegs; s++) {
    const fStart = segStarts[s];
    const fEnd = s + 1 < numSegs ? segStarts[s + 1] : numFrames;
    const chroma = new Float32Array(12);
    let eSum = 0;
    const len = Math.max(1, fEnd - fStart);

    for (let f = fStart; f < fEnd; f++) {
      const rowBase = f * ANALYSIS_NUM_NOTES;
      for (let m = 36; m <= 80; m++) {
        const idx = m - ANALYSIS_MIN_MIDI;
        // Weight mid-low accompaniment slightly higher than high vocal partials
        const regWeight = m <= 67 ? 1.15 : 0.72;
        const val = cleanMix[rowBase + idx] * regWeight;
        if (val > 0) {
          chroma[m % 12] += val;
          eSum += val;
        }
      }
    }
    segEnergies[s] = eSum / len;
    segChromas[s] = chroma;
  }

  const segRefEnergy = Math.max(1e-5, percentileOfPositive(segEnergies, 0.78));
  const numTemplates = CHORD_TEMPLATES.length; // 25 (0 = N, 1..24 = chords)
  const chordDp = new Float32Array(numSegs * numTemplates);
  const chordBack = new Int16Array(numSegs * numTemplates);
  const segBestCosine = new Float32Array(numSegs);

  // Diatonic chord check helper
  const isDiatonicChord = (tpl: ChordTemplate): boolean => {
    if (tpl.pcs.length === 0) return false;
    return tpl.pcs.every((pc) => isPitchInKey(pc, detectedKey.root, detectedKey.mode));
  };

  for (let s = 0; s < numSegs; s++) {
    const ch = segChromas[s];
    let norm = 0;
    for (let i = 0; i < 12; i++) norm += ch[i] * ch[i];
    norm = Math.sqrt(norm);
    const isSilent = segEnergies[s] < segRefEnergy * 0.16 || norm < 1e-6;

    const emissions = new Float32Array(numTemplates);
    let bestCos = 0;
    for (let t = 0; t < numTemplates; t++) {
      const tpl = CHORD_TEMPLATES[t];
      if (t === 0) {
        emissions[0] = isSilent ? 0.95 : 0.36;
      } else if (isSilent) {
        emissions[t] = 0.0;
      } else {
        let dot = 0;
        for (let i = 0; i < 12; i++) dot += (ch[i] / norm) * tpl.weights[i];
        if (dot > bestCos) bestCos = dot;
        const diatonicBoost = isDiatonicChord(tpl) ? 0.11 : 0;
        emissions[t] = Math.max(0, dot) + diatonicBoost;
      }
    }
    segBestCosine[s] = bestCos;

    if (s === 0) {
      for (let t = 0; t < numTemplates; t++) chordDp[t] = emissions[t];
    } else {
      const prevBase = (s - 1) * numTemplates;
      const currBase = s * numTemplates;
      for (let t = 0; t < numTemplates; t++) {
        let bestPrev = 0;
        let bestScore = -Infinity;
        for (let p = 0; p < numTemplates; p++) {
          // Strong self-transition keeps chords steady across musical bars
          const trans = t === p ? 0.34 : t === 0 || p === 0 ? -0.05 : -0.14;
          const cand = chordDp[prevBase + p] + trans;
          if (cand > bestScore) {
            bestScore = cand;
            bestPrev = p;
          }
        }
        chordDp[currBase + t] = bestScore + emissions[t];
        chordBack[currBase + t] = bestPrev;
      }
    }
  }

  const segChords = new Int16Array(numSegs);
  {
    let bestLast = 0;
    let bestVal = -Infinity;
    const base = (numSegs - 1) * numTemplates;
    for (let t = 0; t < numTemplates; t++) {
      if (chordDp[base + t] > bestVal) {
        bestVal = chordDp[base + t];
        bestLast = t;
      }
    }
    segChords[numSegs - 1] = bestLast;
    for (let s = numSegs - 2; s >= 0; s--) {
      segChords[s] = chordBack[(s + 1) * numTemplates + segChords[s + 1]];
    }
  }

  // Map frame -> active chord template index
  const frameChord = new Int16Array(numFrames);
  for (let s = 0; s < numSegs; s++) {
    const fStart = segStarts[s];
    const fEnd = s + 1 < numSegs ? segStarts[s + 1] : numFrames;
    for (let f = fStart; f < fEnd; f++) frameChord[f] = segChords[s];
  }

  const avgFlux = onsetFlux.reduce((acc, v) => acc + v, 0) / Math.max(1, numFrames);
  let noteId = 1;

  // =========================================================================
  // TRACK 1: BASSLINE (Decimated Sub-1Hz CQT + Chord-Anchored Viterbi, E1..E3)
  // =========================================================================
  const bassMin = 28; // E1
  const bassMax = 52; // E3
  const bassSpan = bassMax - bassMin + 1;
  const bassSalience = new Float32Array(numFrames * bassSpan);
  const bassFrameMax = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    const rowBase = f * ANALYSIS_NUM_NOTES;
    const tpl = CHORD_TEMPLATES[frameChord[f]];
    let fMax = 0;

    for (let m = bassMin; m <= bassMax; m++) {
      const idx = m - ANALYSIS_MIN_MIDI;
      const f0 = cleanMix[rowBase + idx];
      if (f0 <= 1e-5) continue;

      // Require fundamental + harmonic reinforcement (rejects kick drums & ghost overtones)
      const h2 = idx + 12 < ANALYSIS_NUM_NOTES ? cleanMix[rowBase + idx + 12] : 0;
      const h3 = idx + 19 < ANALYSIS_NUM_NOTES ? cleanMix[rowBase + idx + 19] : 0;
      const harmRatio = Math.min(1, (h2 + 0.55 * h3) / (f0 + 1e-6));
      let score = f0 * (0.52 + 0.48 * harmRatio);

      // Sub-harmonic guard: if octave below has strong fundamental, penalize upper octave
      if (idx - 12 >= 0 && cleanMix[rowBase + idx - 12] > f0 * 0.72) {
        score *= 0.45;
      }

      // Gentle chord-root anchor keeps bassline rock-solid on musical harmony
      const pc = m % 12;
      if (tpl.root >= 0) {
        if (pc === tpl.root) score *= 1.24;
        else if (tpl.pcs.includes(pc)) score *= 1.07;
        else if (!isPitchInKey(m, detectedKey.root, detectedKey.mode)) score *= 0.78;
      }

      bassSalience[f * bassSpan + (m - bassMin)] = score;
      if (score > fMax) fMax = score;
    }
    bassFrameMax[f] = fMax;
  }

  const bassRef = buildAdaptiveReference(bassFrameMax, 48);
  const bassSilence = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f++) bassSilence[f] = bassRef[f] * 0.28;

  const bassPath = runAdaptiveViterbi(
    bassSalience,
    numFrames,
    bassSpan,
    bassSilence,
    bassRef,
    0.09,
    0.025,
    10,
  );

  const bassNotes: PianoNoteEvent[] = [];
  {
    let activeMidi = -1;
    let startF = 0;
    let peakNorm = 0;
    let sumNorm = 0;

    const flush = (endF: number) => {
      if (activeMidi < 0) return;
      const frames = endF - startF;
      const dur = frames * hopSec;
      if (dur >= 0.14) {
        const meanN = sumNorm / Math.max(1, frames);
        const conf = Math.max(0.15, Math.min(0.99, peakNorm * 0.62 + meanN * 0.38));
        bassNotes.push({
          id: `b-${noteId++}`,
          midi: activeMidi,
          startTime: Number((startF * hopSec).toFixed(3)),
          duration: Number(Math.min(3.8, dur).toFixed(3)),
          velocity: Number(Math.max(0.42, Math.min(0.84, 0.34 + conf * 0.5)).toFixed(2)),
          track: 'bass',
          confidence: Number(conf.toFixed(2)),
          inKey: isPitchInKey(activeMidi, detectedKey.root, detectedKey.mode),
        });
      }
      activeMidi = -1;
    };

    for (let f = 0; f < numFrames; f++) {
      const st = bassPath[f];
      const midi = st === 0 ? -1 : bassMin + (st - 1);
      const rawE = st === 0 ? 0 : bassSalience[f * bassSpan + (st - 1)];
      const normE = rawE / Math.max(1e-5, bassRef[f]);
      const reArticulate =
        midi >= 0 &&
        midi === activeMidi &&
        f - startF >= Math.max(5, Math.round(beatPeriodFrames * 0.7)) &&
        onsetFlux[f] > avgFlux * 2.15 &&
        normE > 0.55;

      if (midi !== activeMidi || reArticulate) {
        flush(f);
        if (midi >= 0 && rawE >= bassSilence[f] * 0.9) {
          activeMidi = midi;
          startF = f;
          peakNorm = Math.min(1, normE);
          sumNorm = Math.min(1, normE);
        }
      } else if (midi >= 0) {
        const c = Math.min(1, normE);
        if (c > peakNorm) peakNorm = c;
        sumNorm += c;
      }
    }
    flush(numFrames);
  }

  options.onProgress?.(0.79, 'Extrayendo melodía vocal principal (Viterbi F0)...');
  yield;

  // =========================================================================
  // TRACK 2: VOCAL MELODY (Linear Center-Isolated Harmonic-Product F0, E3..C6)
  // =========================================================================
  const vocalMin = 52; // E3
  const vocalMax = 84; // C6
  const vocalSpan = vocalMax - vocalMin + 1;
  const vocalSalience = new Float32Array(numFrames * vocalSpan);
  const vocalFrameMax = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    const rowBase = f * ANALYSIS_NUM_NOTES;
    const bassMidi = bassPath[f] === 0 ? -1 : bassMin + (bassPath[f] - 1);
    let fMax = 0;

    for (let m = vocalMin; m <= vocalMax; m++) {
      const idx = m - ANALYSIS_MIN_MIDI;
      const f0 = cleanCenter[rowBase + idx];
      if (f0 <= 1e-5) continue;

      const h2 = idx + 12 < ANALYSIS_NUM_NOTES ? cleanCenter[rowBase + idx + 12] : 0;
      const h3 = idx + 19 < ANALYSIS_NUM_NOTES ? cleanCenter[rowBase + idx + 19] : 0;
      const h4 = idx + 24 < ANALYSIS_NUM_NOTES ? cleanCenter[rowBase + idx + 24] : 0;
      const subOct = idx - 12 >= 0 ? cleanCenter[rowBase + idx - 12] : 0;

      // Geometric-arithmetic harmonic product: requires REAL fundamental + rewards harmonic overtones
      const overtoneSupport = Math.min(1.15, (h2 * 0.85 + h3 * 0.5 + h4 * 0.25) / (f0 + 1e-6));
      let score = f0 * (0.48 + 0.52 * overtoneSupport);

      // Strong octave-error suppression: if m - 12 is already a real strong fundamental,
      // prevent the tracker from jumping up to the 2nd harmonic!
      if (subOct > f0 * 0.62) {
        score *= 0.35;
      }
      // Prevent bass harmonics from leaking into lower vocal register
      if (bassMidi >= 0 && (m === bassMidi + 12 || m === bassMidi + 19)) {
        score *= 0.55;
      }

      // Natural human singing tessitura preference (A3..G5)
      const tessitura = m >= 57 && m <= 79 ? 1.15 : 0.92;
      const scaleWeight = isPitchInKey(m, detectedKey.root, detectedKey.mode) ? 1.1 : 0.84;
      score *= tessitura * scaleWeight;

      vocalSalience[f * vocalSpan + (m - vocalMin)] = score;
      if (score > fMax) fMax = score;
    }
    vocalFrameMax[f] = fMax;
  }

  const vocalRef = buildAdaptiveReference(vocalFrameMax, 48);
  const vocalSilence = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f++) vocalSilence[f] = vocalRef[f] * 0.31;

  const vocalPath = runAdaptiveViterbi(
    vocalSalience,
    numFrames,
    vocalSpan,
    vocalSilence,
    vocalRef,
    0.075,
    0.022,
    9,
  );

  const vocalNotes: PianoNoteEvent[] = [];
  {
    let activeMidi = -1;
    let startF = 0;
    let peakNorm = 0;
    let sumNorm = 0;
    let prevEnergy = 0;

    const flush = (endF: number) => {
      if (activeMidi < 0) return;
      const frames = endF - startF;
      const dur = frames * hopSec;
      if (dur >= 0.12) {
        const meanN = sumNorm / Math.max(1, frames);
        const conf = Math.max(0.15, Math.min(0.99, peakNorm * 0.65 + meanN * 0.35));
        vocalNotes.push({
          id: `v-${noteId++}`,
          midi: activeMidi,
          startTime: Number((startF * hopSec).toFixed(3)),
          duration: Number(Math.min(3.6, dur).toFixed(3)),
          velocity: Number(Math.max(0.54, Math.min(0.98, 0.44 + conf * 0.54)).toFixed(2)),
          track: 'vocal',
          confidence: Number(conf.toFixed(2)),
          inKey: isPitchInKey(activeMidi, detectedKey.root, detectedKey.mode),
        });
      }
      activeMidi = -1;
    };

    for (let f = 0; f < numFrames; f++) {
      const st = vocalPath[f];
      const midi = st === 0 ? -1 : vocalMin + (st - 1);
      const rawE = st === 0 ? 0 : vocalSalience[f * vocalSpan + (st - 1)];
      const normE = rawE / Math.max(1e-5, vocalRef[f]);

      // Syllable re-articulation when vocal envelope dips and rises sharply on same pitch
      const syllableRise =
        midi >= 0 &&
        midi === activeMidi &&
        f - startF >= 5 &&
        rawE > prevEnergy * 1.65 &&
        onsetFlux[f] > avgFlux * 1.95;

      if (midi !== activeMidi || syllableRise) {
        flush(f);
        if (midi >= 0 && rawE >= vocalSilence[f] * 0.88) {
          activeMidi = midi;
          startF = f;
          peakNorm = Math.min(1, normE);
          sumNorm = Math.min(1, normE);
        }
      } else if (midi >= 0) {
        const c = Math.min(1, normE);
        if (c > peakNorm) peakNorm = c;
        sumNorm += c;
      }
      prevEnergy = rawE;
    }
    flush(numFrames);
  }

  options.onProgress?.(0.89, 'Construyendo acordes limpios y arreglos estéreo...');
  yield;

  // =========================================================================
  // TRACK 3: VOICE-LED CONCERT PIANO CHORDS (HMM Harmony + Real Rhythm)
  // =========================================================================
  const chordNotes: PianoNoteEvent[] = [];
  let prevVoicing: number[] | null = null;
  let lastChordId = -1;
  let lastStrikeSeg = -99;

  for (let s = 0; s < numSegs; s++) {
    const chordId = segChords[s];
    if (chordId === 0) {
      lastChordId = 0;
      continue;
    }

    const fStart = segStarts[s];
    const fEnd = s + 1 < numSegs ? segStarts[s + 1] : numFrames;
    const tpl = CHORD_TEMPLATES[chordId];
    const energyNorm = Math.min(1, segEnergies[s] / Math.max(1e-5, segRefEnergy));
    if (energyNorm < 0.18) continue;

    // Measure peak onset inside first two frames of this half-beat
    let segFlux = 0;
    for (let f = fStart; f < Math.min(fEnd, fStart + 3); f++) {
      if (onsetFlux[f] > segFlux) segFlux = onsetFlux[f];
    }

    const chordChanged = chordId !== lastChordId;
    const isDownbeatStrike = (s - lastStrikeSeg) >= 2 && segFlux > avgFlux * 1.15;
    const isSustainedPulse = (s - lastStrikeSeg) >= 4 && energyNorm > 0.38;

    if (!chordChanged && !isDownbeatStrike && !isSustainedPulse) {
      continue;
    }

    // Determine sustain length until next chord change or up to 2 full beats
    let endSeg = s + 1;
    while (
      endSeg < numSegs &&
      segChords[endSeg] === chordId &&
      endSeg - s < 4 &&
      segEnergies[endSeg] >= segRefEnergy * 0.14
    ) {
      endSeg++;
    }
    const endFrame = endSeg < numSegs ? segStarts[endSeg] : numFrames;
    const durSec = Math.max(0.22, Math.min(2.8, (endFrame - fStart) * hopSec * 0.94));

    // Find active vocal & bass at strike frame so chords never clash with the melody
    const vocalAtStrike = vocalPath[fStart] === 0 ? -1 : vocalMin + (vocalPath[fStart] - 1);
    const bassAtStrike = bassPath[fStart] === 0 ? -1 : bassMin + (bassPath[fStart] - 1);

    const voiced = voiceChordSmoothly(tpl.pcs, prevVoicing, vocalAtStrike, bassAtStrike);
    if (voiced.length > 0) {
      prevVoicing = voiced;
      lastChordId = chordId;
      lastStrikeSeg = s;

      const matchQuality = Math.max(0.35, Math.min(1, segBestCosine[s]));
      const baseConf = Math.max(0.3, Math.min(0.98, matchQuality * 0.65 + energyNorm * 0.35));
      const baseVel = Math.max(0.34, Math.min(0.76, 0.28 + energyNorm * 0.48));
      const startTime = Number((fStart * hopSec).toFixed(3));

      for (let vIdx = 0; vIdx < voiced.length; vIdx++) {
        const midi = voiced[vIdx];
        // Slight confidence taper on 3rd voice so Strict cleanliness keeps ultra-pure 2-3 voice voicing
        const voiceConf = Math.max(0.25, Math.min(0.99, baseConf - (vIdx === 2 ? 0.06 : 0)));
        chordNotes.push({
          id: `c-${noteId++}`,
          midi,
          startTime,
          duration: Number(durSec.toFixed(3)),
          velocity: Number((baseVel * (vIdx === 0 ? 1.0 : 0.92)).toFixed(2)),
          track: 'chords',
          confidence: Number(voiceConf.toFixed(2)),
          inKey: isPitchInKey(midi, detectedKey.root, detectedKey.mode),
        });
      }
    }
  }

  // =========================================================================
  // TRACK 4: STEREO LEAD / ARPEGGIOS (Harmonic-Subtracted Side Detail, D4..G#6)
  // =========================================================================
  const leadMin = 62; // D4
  const leadMax = 92; // G#6
  const leadSpan = leadMax - leadMin + 1;
  const leadSalience = new Float32Array(numFrames * leadSpan);
  const leadFrameMax = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    const rowBase = f * ANALYSIS_NUM_NOTES;
    const vocalMidi = vocalPath[f] === 0 ? -1 : vocalMin + (vocalPath[f] - 1);
    const bassMidi = bassPath[f] === 0 ? -1 : bassMin + (bassPath[f] - 1);
    let fMax = 0;

    for (let m = leadMin; m <= leadMax; m++) {
      // Klapuri-style cancellation: exclude vocal, ±1 semitone clash, and vocal/bass harmonics
      if (vocalMidi >= 0) {
        const dv = Math.abs(m - vocalMidi);
        if (dv <= 1 || m === vocalMidi + 12 || m === vocalMidi + 19) continue;
      }
      if (bassMidi >= 0 && (m === bassMidi + 24 || m === bassMidi + 31)) continue;

      const idx = m - ANALYSIS_MIN_MIDI;
      const f0 = cleanSide[rowBase + idx];
      if (f0 <= 1e-5) continue;

      const h2 = idx + 12 < ANALYSIS_NUM_NOTES ? cleanSide[rowBase + idx + 12] : 0;
      const subOct = idx - 12 >= 0 ? cleanSide[rowBase + idx - 12] : 0;
      let score = f0 * (0.62 + 0.38 * Math.min(1, h2 / (f0 + 1e-6)));
      if (subOct > f0 * 0.7) score *= 0.4;
      score *= isPitchInKey(m, detectedKey.root, detectedKey.mode) ? 1.15 : 0.65;

      leadSalience[f * leadSpan + (m - leadMin)] = score;
      if (score > fMax) fMax = score;
    }
    leadFrameMax[f] = fMax;
  }

  const leadRef = buildAdaptiveReference(leadFrameMax, 48);
  const leadSilence = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f++) leadSilence[f] = leadRef[f] * 0.42;

  const leadPath = runAdaptiveViterbi(
    leadSalience,
    numFrames,
    leadSpan,
    leadSilence,
    leadRef,
    0.06,
    0.025,
    9,
  );

  const leadNotes: PianoNoteEvent[] = [];
  {
    let activeMidi = -1;
    let startF = 0;
    let peakNorm = 0;
    let sumNorm = 0;

    const flush = (endF: number) => {
      if (activeMidi < 0) return;
      const frames = endF - startF;
      const dur = frames * hopSec;
      if (dur >= 0.13) {
        const meanN = sumNorm / Math.max(1, frames);
        const conf = Math.max(0.15, Math.min(0.95, peakNorm * 0.65 + meanN * 0.35));
        leadNotes.push({
          id: `l-${noteId++}`,
          midi: activeMidi,
          startTime: Number((startF * hopSec).toFixed(3)),
          duration: Number(Math.min(2.6, dur).toFixed(3)),
          velocity: Number(Math.max(0.32, Math.min(0.74, 0.26 + conf * 0.48)).toFixed(2)),
          track: 'lead',
          confidence: Number(conf.toFixed(2)),
          inKey: isPitchInKey(activeMidi, detectedKey.root, detectedKey.mode),
        });
      }
      activeMidi = -1;
    };

    for (let f = 0; f < numFrames; f++) {
      const st = leadPath[f];
      const midi = st === 0 ? -1 : leadMin + (st - 1);
      const rawE = st === 0 ? 0 : leadSalience[f * leadSpan + (st - 1)];
      const normE = rawE / Math.max(1e-5, leadRef[f]);

      if (midi !== activeMidi) {
        flush(f);
        if (midi >= 0 && rawE >= leadSilence[f] * 0.92) {
          activeMidi = midi;
          startF = f;
          peakNorm = Math.min(1, normE);
          sumNorm = Math.min(1, normE);
        }
      } else if (midi >= 0) {
        const c = Math.min(1, normE);
        if (c > peakNorm) peakNorm = c;
        sumNorm += c;
      }
    }
    flush(numFrames);
  }

  const allNotes = [...vocalNotes, ...bassNotes, ...chordNotes, ...leadNotes].sort(
    (a, b) => a.startTime - b.startTime || a.midi - b.midi,
  );

  options.onProgress?.(1, 'Partitura limpia y pistas MIDI listas');

  return {
    id: `upload-${Date.now()}`,
    title,
    subtitle: 'MP3 transcrito con Constant-Q y acordes armónicos',
    source: 'upload',
    duration: Number(duration.toFixed(2)),
    bpm,
    keySignature: detectedKey.label,
    keyRoot: detectedKey.root,
    keyMode: detectedKey.mode,
    notes: allNotes,
    countsByTrack: {
      vocal: vocalNotes.length,
      chords: chordNotes.length,
      bass: bassNotes.length,
      lead: leadNotes.length,
    },
  };
}

function* runTranscriptionPipelineGen(
  leftChannel: Float32Array,
  rightChannel: Float32Array | null,
  sourceSampleRate: number,
  title: string,
  options: TranscriptionOptions,
): Generator<void, TranscribedPiece, void> {
  const pyr = yield* buildMultiRateTonalMatricesGen(
    leftChannel,
    rightChannel,
    sourceSampleRate,
    options,
  );
  return yield* extractFourTracksFromMatricesGen(pyr, title, options);
}

/**
 * Non-blocking async MP3-to-4-Track MIDI transcription engine for browser usage.
 * Yields smoothly to the event loop between frame chunks so the UI never freezes.
 */
export async function transcribePcmToPianoAndVocalAsync(
  leftChannel: Float32Array,
  rightChannel: Float32Array | null,
  sourceSampleRate: number,
  title: string,
  options: TranscriptionOptions = {},
): Promise<TranscribedPiece> {
  const gen = runTranscriptionPipelineGen(
    leftChannel,
    rightChannel,
    sourceSampleRate,
    title,
    options,
  );
  let step = gen.next();
  while (!step.done) {
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    step = gen.next();
  }
  return step.value;
}

/**
 * Synchronous entry point (identical algorithm to async path; ideal for unit tests).
 */
export function transcribePcmToPianoAndVocal(
  leftChannel: Float32Array,
  rightChannel: Float32Array | null,
  sourceSampleRate: number,
  title: string,
  options: TranscriptionOptions = {},
): TranscribedPiece {
  const gen = runTranscriptionPipelineGen(
    leftChannel,
    rightChannel,
    sourceSampleRate,
    title,
    options,
  );
  let step = gen.next();
  while (!step.done) {
    step = gen.next();
  }
  return step.value;
}

/**
 * Live non-destructive noise cleaner & instrument-track filter.
 * Lets the user adjust cleanliness (Suave / Media / Alta) and toggle any of the
 * 4 instruments instantaneously without re-running CQT analysis.
 */
export function filterAndCleanNotes(
  piece: TranscribedPiece,
  enabledTracks: Record<TrackRole, boolean>,
  cleanliness: CleanlinessLevel,
): PianoNoteEvent[] {
  if (piece.source === 'demo' || piece.source === 'midi') {
    return piece.notes.filter((n) => enabledTracks[n.track]);
  }

  // Solo piano recordings (ByteDance CRNN & Basic Pitch Piano) need full two-handed polyphony,
  // bass octaves, fast 16th/32nd runs, and unrestricted chromatic/jazz/classical accidentals.
  if (piece.transcriptionMode === 'piano-hd' || piece.transcriptionMode === 'piano') {
    const pianoCfg =
      cleanliness === 'strict'
        ? { minConf: 0.44, minDur: 0.075, maxSimulChords: 6 }
        : cleanliness === 'balanced'
          ? { minConf: 0.28, minDur: 0.055, maxSimulChords: 8 }
          : { minConf: 0.14, minDur: 0.042, maxSimulChords: 10 };

    const kept: PianoNoteEvent[] = [];
    for (const n of piece.notes) {
      if (!enabledTracks[n.track]) continue;
      if (n.confidence < pianoCfg.minConf) continue;
      if (n.duration < pianoCfg.minDur) continue;

      if (n.track === 'chords') {
        let activeChords = 0;
        for (let i = kept.length - 1; i >= 0; i--) {
          const prev = kept[i];
          if (prev.startTime < n.startTime - 4.0) break;
          if (prev.startTime + prev.duration <= n.startTime + 0.02) continue;
          if (prev.track === 'chords') activeChords++;
        }
        if (activeChords >= pianoCfg.maxSimulChords) continue;
      }

      kept.push(n);
    }
    return kept;
  }

  const cfg =
    cleanliness === 'strict'
      ? { minConf: 0.52, minDur: 0.14, requireInKeyBelowConf: 0.8, maxSimulChords: 3, minGap: 0.06 }
      : cleanliness === 'balanced'
        ? { minConf: 0.36, minDur: 0.12, requireInKeyBelowConf: 0.65, maxSimulChords: 3, minGap: 0.04 }
        : { minConf: 0.18, minDur: 0.08, requireInKeyBelowConf: 0.0, maxSimulChords: 4, minGap: 0.0 };

  const kept: PianoNoteEvent[] = [];
  const lastTrackStart: Record<TrackRole, number> = {
    vocal: -99,
    bass: -99,
    chords: -99,
    lead: -99,
  };

  for (const n of piece.notes) {
    if (!enabledTracks[n.track]) continue;
    if (n.confidence < cfg.minConf) continue;
    if (n.duration < cfg.minDur) continue;
    if (!n.inKey && n.confidence < cfg.requireInKeyBelowConf) continue;

    if (n.track !== 'chords' && n.startTime - lastTrackStart[n.track] < cfg.minGap) {
      continue;
    }

    if (n.track === 'chords') {
      let activeChords = 0;
      let clashes = false;
      for (let i = kept.length - 1; i >= 0; i--) {
        const prev = kept[i];
        if (prev.startTime < n.startTime - 3.0) break;
        if (prev.startTime + prev.duration <= n.startTime + 0.02) continue;
        if (prev.track === 'chords') {
          activeChords++;
          if (Math.abs(prev.midi - n.midi) <= 1) clashes = true;
        }
        if (prev.track === 'vocal' && Math.abs(prev.midi - n.midi) <= 1 && cleanliness !== 'soft') {
          clashes = true;
        }
      }
      if (clashes || activeChords >= cfg.maxSimulChords) continue;
    }

    lastTrackStart[n.track] = n.startTime;
    kept.push(n);
  }

  return kept;
}

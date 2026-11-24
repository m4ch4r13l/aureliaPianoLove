import { MAX_MIDI, MIN_MIDI, midiToFreq } from './pianoTheory';

export type TrackRole = 'vocal' | 'chords' | 'bass' | 'lead';

export type CleanlinessLevel = 'soft' | 'balanced' | 'strict';

export interface PianoNoteEvent {
  id: string;
  midi: number;
  startTime: number;
  duration: number;
  velocity: number;
  track: TrackRole;
  /** Spectral tonal confidence (0..1) used for instant live noise filtering */
  confidence: number;
  /** Whether this pitch belongs to the estimated musical key scale */
  inKey: boolean;
}

export interface TranscribedPiece {
  id: string;
  title: string;
  subtitle: string;
  source: 'demo' | 'upload';
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
  fftSize?: number;
  hopSize?: number;
  onProgress?: (progress: number, stage: string) => void;
}

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

export function downsampleMono(
  input: Float32Array,
  fromRate: number,
  toRate: number,
): Float32Array {
  if (toRate >= fromRate) return input;
  const ratio = fromRate / toRate;
  const outLength = Math.floor(input.length / ratio);
  const out = new Float32Array(outLength);

  for (let i = 0; i < outLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    const count = Math.max(1, end - start);
    for (let k = start; k < end; k++) {
      sum += input[k];
    }
    out[i] = sum / count;
  }
  return out;
}

/**
 * Extracts a center-panned vocal-emphasized signal and a side-panned instrumental
 * signal (where stereo guitars, pads, and arpeggiated synths live).
 */
export function splitMidSideSignals(
  left: Float32Array,
  right: Float32Array | null,
  sampleRate: number,
): { vocal: Float32Array; side: Float32Array } {
  const n = left.length;
  const vocal = new Float32Array(n);
  const side = new Float32Array(n);

  if (right && right.length === n) {
    for (let i = 0; i < n; i++) {
      const l = left[i];
      const r = right[i];
      const mid = 0.5 * (l + r);
      const diff = 0.5 * (l - r);
      const midAbs = Math.abs(mid);
      const diffAbs = Math.abs(diff);
      const centerMask = midAbs / (midAbs + diffAbs * 1.45 + 1e-6);
      vocal[i] = mid * (0.2 + 0.8 * centerMask);
      side[i] = diff;
    }
  } else {
    vocal.set(left);
    side.set(left);
  }

  // Gentle high-pass (~170 Hz) on vocal signal to strip kick/bass bleed
  const rc = 1 / (2 * Math.PI * 170);
  const dt = 1 / sampleRate;
  const alpha = rc / (rc + dt);
  let prevIn = 0;
  let prevOut = 0;
  for (let i = 0; i < n; i++) {
    const x = vocal[i];
    const y = alpha * (prevOut + x - prevIn);
    prevIn = x;
    prevOut = y;
    vocal[i] = y;
  }

  return { vocal, side };
}

function createHannWindow(size: number): Float32Array {
  const win = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return win;
}

function medianOfSmall(arr: Float32Array, len: number): number {
  // Insertion sort on tiny scratch buffer (len <= 13) — zero allocations
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

/**
 * Harmonic-Percussive Source Separation (HPSS) on a Semitone Energy Matrix [frames x 88].
 * Strips broadband drum hits, snares, hi-hats, and clicks using horizontal (time) vs
 * vertical (pitch) median filtering + Wiener soft harmonic masking.
 */
export function applyHarmonicPercussiveMask(
  matrix: Float32Array,
  numFrames: number,
  numBins: number,
  timeHalfWin = 4,
  pitchHalfWin = 4,
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

      // Temporal median -> sustained harmonic tone strength
      let ti = 0;
      for (let t = tStart; t <= tEnd; t++) {
        scratchT[ti++] = matrix[t * numBins + b];
      }
      const hEst = medianOfSmall(scratchT, tLen);

      // Pitch/frequency median -> broadband percussive / noise floor across nearby semitones
      const bStart = Math.max(0, b - pitchHalfWin);
      const bEnd = Math.min(numBins - 1, b + pitchHalfWin);
      const bLen = bEnd - bStart + 1;
      let bi = 0;
      for (let k = bStart; k <= bEnd; k++) {
        scratchF[bi++] = matrix[rowBase + k];
      }
      const pEst = medianOfSmall(scratchF, bLen);

      // Wiener harmonic mask + spectral floor subtraction
      const h2 = hEst * hEst;
      const p2 = (pEst * 1.18) * (pEst * 1.18);
      const harmonicMask = h2 / (h2 + p2 + 1e-8);
      const aboveFloor = Math.max(0, val - pEst * 0.92);

      cleaned[rowBase + b] = aboveFloor * harmonicMask;
    }
  }

  return cleaned;
}

function sampleSpectralPeak(
  mag: Float32Array,
  freqHz: number,
  sampleRate: number,
  fftSize: number,
): number {
  const binFloat = (freqHz * fftSize) / sampleRate;
  const center = Math.round(binFloat);
  const halfBins = mag.length - 1;
  if (center <= 2 || center >= halfBins - 2) return 0;

  const m0 = mag[center - 1];
  const m1 = mag[center];
  const m2 = mag[center + 1];
  const peak = Math.max(m0, m1, m2);
  // Local noise floor 3 bins away to reject flat broadband static/cymbals
  const floor = 0.5 * (mag[Math.max(1, center - 3)] + mag[Math.min(halfBins, center + 3)]);
  return Math.max(0, peak - floor * 0.55);
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
const MINOR_SCALE_INTERVALS = new Set([0, 2, 3, 5, 7, 8, 10, 11]); // natural + harmonic leading tone

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
  if (onsetFlux.length < 20) return 100;
  const minLag = Math.max(2, Math.round(60 / 168 / hopDurationSec));
  const maxLag = Math.min(Math.floor(onsetFlux.length / 2), Math.round(60 / 64 / hopDurationSec));

  let bestLag = Math.round(60 / 108 / hopDurationSec);
  let bestCorr = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let corr = 0;
    const count = onsetFlux.length - lag;
    for (let i = 0; i < count; i++) {
      corr += onsetFlux[i] * onsetFlux[i + lag];
    }
    const normalized = corr / Math.max(1, count);
    const bpm = 60 / (lag * hopDurationSec);
    const prior = 1 - (0.14 * Math.abs(bpm - 108)) / 60;
    if (normalized * prior > bestCorr) {
      bestCorr = normalized * prior;
      bestLag = lag;
    }
  }

  const estimated = Math.round(60 / (bestLag * hopDurationSec));
  return Math.max(64, Math.min(168, estimated));
}

/**
 * Monophonic Viterbi pitch tracker with pitch-continuity smoothing and vibrato stabilization.
 * Used cleanly for both Vocal Melody (Voz) and Bassline (Bajo).
 */
function runViterbiMonophonicTrack(
  salienceMatrix: Float32Array, // [numFrames * span]
  numFrames: number,
  minMidi: number,
  maxMidi: number,
  silenceThreshold: number,
  maxEnergy: number,
  stayBonusRatio: number,
  jumpPenaltyRatio: number,
): Int16Array {
  const span = maxMidi - minMidi + 1;
  const states = span + 1; // 0 = silent, 1..span = minMidi + (s - 1)
  const dp = new Float32Array(numFrames * states);
  const backptr = new Int16Array(numFrames * states);

  for (let s = 0; s < states; s++) {
    dp[s] = s === 0 ? silenceThreshold : salienceMatrix[s - 1];
  }

  for (let f = 1; f < numFrames; f++) {
    const prevBase = (f - 1) * states;
    const currBase = f * states;
    const salBase = f * span;

    for (let s = 0; s < states; s++) {
      const emission = s === 0 ? silenceThreshold : salienceMatrix[salBase + (s - 1)];
      let bestPrev = 0;
      let bestScore = -Infinity;

      for (let p = 0; p < states; p++) {
        let trans = 0;
        if (s === p) {
          trans = stayBonusRatio * maxEnergy;
        } else if (s === 0 || p === 0) {
          trans = -0.07 * maxEnergy;
        } else {
          const jump = Math.abs(s - p);
          trans = -Math.min(jump, 12) * jumpPenaltyRatio * maxEnergy;
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

  // Median-of-3 spike cleanup to eliminate 1-frame pitch flickers / vibrato jitter
  const smoothed = new Int16Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    if (f === 0 || f === numFrames - 1) {
      smoothed[f] = path[f];
    } else if (path[f - 1] === path[f + 1] && path[f] !== path[f - 1]) {
      smoothed[f] = path[f - 1];
    } else {
      smoothed[f] = path[f];
    }
  }

  return smoothed;
}

/**
 * Pure PCM-to-4-Stem MIDI Transcription Engine:
 * Separates an MP3 into 4 clean, HPSS-denoised musical tracks:
 *   1. `vocal`  (Melodía vocal)
 *   2. `bass`   (Bajo monofónico)
 *   3. `chords` (Acordes / Piano armónico)
 *   4. `lead`   (Arreglos / Guitarra / Teclados agudos)
 */
export function transcribePcmToPianoAndVocal(
  leftChannel: Float32Array,
  rightChannel: Float32Array | null,
  sourceSampleRate: number,
  title: string,
  options: TranscriptionOptions = {},
): TranscribedPiece {
  const targetSampleRate = options.targetSampleRate ?? 22050;
  const fftSize = options.fftSize ?? 4096;
  const hopSize = options.hopSize ?? 1024;
  const maxDurationSec = options.maxDurationSec ?? 210;

  options.onProgress?.(0.05, 'Separando canal central (voz) y estéreo (instrumentos)...');

  const maxSamples = Math.min(leftChannel.length, Math.floor(maxDurationSec * sourceSampleRate));
  const leftTrimmed = leftChannel.subarray(0, maxSamples);
  const rightTrimmed = rightChannel ? rightChannel.subarray(0, maxSamples) : null;

  const monoMix = new Float32Array(maxSamples);
  if (rightTrimmed) {
    for (let i = 0; i < maxSamples; i++) {
      monoMix[i] = 0.5 * (leftTrimmed[i] + rightTrimmed[i]);
    }
  } else {
    monoMix.set(leftTrimmed);
  }

  const { vocal: vocalRaw, side: sideRaw } = splitMidSideSignals(
    leftTrimmed,
    rightTrimmed,
    sourceSampleRate,
  );

  const monoDs = downsampleMono(monoMix, sourceSampleRate, targetSampleRate);
  const vocalDs = downsampleMono(vocalRaw, sourceSampleRate, targetSampleRate);
  const sideDs = downsampleMono(sideRaw, sourceSampleRate, targetSampleRate);

  const duration = monoDs.length / targetSampleRate;
  const hopSec = hopSize / targetSampleRate;
  const numFrames = Math.max(1, Math.floor((monoDs.length - fftSize) / hopSize) + 1);

  const hann = createHannWindow(fftSize);
  const halfN = fftSize >> 1;
  const realBuf = new Float32Array(fftSize);
  const imagBuf = new Float32Array(fftSize);
  const magMono = new Float32Array(halfN);
  const magVocal = new Float32Array(halfN);
  const magSide = new Float32Array(halfN);
  const prevMagMono = new Float32Array(halfN);
  const onsetFlux = new Float32Array(numFrames);

  const NUM_NOTES = MAX_MIDI - MIN_MIDI + 1; // 88
  const noteFreqs = new Float32Array(NUM_NOTES);
  for (let i = 0; i < NUM_NOTES; i++) {
    noteFreqs[i] = midiToFreq(MIN_MIDI + i);
  }

  // Raw 88-semitone harmonic salience matrices for full mix, center vocal, and stereo side
  const rawMonoMat = new Float32Array(numFrames * NUM_NOTES);
  const rawVocalMat = new Float32Array(numFrames * NUM_NOTES);
  const rawSideMat = new Float32Array(numFrames * NUM_NOTES);

  const runFftIntoMag = (signal: Float32Array, offset: number, outMag: Float32Array): number => {
    let rms = 0;
    for (let i = 0; i < fftSize; i++) {
      const s = offset + i < signal.length ? signal[offset + i] : 0;
      rms += s * s;
      realBuf[i] = s * hann[i];
      imagBuf[i] = 0;
    }
    fftRadix2(realBuf, imagBuf);
    for (let k = 0; k < halfN; k++) {
      const m = Math.hypot(realBuf[k], imagBuf[k]) / (fftSize * 0.25);
      outMag[k] = Math.log1p(m * 16);
    }
    return Math.sqrt(rms / fftSize);
  };

  for (let f = 0; f < numFrames; f++) {
    if (f % 45 === 0) {
      options.onProgress?.(
        0.08 + 0.42 * (f / numFrames),
        'Extrayendo espectro armónico por bandas...',
      );
    }

    const offset = f * hopSize;
    const monoRms = runFftIntoMag(monoDs, offset, magMono);

    let flux = 0;
    for (let k = 0; k < halfN; k++) {
      const diff = magMono[k] - prevMagMono[k];
      if (diff > 0) flux += diff;
      prevMagMono[k] = magMono[k];
    }
    onsetFlux[f] = flux;

    if (monoRms < 0.008) continue;

    runFftIntoMag(vocalDs, offset, magVocal);
    runFftIntoMag(sideDs, offset, magSide);

    const rowBase = f * NUM_NOTES;
    for (let idx = 7; idx <= 75; idx++) {
      // MIDI 28 (E1) .. 96 (C7)
      const f0 = noteFreqs[idx];

      // Full mix harmonic comb (for bass & piano chords)
      const m1 = sampleSpectralPeak(magMono, f0, targetSampleRate, fftSize);
      if (m1 > 0.015) {
        const m2 = sampleSpectralPeak(magMono, f0 * 2, targetSampleRate, fftSize);
        const m3 = sampleSpectralPeak(magMono, f0 * 3, targetSampleRate, fftSize);
        const m4 = sampleSpectralPeak(magMono, f0 * 4, targetSampleRate, fftSize);
        const mSub = sampleSpectralPeak(magMono, f0 * 0.5, targetSampleRate, fftSize);
        rawMonoMat[rowBase + idx] = Math.max(
          0,
          m1 * 1.0 + m2 * 0.52 + m3 * 0.26 + m4 * 0.12 - mSub * 0.32,
        );
      }

      // Vocal center-channel harmonic comb (MIDI 52..84)
      const midi = MIN_MIDI + idx;
      if (midi >= 50 && midi <= 84) {
        const v1 = sampleSpectralPeak(magVocal, f0, targetSampleRate, fftSize);
        if (v1 > 0.015) {
          const v2 = sampleSpectralPeak(magVocal, f0 * 2, targetSampleRate, fftSize);
          const v3 = sampleSpectralPeak(magVocal, f0 * 3, targetSampleRate, fftSize);
          const v4 = sampleSpectralPeak(magVocal, f0 * 4, targetSampleRate, fftSize);
          const vSub = sampleSpectralPeak(magVocal, f0 * 0.5, targetSampleRate, fftSize);
          const tessitura = midi >= 57 && midi <= 79 ? 1.18 : 0.94;
          rawVocalMat[rowBase + idx] =
            Math.max(0, v1 * 1.0 + v2 * 0.62 + v3 * 0.34 + v4 * 0.15 - vSub * 0.42) * tessitura;
        }
      }

      // Stereo side / upper instrumental detail (MIDI 60..92)
      if (midi >= 60 && midi <= 92) {
        const s1 = sampleSpectralPeak(magSide, f0, targetSampleRate, fftSize);
        if (s1 > 0.015) {
          const s2 = sampleSpectralPeak(magSide, f0 * 2, targetSampleRate, fftSize);
          const s3 = sampleSpectralPeak(magSide, f0 * 3, targetSampleRate, fftSize);
          const sSub = sampleSpectralPeak(magSide, f0 * 0.5, targetSampleRate, fftSize);
          rawSideMat[rowBase + idx] = Math.max(0, s1 + s2 * 0.5 + s3 * 0.24 - sSub * 0.35);
        }
      }
    }
  }

  options.onProgress?.(0.55, 'Eliminando percusión y ruido de fondo (HPSS)...');

  // Apply Harmonic-Percussive Source Separation to strip drums, snares, hi-hats & clicks!
  const cleanMonoMat = applyHarmonicPercussiveMask(rawMonoMat, numFrames, NUM_NOTES, 4, 4);
  const cleanVocalMat = applyHarmonicPercussiveMask(rawVocalMat, numFrames, NUM_NOTES, 3, 4);
  const cleanSideMat = applyHarmonicPercussiveMask(rawSideMat, numFrames, NUM_NOTES, 3, 4);

  // Estimate global musical key from HPSS-cleaned harmonic chroma BEFORE selecting chord voices
  const globalChroma = new Float64Array(12);
  for (let f = 0; f < numFrames; f++) {
    const rowBase = f * NUM_NOTES;
    for (let idx = 7; idx <= 75; idx++) {
      const e = cleanMonoMat[rowBase + idx];
      if (e > 0) {
        globalChroma[(MIN_MIDI + idx) % 12] += e;
      }
    }
  }
  const detectedKey = estimateKeyFromChroma(globalChroma);

  options.onProgress?.(0.70, 'Separando pistas: Voz, Bajo, Acordes y Arreglos...');

  const avgFlux = onsetFlux.reduce((acc, v) => acc + v, 0) / Math.max(1, numFrames);
  let noteIdCounter = 1;

  // =========================================================================
  // TRACK 1: VOCAL MELODY (Monophonic Center-Weighted Viterbi, MIDI 52..84)
  // =========================================================================
  const vocalMin = 52; // E3
  const vocalMax = 84; // C6
  const vocalSpan = vocalMax - vocalMin + 1;
  const vocalSlice = new Float32Array(numFrames * vocalSpan);
  let maxVocalE = 1e-6;

  for (let f = 0; f < numFrames; f++) {
    for (let m = vocalMin; m <= vocalMax; m++) {
      const raw = cleanVocalMat[f * NUM_NOTES + (m - MIN_MIDI)];
      // Slight gentle preference for notes inside detected scale without blocking real blue notes
      const scaleWeight = isPitchInKey(m, detectedKey.root, detectedKey.mode) ? 1.08 : 0.88;
      const val = raw * scaleWeight;
      vocalSlice[f * vocalSpan + (m - vocalMin)] = val;
      if (val > maxVocalE) maxVocalE = val;
    }
  }

  const vocalSilence = maxVocalE * 0.30;
  const vocalPath = runViterbiMonophonicTrack(
    vocalSlice,
    numFrames,
    vocalMin,
    vocalMax,
    vocalSilence,
    maxVocalE,
    0.055,
    0.018,
  );

  const vocalNotes: PianoNoteEvent[] = [];
  {
    let activeMidi = -1;
    let startFrame = 0;
    let peakE = 0;
    let sumE = 0;

    const flush = (endFrame: number) => {
      if (activeMidi < 0) return;
      const frames = endFrame - startFrame;
      const dur = frames * hopSec;
      if (dur >= 0.11) {
        const meanNorm = sumE / Math.max(1, frames) / maxVocalE;
        const peakNorm = peakE / maxVocalE;
        const conf = Math.max(0.1, Math.min(1, peakNorm * 0.65 + meanNorm * 0.35));
        vocalNotes.push({
          id: `v-${noteIdCounter++}`,
          midi: activeMidi,
          startTime: Number((startFrame * hopSec).toFixed(3)),
          duration: Number(Math.min(3.8, dur).toFixed(3)),
          velocity: Number(Math.max(0.48, Math.min(0.98, 0.38 + conf * 0.62)).toFixed(2)),
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
      const energy = st === 0 ? 0 : vocalSlice[f * vocalSpan + (st - 1)];
      const reArticulate =
        midi >= 0 &&
        midi === activeMidi &&
        f - startFrame >= 5 &&
        onsetFlux[f] > avgFlux * 2.35 &&
        energy > peakE * 0.78;

      if (midi !== activeMidi || reArticulate) {
        flush(f);
        if (midi >= 0 && energy >= vocalSilence * 0.85) {
          activeMidi = midi;
          startFrame = f;
          peakE = energy;
          sumE = energy;
        }
      } else if (midi >= 0) {
        if (energy > peakE) peakE = energy;
        sumE += energy;
      }
    }
    flush(numFrames);
  }

  // =========================================================================
  // TRACK 2: BASSLINE (Monophonic Low Register Viterbi, MIDI 28..54)
  // =========================================================================
  const bassMin = 28; // E1
  const bassMax = 54; // F#3
  const bassSpan = bassMax - bassMin + 1;
  const bassSlice = new Float32Array(numFrames * bassSpan);
  let maxBassE = 1e-6;

  for (let f = 0; f < numFrames; f++) {
    for (let m = bassMin; m <= bassMax; m++) {
      const raw = cleanMonoMat[f * NUM_NOTES + (m - MIN_MIDI)];
      const scaleWeight = isPitchInKey(m, detectedKey.root, detectedKey.mode) ? 1.12 : 0.82;
      const val = raw * scaleWeight;
      bassSlice[f * bassSpan + (m - bassMin)] = val;
      if (val > maxBassE) maxBassE = val;
    }
  }

  const bassSilence = maxBassE * 0.30;
  const bassPath = runViterbiMonophonicTrack(
    bassSlice,
    numFrames,
    bassMin,
    bassMax,
    bassSilence,
    maxBassE,
    0.075,
    0.022,
  );

  const bassNotes: PianoNoteEvent[] = [];
  {
    let activeMidi = -1;
    let startFrame = 0;
    let peakE = 0;
    let sumE = 0;

    const flush = (endFrame: number) => {
      if (activeMidi < 0) return;
      const frames = endFrame - startFrame;
      const dur = frames * hopSec;
      if (dur >= 0.14) {
        const conf = Math.max(
          0.1,
          Math.min(1, (peakE / maxBassE) * 0.6 + (sumE / Math.max(1, frames) / maxBassE) * 0.4),
        );
        bassNotes.push({
          id: `b-${noteIdCounter++}`,
          midi: activeMidi,
          startTime: Number((startFrame * hopSec).toFixed(3)),
          duration: Number(Math.min(4.0, dur).toFixed(3)),
          velocity: Number(Math.max(0.42, Math.min(0.86, 0.34 + conf * 0.52)).toFixed(2)),
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
      const energy = st === 0 ? 0 : bassSlice[f * bassSpan + (st - 1)];
      const reArticulate =
        midi >= 0 &&
        midi === activeMidi &&
        f - startFrame >= 6 &&
        onsetFlux[f] > avgFlux * 2.45 &&
        energy > peakE * 0.8;

      if (midi !== activeMidi || reArticulate) {
        flush(f);
        if (midi >= 0 && energy >= bassSilence * 0.85) {
          activeMidi = midi;
          startFrame = f;
          peakE = energy;
          sumE = energy;
        }
      } else if (midi >= 0) {
        if (energy > peakE) peakE = energy;
        sumE += energy;
      }
    }
    flush(numFrames);
  }

  options.onProgress?.(0.84, 'Limpiando acordes armónicos y arreglos agudos...');

  // =========================================================================
  // TRACK 3: HARMONIC CHORDS / PIANO (Consonant Max-3 Voices, MIDI 48..76)
  // =========================================================================
  const chordMin = 48; // C3
  const chordMax = 76; // E5
  const chordFramePicks: Array<Array<{ midi: number; strength: number }>> = new Array(numFrames);
  let maxChordE = 1e-6;
  const workScores = new Float32Array(NUM_NOTES);

  for (let f = 0; f < numFrames; f++) {
    const rowBase = f * NUM_NOTES;
    const vocalMidi = vocalPath[f] === 0 ? -1 : vocalMin + (vocalPath[f] - 1);
    const bassMidi = bassPath[f] === 0 ? -1 : bassMin + (bassPath[f] - 1);

    workScores.fill(0);
    for (let m = chordMin; m <= chordMax; m++) {
      const idx = m - MIN_MIDI;
      let s = cleanMonoMat[rowBase + idx];
      if (s <= 0.01) continue;
      // Scale preference reduces out-of-key muddy overtones dramatically
      s *= isPitchInKey(m, detectedKey.root, detectedKey.mode) ? 1.15 : 0.65;
      workScores[idx] = s;
    }

    // Attenuate harmonics cast upward by the bass note so bass overtones don't fake chords
    if (bassMidi >= 0) {
      const bassHarmonics = [12, 19, 24, 28];
      const damp = [0.42, 0.38, 0.45, 0.55];
      for (let h = 0; h < bassHarmonics.length; h++) {
        const hi = bassMidi + bassHarmonics[h] - MIN_MIDI;
        if (hi >= 0 && hi < NUM_NOTES) workScores[hi] *= damp[h];
      }
    }
    // Avoid duplicating exact vocal note or ±1 semitone clash against the singing melody
    if (vocalMidi >= 0) {
      const vi = vocalMidi - MIN_MIDI;
      workScores[vi] = 0;
      if (vi - 1 >= 0) workScores[vi - 1] *= 0.25;
      if (vi + 1 < NUM_NOTES) workScores[vi + 1] *= 0.25;
    }
    if (bassMidi >= 0) {
      workScores[bassMidi - MIN_MIDI] = 0;
    }

    const picks: Array<{ midi: number; strength: number }> = [];
    for (let p = 0; p < 3; p++) {
      let bestIdx = -1;
      let bestVal = 0;
      for (let m = chordMin; m <= chordMax; m++) {
        const idx = m - MIN_MIDI;
        const val = workScores[idx];
        if (val > bestVal) {
          const leftV = idx > 0 ? workScores[idx - 1] : 0;
          const rightV = idx < NUM_NOTES - 1 ? workScores[idx + 1] : 0;
          if (val >= leftV && val >= rightV) {
            bestVal = val;
            bestIdx = idx;
          }
        }
      }
      if (bestIdx < 0 || bestVal < 0.05) break;

      const pickedMidi = MIN_MIDI + bestIdx;
      picks.push({ midi: pickedMidi, strength: bestVal });
      if (bestVal > maxChordE) maxChordE = bestVal;

      // Eliminate picked note, adjacent minor-2nd semitone clashes, and its overtones
      workScores[bestIdx] = 0;
      if (bestIdx - 1 >= 0) workScores[bestIdx - 1] = 0;
      if (bestIdx + 1 < NUM_NOTES) workScores[bestIdx + 1] = 0;
      for (const offset of [12, 19, 24]) {
        const hi = bestIdx + offset;
        if (hi < NUM_NOTES) workScores[hi] *= 0.3;
      }
    }
    chordFramePicks[f] = picks;
  }

  const chordThreshold = maxChordE * 0.30;
  const chordNotes: PianoNoteEvent[] = [];
  {
    const active = new Map<number, { start: number; last: number; peak: number; sum: number; cnt: number }>();

    const flush = (midi: number, v: { start: number; last: number; peak: number; sum: number; cnt: number }) => {
      const dur = (v.last - v.start + 1) * hopSec;
      if (dur >= 0.14) {
        const conf = Math.max(
          0.1,
          Math.min(1, (v.peak / maxChordE) * 0.65 + (v.sum / Math.max(1, v.cnt) / maxChordE) * 0.35),
        );
        chordNotes.push({
          id: `c-${noteIdCounter++}`,
          midi,
          startTime: Number((v.start * hopSec).toFixed(3)),
          duration: Number(Math.min(3.5, dur + hopSec * 0.4).toFixed(3)),
          velocity: Number(Math.max(0.32, Math.min(0.80, 0.25 + conf * 0.55)).toFixed(2)),
          track: 'chords',
          confidence: Number(conf.toFixed(2)),
          inKey: isPitchInKey(midi, detectedKey.root, detectedKey.mode),
        });
      }
    };

    for (let f = 0; f < numFrames; f++) {
      const seen = new Set<number>();
      for (const pk of chordFramePicks[f]) {
        if (pk.strength < chordThreshold) continue;
        seen.add(pk.midi);
        const ex = active.get(pk.midi);
        if (ex) {
          const reStrike =
            f - ex.start >= 6 && onsetFlux[f] > avgFlux * 2.5 && pk.strength > ex.peak * 0.85;
          if (reStrike) {
            flush(pk.midi, ex);
            active.set(pk.midi, { start: f, last: f, peak: pk.strength, sum: pk.strength, cnt: 1 });
          } else {
            ex.last = f;
            ex.sum += pk.strength;
            ex.cnt++;
            if (pk.strength > ex.peak) ex.peak = pk.strength;
          }
        } else {
          active.set(pk.midi, { start: f, last: f, peak: pk.strength, sum: pk.strength, cnt: 1 });
        }
      }

      for (const [m, v] of active.entries()) {
        if (!seen.has(m) && f - v.last >= 2) {
          flush(m, v);
          active.delete(m);
        }
      }
    }
    for (const [m, v] of active.entries()) flush(m, v);
  }

  // =========================================================================
  // TRACK 4: UPPER LEAD / GUITAR / ARPEGGIOS (Stereo Side + Upper Detail, MIDI 62..92)
  // =========================================================================
  const leadMin = 62; // D4
  const leadMax = 92; // G#6
  const leadSpan = leadMax - leadMin + 1;
  const leadSlice = new Float32Array(numFrames * leadSpan);
  let maxLeadE = 1e-6;

  for (let f = 0; f < numFrames; f++) {
    const vocalMidi = vocalPath[f] === 0 ? -1 : vocalMin + (vocalPath[f] - 1);
    const chordSet = new Set(chordFramePicks[f].map((c) => c.midi));
    for (let m = leadMin; m <= leadMax; m++) {
      if (m === vocalMidi || Math.abs(m - vocalMidi) === 1) continue;
      if (chordSet.has(m)) continue;
      const raw = cleanSideMat[f * NUM_NOTES + (m - MIN_MIDI)];
      if (raw <= 0.015) continue;
      const val = raw * (isPitchInKey(m, detectedKey.root, detectedKey.mode) ? 1.15 : 0.65);
      leadSlice[f * leadSpan + (m - leadMin)] = val;
      if (val > maxLeadE) maxLeadE = val;
    }
  }

  const leadSilence = maxLeadE * 0.36;
  const leadPath = runViterbiMonophonicTrack(
    leadSlice,
    numFrames,
    leadMin,
    leadMax,
    leadSilence,
    maxLeadE,
    0.05,
    0.02,
  );

  const leadNotes: PianoNoteEvent[] = [];
  {
    let activeMidi = -1;
    let startFrame = 0;
    let peakE = 0;
    let sumE = 0;

    const flush = (endFrame: number) => {
      if (activeMidi < 0) return;
      const frames = endFrame - startFrame;
      const dur = frames * hopSec;
      if (dur >= 0.12) {
        const conf = Math.max(
          0.1,
          Math.min(1, (peakE / maxLeadE) * 0.65 + (sumE / Math.max(1, frames) / maxLeadE) * 0.35),
        );
        leadNotes.push({
          id: `l-${noteIdCounter++}`,
          midi: activeMidi,
          startTime: Number((startFrame * hopSec).toFixed(3)),
          duration: Number(Math.min(2.8, dur).toFixed(3)),
          velocity: Number(Math.max(0.32, Math.min(0.76, 0.25 + conf * 0.5)).toFixed(2)),
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
      const energy = st === 0 ? 0 : leadSlice[f * leadSpan + (st - 1)];
      if (midi !== activeMidi) {
        flush(f);
        if (midi >= 0 && energy >= leadSilence * 0.9) {
          activeMidi = midi;
          startFrame = f;
          peakE = energy;
          sumE = energy;
        }
      } else if (midi >= 0) {
        if (energy > peakE) peakE = energy;
        sumE += energy;
      }
    }
    flush(numFrames);
  }

  const allNotes = [...vocalNotes, ...bassNotes, ...chordNotes, ...leadNotes].sort(
    (a, b) => a.startTime - b.startTime || a.midi - b.midi,
  );

  const bpm = estimateTempoBpm(onsetFlux, hopSec);
  options.onProgress?.(1, 'Partitura limpia y pistas separadas listas');

  return {
    id: `upload-${Date.now()}`,
    title,
    subtitle: 'MP3 separado en 4 pistas MIDI limpias',
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

/**
 * Live non-destructive noise cleaner & instrument-track filter.
 * Lets the user adjust cleanliness (Suave / Media / Alta) and toggle any of the
 * 4 instruments instantaneously without re-running FFT analysis.
 */
export function filterAndCleanNotes(
  piece: TranscribedPiece,
  enabledTracks: Record<TrackRole, boolean>,
  cleanliness: CleanlinessLevel,
): PianoNoteEvent[] {
  // Demo scores are already hand-composed cleanly — only filter by enabledTracks
  if (piece.source === 'demo') {
    return piece.notes.filter((n) => enabledTracks[n.track]);
  }

  const cfg =
    cleanliness === 'strict'
      ? { minConf: 0.56, minDur: 0.16, requireInKeyBelowConf: 0.84, maxSimulChords: 2, minGap: 0.08 }
      : cleanliness === 'balanced'
        ? { minConf: 0.42, minDur: 0.13, requireInKeyBelowConf: 0.72, maxSimulChords: 3, minGap: 0.05 }
        : { minConf: 0.22, minDur: 0.09, requireInKeyBelowConf: 0.0, maxSimulChords: 4, minGap: 0.0 };

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

    // Prevent rapid machine-gun re-triggers on vocal/bass/lead
    if (n.track !== 'chords' && n.startTime - lastTrackStart[n.track] < cfg.minGap) {
      continue;
    }

    // Cap simultaneous chord cluster density and prevent semitone clashes in strict/balanced modes
    if (n.track === 'chords') {
      let activeChords = 0;
      let clashes = false;
      for (let i = kept.length - 1; i >= 0; i--) {
        const prev = kept[i];
        if (prev.startTime < n.startTime - 3.6) break;
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

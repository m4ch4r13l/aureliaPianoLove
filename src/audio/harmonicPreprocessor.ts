import { midiToFreq } from './pianoTheory';
import { fftRadix2 } from './pitchDsp';

export const PREPROCESS_SAMPLE_RATE = 22050;
const STFT_SIZE = 2048;
const STFT_HOP = 512; // 75% overlap -> exact COLA constant = 1.5 with periodic Hann WOLA
const MIN_MIDI_PROFILE = 21;
const NUM_PIANO_KEYS = 88;

/**
 * Per-frame stereo center-panning profile across the 88 piano pitches.
 * Values near 1.0 indicate dead-center harmonic energy (typically lead vocals/bass),
 * while values near 0.0..0.35 indicate wide stereo-panned accompaniment.
 */
export interface StereoCenterProfile {
  isStereo: boolean;
  hopSec: number;
  numFrames: number;
  /** Flattened [numFrames * 88] matrix of center dominance scores (0..1) */
  scores: Float32Array;
}

export interface HarmonicPreprocessResult {
  /** Drum-suppressed, RMS-leveled mono 22,050 Hz PCM ready for Basic Pitch */
  harmonicPcm: Float32Array;
  /** Stereo center-panning profile for separating center vocals from stereo chords */
  centerProfile: StereoCenterProfile;
  /** Estimated percussive energy reduction ratio (0..1, higher = more drums stripped) */
  percussiveReduction: number;
}

/**
 * In-place Radix-2 Inverse FFT using the conjugate symmetry identity:
 * IFFT(X) = conj(FFT(conj(X))) / N.
 */
export function ifftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  for (let i = 0; i < n; i++) {
    imag[i] = -imag[i];
  }
  fftRadix2(real, imag);
  const invN = 1 / n;
  for (let i = 0; i < n; i++) {
    real[i] *= invN;
    imag[i] = -imag[i] * invN;
  }
}

function createPeriodicHannWindow(size: number): Float32Array {
  const win = new Float32Array(size);
  const factor = (2 * Math.PI) / size;
  for (let i = 0; i < size; i++) {
    win[i] = 0.5 * (1 - Math.cos(factor * i));
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

/**
 * Adaptive look-ahead RMS leveling (gentle AGC + soft peak limiter).
 * Brings quiet piano/vocal verses and loud choruses into Basic Pitch's
 * optimal dynamic range without amplifying near-silent background noise.
 */
export function applyAdaptiveRmsLeveling(
  input: Float32Array,
  sampleRate = PREPROCESS_SAMPLE_RATE,
): Float32Array {
  const n = input.length;
  if (n === 0) return new Float32Array(0);

  const blockHop = Math.max(256, Math.round(sampleRate * 0.1)); // 100ms blocks
  const numBlocks = Math.max(1, Math.ceil(n / blockHop));
  const blockRms = new Float32Array(numBlocks);

  let maxBlockRms = 0;
  for (let b = 0; b < numBlocks; b++) {
    const start = b * blockHop;
    const end = Math.min(n, start + blockHop * 2);
    let sumSq = 0;
    for (let i = start; i < end; i++) {
      const s = input[i];
      sumSq += s * s;
    }
    const rms = Math.sqrt(sumSq / Math.max(1, end - start));
    blockRms[b] = rms;
    if (rms > maxBlockRms) maxBlockRms = rms;
  }

  if (maxBlockRms < 1e-5) {
    return new Float32Array(input);
  }

  // Smooth RMS envelope over ±5 blocks (~1 second) so musical phrasing stays natural
  const smoothGain = new Float32Array(numBlocks);
  const targetRms = 0.145;
  const noiseFloor = Math.max(0.004, maxBlockRms * 0.04);

  for (let b = 0; b < numBlocks; b++) {
    let acc = 0;
    let cnt = 0;
    const lo = Math.max(0, b - 5);
    const hi = Math.min(numBlocks - 1, b + 5);
    for (let k = lo; k <= hi; k++) {
      acc += blockRms[k];
      cnt++;
    }
    const localRms = acc / cnt;
    if (localRms < noiseFloor) {
      smoothGain[b] = 1.0;
    } else {
      const rawGain = targetRms / localRms;
      // Gentle clamping: avoid over-boosting quiet room noise or crushing forte peaks
      smoothGain[b] = Math.max(0.55, Math.min(2.8, rawGain));
    }
  }

  const out = new Float32Array(n);
  let peak = 0;

  for (let i = 0; i < n; i++) {
    const pos = i / blockHop;
    const idx = Math.min(numBlocks - 1, Math.floor(pos));
    const nextIdx = Math.min(numBlocks - 1, idx + 1);
    const frac = pos - idx;
    const g = smoothGain[idx] + (smoothGain[nextIdx] - smoothGain[idx]) * frac;

    let sample = input[i] * g;
    // Soft knee limiter above 0.90 to avoid harsh digital clipping
    const absS = Math.abs(sample);
    if (absS > 0.9) {
      const excess = absS - 0.9;
      sample = Math.sign(sample) * (0.9 + 0.06 * Math.tanh(excess / 0.15));
    }
    out[i] = sample;
    const a = Math.abs(sample);
    if (a > peak) peak = a;
  }

  // Final safety peak normalization to <= 0.95
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < n; i++) out[i] *= scale;
  }

  return out;
}

/**
 * Queries the stereo center-panning score (0..1) for a detected note event.
 * Returns 0.5 (neutral) if the source audio was mono or out of bounds.
 */
export function queryCenterScore(
  profile: StereoCenterProfile | null | undefined,
  startTime: number,
  duration: number,
  midi: number,
): number {
  if (!profile || !profile.isStereo || profile.numFrames === 0) return 0.5;
  const keyIdx = midi - MIN_MIDI_PROFILE;
  if (keyIdx < 0 || keyIdx >= NUM_PIANO_KEYS) return 0.5;

  const startFrame = Math.max(0, Math.floor(startTime / profile.hopSec));
  const endFrame = Math.min(
    profile.numFrames - 1,
    Math.max(startFrame, Math.ceil((startTime + duration) / profile.hopSec)),
  );

  let sum = 0;
  let count = 0;
  for (let f = startFrame; f <= endFrame; f++) {
    sum += profile.scores[f * NUM_PIANO_KEYS + keyIdx];
    count++;
  }
  return count > 0 ? sum / count : 0.5;
}

function* preprocessHarmonicGen(
  left22050: Float32Array,
  right22050: Float32Array | null,
  sampleRate: number,
  onProgress?: (progress: number, stage: string) => void,
): Generator<void, HarmonicPreprocessResult, void> {
  const n = left22050.length;
  const isStereo = right22050 !== null && right22050.length === n;

  // Build mid and side channels
  const mid = new Float32Array(n);
  const side = isStereo ? new Float32Array(n) : null;

  if (isStereo && right22050) {
    let sideEnergy = 0;
    let midEnergy = 0;
    for (let i = 0; i < n; i++) {
      const m = 0.5 * (left22050[i] + right22050[i]);
      const s = 0.5 * (left22050[i] - right22050[i]);
      mid[i] = m;
      side![i] = s;
      midEnergy += m * m;
      sideEnergy += s * s;
    }
    // If left & right are virtually identical (dual-mono MP3), mark stereo false
    const stereoActive = sideEnergy > midEnergy * 0.0015;
    if (!stereoActive) {
      // Fall through cleanly with mono handling
    }
  } else {
    mid.set(left22050);
  }

  let hasTrueStereo = false;
  if (side) {
    let sSq = 0;
    let mSq = 0;
    for (let i = 0; i < n; i += 4) {
      sSq += side[i] * side[i];
      mSq += mid[i] * mid[i];
    }
    hasTrueStereo = sSq > mSq * 0.0015;
  }

  const numFrames = Math.max(1, Math.floor((n - STFT_SIZE) / STFT_HOP) + 1);
  const binHz = sampleRate / STFT_SIZE;
  // Active harmonic band: ~36 Hz (D1) up to ~4400 Hz (cleans sub-kick rumble & hi-hat/cymbal hiss)
  const minBin = Math.max(1, Math.floor(36 / binHz));
  const rolloffStartBin = Math.floor(3700 / binHz);
  const maxBin = Math.min(STFT_SIZE / 2 - 1, Math.ceil(4450 / binHz));
  const activeBins = maxBin - minBin + 1;

  const win = createPeriodicHannWindow(STFT_SIZE);

  // Store complex STFT bins ONLY for activeBins [minBin..maxBin] to keep RAM lean
  const specRe = new Float32Array(numFrames * activeBins);
  const specIm = new Float32Array(numFrames * activeBins);
  const specMag = new Float32Array(numFrames * activeBins);
  const centerScores = hasTrueStereo
    ? new Float32Array(numFrames * NUM_PIANO_KEYS)
    : new Float32Array(0);

  // Precompute fundamental & 2nd harmonic FFT bin indices for 88 piano keys
  const keyBin1 = new Int32Array(NUM_PIANO_KEYS);
  const keyBin2 = new Int32Array(NUM_PIANO_KEYS);
  for (let k = 0; k < NUM_PIANO_KEYS; k++) {
    const f0 = midiToFreq(MIN_MIDI_PROFILE + k);
    keyBin1[k] = Math.round(f0 / binHz);
    keyBin2[k] = Math.round((f0 * 2) / binHz);
  }

  const fftRe = new Float32Array(STFT_SIZE);
  const fftIm = new Float32Array(STFT_SIZE);
  const sideRe = hasTrueStereo ? new Float32Array(STFT_SIZE) : null;
  const sideIm = hasTrueStereo ? new Float32Array(STFT_SIZE) : null;

  onProgress?.(0.08, 'Eliminando batería y ruido percusivo (HPSS espectral)...');

  // Pass 1: Forward STFT on mid (+ side magnitude for stereo center-vocal profile)
  for (let f = 0; f < numFrames; f++) {
    const offset = f * STFT_HOP;
    for (let i = 0; i < STFT_SIZE; i++) {
      const idx = offset + i;
      fftRe[i] = (idx < n ? mid[idx] : 0) * win[i];
      fftIm[i] = 0;
    }
    fftRadix2(fftRe, fftIm);

    const rowBase = f * activeBins;
    for (let b = 0; b < activeBins; b++) {
      const bin = minBin + b;
      const re = fftRe[bin];
      const im = fftIm[bin];
      specRe[rowBase + b] = re;
      specIm[rowBase + b] = im;
      specMag[rowBase + b] = Math.hypot(re, im);
    }

    if (hasTrueStereo && side && sideRe && sideIm) {
      for (let i = 0; i < STFT_SIZE; i++) {
        const idx = offset + i;
        sideRe[i] = (idx < n ? side[idx] : 0) * win[i];
        sideIm[i] = 0;
      }
      fftRadix2(sideRe, sideIm);

      const cRow = f * NUM_PIANO_KEYS;
      for (let k = 0; k < NUM_PIANO_KEYS; k++) {
        const b1 = keyBin1[k];
        const b2 = keyBin2[k];
        let mEnergy = 0;
        let sEnergy = 0;
        if (b1 >= minBin && b1 <= maxBin) {
          mEnergy += specMag[rowBase + (b1 - minBin)];
          sEnergy += Math.hypot(sideRe[b1], sideIm[b1]);
        }
        if (b2 >= minBin && b2 <= maxBin) {
          mEnergy += 0.65 * specMag[rowBase + (b2 - minBin)];
          sEnergy += 0.65 * Math.hypot(sideRe[b2], sideIm[b2]);
        }
        const denom = mEnergy + sEnergy + 1e-6;
        centerScores[cRow + k] = Math.max(0, Math.min(1, (mEnergy - 0.8 * sEnergy) / denom));
      }
    }

    if (f > 0 && f % 450 === 0) {
      onProgress?.(
        0.08 + (f / numFrames) * 0.05,
        'Analizando espectro armónico y separación estéreo...',
      );
      yield;
    }
  }

  // Pass 2: 2D Median HPSS Harmonic Wiener Masking + WOLA iSTFT waveform synthesis
  const TIME_HALF = 3; // 7 frames ≈ 162 ms: preserves fast vocal syllables & piano runs while killing short drum hits
  const FREQ_HALF = 5; // 11 bins ≈ 118 Hz vertical percussive window
  const BETA_PERCUSSIVE = 1.05; // Strong rejection of kick/snare/hi-hat transients without dulling piano hammers
  const scratchT = new Float32Array(TIME_HALF * 2 + 1);
  const scratchF = new Float32Array(FREQ_HALF * 2 + 1);

  const outLen = Math.max(n, (numFrames - 1) * STFT_HOP + STFT_SIZE);
  const harmonicAccum = new Float32Array(outLen);
  const winSqSum = new Float32Array(outLen);

  let totalMagBefore = 0;
  let totalMagAfter = 0;

  for (let f = 0; f < numFrames; f++) {
    fftRe.fill(0);
    fftIm.fill(0);

    const tLo = Math.max(0, f - TIME_HALF);
    const tHi = Math.min(numFrames - 1, f + TIME_HALF);
    const tLen = tHi - tLo + 1;
    const rowBase = f * activeBins;

    for (let b = 0; b < activeBins; b++) {
      const mag = specMag[rowBase + b];
      if (mag <= 1e-7) continue;

      totalMagBefore += mag;

      let ti = 0;
      for (let t = tLo; t <= tHi; t++) {
        scratchT[ti++] = specMag[t * activeBins + b];
      }
      const hEst = medianOfSmall(scratchT, tLen);

      const bLo = Math.max(0, b - FREQ_HALF);
      const bHi = Math.min(activeBins - 1, b + FREQ_HALF);
      const bLen = bHi - bLo + 1;
      let bi = 0;
      for (let k = bLo; k <= bHi; k++) {
        scratchF[bi++] = specMag[rowBase + k];
      }
      const pEst = medianOfSmall(scratchF, bLen);

      const h2 = hEst * hEst;
      const p2 = (BETA_PERCUSSIVE * pEst) * (BETA_PERCUSSIVE * pEst);
      const wiener = h2 / (h2 + p2 + 1e-12);
      // Soft floor (0.10) preserves crisp piano/vocal attack transients while attenuating drums ~14-18 dB
      let mask = 0.1 + 0.9 * wiener;

      // Smooth cosine high-frequency rolloff between 3700 Hz and 4450 Hz
      const bin = minBin + b;
      if (bin > rolloffStartBin) {
        const frac = (bin - rolloffStartBin) / Math.max(1, maxBin - rolloffStartBin);
        mask *= 0.5 * (1 + Math.cos(Math.PI * Math.min(1, frac)));
      }

      totalMagAfter += mag * mask;

      const re = specRe[rowBase + b] * mask;
      const im = specIm[rowBase + b] * mask;
      fftRe[bin] = re;
      fftIm[bin] = im;
      // Hermitian symmetry so iFFT yields real time-domain audio
      const mirror = STFT_SIZE - bin;
      fftRe[mirror] = re;
      fftIm[mirror] = -im;
    }

    ifftRadix2(fftRe, fftIm);

    const offset = f * STFT_HOP;
    for (let i = 0; i < STFT_SIZE; i++) {
      const w = win[i];
      harmonicAccum[offset + i] += fftRe[i] * w;
      winSqSum[offset + i] += w * w;
    }

    if (f > 0 && f % 320 === 0) {
      onProgress?.(
        0.13 + (f / numFrames) * 0.07,
        'Reconstruyendo señal armónica limpia sin batería...',
      );
      yield;
    }
  }

  // Normalize Weighted Overlap-Add (WOLA)
  const harmonicRaw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const denom = winSqSum[i];
    harmonicRaw[i] = denom > 1e-4 ? harmonicAccum[i] / denom : 0;
  }

  onProgress?.(0.21, 'Nivelando dinámica para máxima sensibilidad de Basic Pitch...');
  const harmonicPcm = applyAdaptiveRmsLeveling(harmonicRaw, sampleRate);

  const percussiveReduction =
    totalMagBefore > 1e-6
      ? Math.max(0, Math.min(1, 1 - totalMagAfter / totalMagBefore))
      : 0;

  return {
    harmonicPcm,
    centerProfile: {
      isStereo: hasTrueStereo,
      hopSec: STFT_HOP / sampleRate,
      numFrames,
      scores: centerScores,
    },
    percussiveReduction,
  };
}

/**
 * Async non-blocking HPSS + Stereo Center-Profile + Adaptive AGC preprocessor
 * for browser audio uploads (yields smoothly between STFT blocks).
 */
export async function preprocessAudioForBasicPitchAsync(
  left22050: Float32Array,
  right22050: Float32Array | null,
  sampleRate = PREPROCESS_SAMPLE_RATE,
  onProgress?: (progress: number, stage: string) => void,
): Promise<HarmonicPreprocessResult> {
  const gen = preprocessHarmonicGen(left22050, right22050, sampleRate, onProgress);
  let step = gen.next();
  while (!step.done) {
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    step = gen.next();
  }
  return step.value;
}

/**
 * Synchronous variant for deterministic Node unit tests.
 */
export function preprocessAudioForBasicPitch(
  left22050: Float32Array,
  right22050: Float32Array | null,
  sampleRate = PREPROCESS_SAMPLE_RATE,
): HarmonicPreprocessResult {
  const gen = preprocessHarmonicGen(left22050, right22050, sampleRate);
  let step = gen.next();
  while (!step.done) {
    step = gen.next();
  }
  return step.value;
}

/**
 * Zero-phase 1st-order forward-backward high-pass filter (~24 Hz) that removes DC offset
 * and low-frequency room/mic rumble while keeping the lowest piano key A0 (27.5 Hz)
 * and all crisp hammer attack transients 100% intact.
 */
function applyZeroPhaseHighPass24Hz(input: Float32Array, sampleRate: number): Float32Array {
  const n = input.length;
  if (n < 4) return new Float32Array(input);
  const rc = 1 / (2 * Math.PI * 24);
  const dt = 1 / sampleRate;
  const alpha = rc / (rc + dt);

  const fwd = new Float32Array(n);
  fwd[0] = input[0];
  for (let i = 1; i < n; i++) {
    fwd[i] = alpha * (fwd[i - 1] + input[i] - input[i - 1]);
  }

  const out = new Float32Array(n);
  out[n - 1] = fwd[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    out[i] = alpha * (out[i + 1] + fwd[i] - fwd[i + 1]);
  }
  return out;
}

/**
 * Solo-piano-optimized preprocessor for Basic Pitch.
 * Unlike pop-mix HPSS (which softens percussive hits to strip drums), solo piano relies
 * on crisp hammer attack transients to detect fast arpeggios, trills, and repeated notes.
 * This stage:
 *   1. Sums stereo piano cleanly to mono without phase smear
 *   2. Removes DC & sub-24 Hz mic rumble while preserving all 88 keys (A0 27.5 Hz .. C8 4186 Hz)
 *   3. Applies gentle pianissimo-friendly adaptive RMS leveling so soft passages transcribe
 *      just as accurately as forte passages without crushing expressive piano dynamics.
 */
export function preprocessSoloPianoForBasicPitch(
  left22050: Float32Array,
  right22050: Float32Array | null,
  sampleRate = PREPROCESS_SAMPLE_RATE,
): Float32Array {
  const n = left22050.length;
  const mono = new Float32Array(n);
  if (right22050 && right22050.length === n) {
    for (let i = 0; i < n; i++) {
      mono[i] = 0.5 * (left22050[i] + right22050[i]);
    }
  } else {
    mono.set(left22050);
  }

  const hp = applyZeroPhaseHighPass24Hz(mono, sampleRate);
  return applyAdaptiveRmsLeveling(hp, sampleRate);
}


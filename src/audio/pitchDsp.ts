import { MAX_MIDI, MIN_MIDI, midiToFreq } from './pianoTheory';

export type TrackRole = 'vocal' | 'piano';

export interface PianoNoteEvent {
  id: string;
  midi: number;
  startTime: number;
  duration: number;
  velocity: number;
  track: TrackRole;
}

export interface TranscribedPiece {
  id: string;
  title: string;
  subtitle: string;
  source: 'demo' | 'upload';
  duration: number;
  bpm: number;
  keySignature: string;
  notes: PianoNoteEvent[];
  pianoNoteCount: number;
  vocalNoteCount: number;
}

export interface TranscriptionOptions {
  /** Maximum duration to analyze in seconds (default 240s) */
  maxDurationSec?: number;
  /** Target internal sample rate for FFT analysis (default 22050) */
  targetSampleRate?: number;
  /** FFT size (power of 2, default 4096) */
  fftSize?: number;
  /** Hop size in samples at targetSampleRate (default 1024 -> ~46ms) */
  hopSize?: number;
  /** Progress callback (0..1) */
  onProgress?: (progress: number, stage: string) => void;
}

/**
 * In-place Radix-2 Cooley-Tukey FFT.
 * `real` and `imag` must have equal power-of-two length.
 */
export function fftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (n <= 1) return;

  // Bit-reversal permutation
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

  // Butterfly stages
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
 * Downsamples a PCM Float32Array via simple box-averaging antialiasing filter.
 */
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
 * Extracts a vocal-emphasized signal from mono or stereo PCM channels.
 * For stereo recordings, lead vocals are almost universally panned dead-center,
 * while guitars/pads/reverb spread to the sides. We compute a center-weighted
 * spectral-ready time signal plus gentle high-pass pre-emphasis to lift 220–1400 Hz.
 */
export function buildVocalEmphasisSignal(
  left: Float32Array,
  right: Float32Array | null,
  sampleRate: number,
): Float32Array {
  const n = left.length;
  const out = new Float32Array(n);

  if (right && right.length === n) {
    // Soft center extraction: boost correlated mid while attenuating wide side bleed
    for (let i = 0; i < n; i++) {
      const l = left[i];
      const r = right[i];
      const mid = 0.5 * (l + r);
      const side = 0.5 * (l - r);
      const midAbs = Math.abs(mid);
      const sideAbs = Math.abs(side);
      // Smooth soft-mask weighting favoring center-panned vocal energy
      const centerWeight = midAbs / (midAbs + sideAbs * 1.15 + 1e-6);
      out[i] = mid * (0.35 + 0.65 * centerWeight);
    }
  } else {
    out.set(left);
  }

  // First-order high-pass (~150 Hz) to strip bass/kick drum rumble from vocal tracker
  const rc = 1 / (2 * Math.PI * 155);
  const dt = 1 / sampleRate;
  const alpha = rc / (rc + dt);
  let prevIn = 0;
  let prevOut = 0;
  for (let i = 0; i < n; i++) {
    const x = out[i];
    const y = alpha * (prevOut + x - prevIn);
    prevIn = x;
    prevOut = y;
    out[i] = y;
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

/**
 * Measures log-compressed spectral magnitude around a target frequency using
 * narrow Gaussian-weighted bin pooling around the fractional FFT bin.
 */
function sampleFreqMagnitude(
  mag: Float32Array,
  freqHz: number,
  sampleRate: number,
  fftSize: number,
): number {
  const binFloat = (freqHz * fftSize) / sampleRate;
  const center = Math.round(binFloat);
  const halfBins = mag.length - 1;
  if (center <= 1 || center >= halfBins - 1) return 0;

  // Parabolic interpolation around peak bin for crisp pitch resolution
  const m0 = mag[center - 1];
  const m1 = mag[center];
  const m2 = mag[center + 1];
  const maxLocal = Math.max(m0, m1, m2);
  const weighted = m1 * 0.6 + (m0 + m2) * 0.2;
  return Math.max(maxLocal * 0.85, weighted);
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

// Krumhansl-Schmuckler major & minor key profiles
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

export function estimateKeySignature(notes: PianoNoteEvent[]): string {
  if (notes.length === 0) return 'Do mayor';
  const chroma = new Float64Array(12);
  for (const n of notes) {
    chroma[n.midi % 12] += n.duration * n.velocity * (n.track === 'vocal' ? 1.35 : 1.0);
  }

  let bestScore = -Infinity;
  let bestLabel = 'Do mayor';

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
      bestLabel = `${NOTE_NAMES_ES[root]} mayor`;
    }
    const min = correlate(MINOR_PROFILE, root);
    if (min > bestScore) {
      bestScore = min;
      bestLabel = `${NOTE_NAMES_ES[root]} menor`;
    }
  }
  return bestLabel;
}

export function estimateTempoBpm(onsetFlux: Float32Array, hopDurationSec: number): number {
  if (onsetFlux.length < 20) return 100;
  // Autocorrelation over BPM range 64..168
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
    // Gentle prior toward 90-130 BPM musical sweet spot
    const bpm = 60 / (lag * hopDurationSec);
    const prior = 1 - 0.12 * Math.abs(bpm - 110) / 60;
    if (normalized * prior > bestCorr) {
      bestCorr = normalized * prior;
      bestLag = lag;
    }
  }

  const estimated = Math.round(60 / (bestLag * hopDurationSec));
  return Math.max(64, Math.min(168, estimated));
}

/**
 * Pure PCM-to-Piano+Vocal transcription engine.
 * Works directly on Float32Array PCM buffers (so it runs both in Web Audio browser
 * and in Node unit tests without browser APIs).
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
  const maxDurationSec = options.maxDurationSec ?? 240;

  options.onProgress?.(0.05, 'Preparando señal estéreo y canal vocal');

  const maxSamples = Math.min(leftChannel.length, Math.floor(maxDurationSec * sourceSampleRate));
  const leftTrimmed = leftChannel.subarray(0, maxSamples);
  const rightTrimmed = rightChannel ? rightChannel.subarray(0, maxSamples) : null;

  // Build mono mix for piano harmony & center-emphasized vocal signal
  const monoMix = new Float32Array(maxSamples);
  if (rightTrimmed) {
    for (let i = 0; i < maxSamples; i++) {
      monoMix[i] = 0.5 * (leftTrimmed[i] + rightTrimmed[i]);
    }
  } else {
    monoMix.set(leftTrimmed);
  }

  const vocalRaw = buildVocalEmphasisSignal(leftTrimmed, rightTrimmed, sourceSampleRate);

  const monoDs = downsampleMono(monoMix, sourceSampleRate, targetSampleRate);
  const vocalDs = downsampleMono(vocalRaw, sourceSampleRate, targetSampleRate);

  const duration = monoDs.length / targetSampleRate;
  const hopSec = hopSize / targetSampleRate;
  const numFrames = Math.max(1, Math.floor((monoDs.length - fftSize) / hopSize) + 1);

  const hann = createHannWindow(fftSize);
  const halfN = fftSize >> 1;
  const realBuf = new Float32Array(fftSize);
  const imagBuf = new Float32Array(fftSize);
  const magMono = new Float32Array(halfN);
  const magVocal = new Float32Array(halfN);
  const prevMagMono = new Float32Array(halfN);
  const onsetFlux = new Float32Array(numFrames);

  // Precompute piano note frequencies (MIDI 21..108 -> index 0..87)
  const NUM_NOTES = MAX_MIDI - MIN_MIDI + 1;
  const noteFreqs = new Float32Array(NUM_NOTES);
  for (let i = 0; i < NUM_NOTES; i++) {
    noteFreqs[i] = midiToFreq(MIN_MIDI + i);
  }

  // Per-frame extracted representations
  const vocalRangeMin = 48; // C3
  const vocalRangeMax = 84; // C6
  const vocalSpan = vocalRangeMax - vocalRangeMin + 1;

  // Salience matrix for Viterbi vocal pitch tracking: [frame * vocalSpan + idx]
  const vocalSalience = new Float32Array(numFrames * vocalSpan);
  const frameVocalEnergy = new Float32Array(numFrames);

  // Piano polyphonic detections per frame: array of { midi, strength }
  const pianoFramePicks: Array<Array<{ midi: number; strength: number }>> = new Array(numFrames);

  let globalMaxPianoStrength = 1e-6;
  let globalMaxVocalStrength = 1e-6;

  // Scratch arrays for harmonic analysis
  const rawPitchEnergy = new Float32Array(NUM_NOTES);
  const harmonicScore = new Float32Array(NUM_NOTES);

  for (let f = 0; f < numFrames; f++) {
    if (f % 40 === 0) {
      options.onProgress?.(
        0.1 + 0.65 * (f / numFrames),
        'Analizando espectro armónico y melodía vocal',
      );
    }

    const offset = f * hopSize;

    // 1. FFT of full mono mix (piano chords + bass + accompaniment)
    let frameRms = 0;
    for (let i = 0; i < fftSize; i++) {
      const s = offset + i < monoDs.length ? monoDs[offset + i] : 0;
      frameRms += s * s;
      realBuf[i] = s * hann[i];
      imagBuf[i] = 0;
    }
    frameRms = Math.sqrt(frameRms / fftSize);

    fftRadix2(realBuf, imagBuf);

    let flux = 0;
    for (let k = 0; k < halfN; k++) {
      const m = Math.hypot(realBuf[k], imagBuf[k]) / (fftSize * 0.25);
      const logM = Math.log1p(m * 18);
      magMono[k] = logM;
      const diff = logM - prevMagMono[k];
      if (diff > 0) flux += diff;
      prevMagMono[k] = logM;
    }
    onsetFlux[f] = flux;

    // 2. FFT of vocal-emphasized center signal
    for (let i = 0; i < fftSize; i++) {
      const s = offset + i < vocalDs.length ? vocalDs[offset + i] : 0;
      realBuf[i] = s * hann[i];
      imagBuf[i] = 0;
    }
    fftRadix2(realBuf, imagBuf);
    for (let k = 0; k < halfN; k++) {
      const m = Math.hypot(realBuf[k], imagBuf[k]) / (fftSize * 0.25);
      magVocal[k] = Math.log1p(m * 18);
    }

    // 3. Compute vocal melody salience (MIDI 48..84) using harmonic summation
    let bestVocalFrame = 0;
    if (frameRms > 0.008) {
      for (let midi = vocalRangeMin; midi <= vocalRangeMax; midi++) {
        const f0 = noteFreqs[midi - MIN_MIDI];
        const h1 = sampleFreqMagnitude(magVocal, f0, targetSampleRate, fftSize);
        const h2 = sampleFreqMagnitude(magVocal, f0 * 2, targetSampleRate, fftSize);
        const h3 = sampleFreqMagnitude(magVocal, f0 * 3, targetSampleRate, fftSize);
        const h4 = sampleFreqMagnitude(magVocal, f0 * 4, targetSampleRate, fftSize);
        // Sub-harmonic penalty prevents selecting octave-above harmonics as vocal melody
        const subHalf = sampleFreqMagnitude(magVocal, f0 * 0.5, targetSampleRate, fftSize);

        // Vocal timbre favors strong fundamental + 2nd/3rd harmonics in human tessitura (C4–G5)
        const tessituraBoost = midi >= 57 && midi <= 79 ? 1.18 : 0.96;
        const score =
          Math.max(0, h1 * 1.0 + h2 * 0.62 + h3 * 0.34 + h4 * 0.16 - subHalf * 0.38) *
          tessituraBoost;

        vocalSalience[f * vocalSpan + (midi - vocalRangeMin)] = score;
        if (score > bestVocalFrame) bestVocalFrame = score;
      }
    }
    frameVocalEnergy[f] = bestVocalFrame;
    if (bestVocalFrame > globalMaxVocalStrength) globalMaxVocalStrength = bestVocalFrame;

    // 4. Polyphonic piano extraction with iterative harmonic cancellation
    const picks: Array<{ midi: number; strength: number }> = [];
    if (frameRms > 0.006) {
      // Sample raw energy for each piano key (MIDI 28..96)
      rawPitchEnergy.fill(0);
      for (let idx = 7; idx <= 75; idx++) {
        const f0 = noteFreqs[idx];
        rawPitchEnergy[idx] = sampleFreqMagnitude(magMono, f0, targetSampleRate, fftSize);
      }

      // Compute harmonic comb score per note
      harmonicScore.fill(0);
      for (let idx = 7; idx <= 75; idx++) {
        const f0 = noteFreqs[idx];
        const e1 = rawPitchEnergy[idx];
        if (e1 < 0.02) continue;
        const e2 = sampleFreqMagnitude(magMono, f0 * 2, targetSampleRate, fftSize);
        const e3 = sampleFreqMagnitude(magMono, f0 * 3, targetSampleRate, fftSize);
        const e4 = sampleFreqMagnitude(magMono, f0 * 4, targetSampleRate, fftSize);
        harmonicScore[idx] = e1 + e2 * 0.55 + e3 * 0.28 + e4 * 0.14;
      }

      // Iteratively pick up to 5 polyphonic piano voices and attenuate their overtones
      const maxPoly = 5;
      for (let p = 0; p < maxPoly; p++) {
        let bestIdx = -1;
        let bestVal = 0;
        for (let idx = 7; idx <= 75; idx++) {
          const val = harmonicScore[idx];
          if (val > bestVal) {
            // Ensure it's a local pitch peak (not spectral leakage from adjacent semitone)
            const leftVal = idx > 7 ? harmonicScore[idx - 1] : 0;
            const rightVal = idx < 75 ? harmonicScore[idx + 1] : 0;
            if (val >= leftVal * 0.98 && val >= rightVal * 0.98) {
              bestVal = val;
              bestIdx = idx;
            }
          }
        }
        if (bestIdx < 0 || bestVal < 0.08) break;

        picks.push({ midi: MIN_MIDI + bestIdx, strength: bestVal });
        if (bestVal > globalMaxPianoStrength) globalMaxPianoStrength = bestVal;

        // Suppress chosen note, its immediate semitone leakage, and its 2nd/3rd/4th harmonics
        const chosenMidi = MIN_MIDI + bestIdx;
        harmonicScore[bestIdx] = 0;
        if (bestIdx - 1 >= 0) harmonicScore[bestIdx - 1] *= 0.25;
        if (bestIdx + 1 < NUM_NOTES) harmonicScore[bestIdx + 1] *= 0.25;

        const harmonicOffsets = [12, 19, 24, 28, 31]; // octave, 12th, 2 octaves, major 17th, 19th
        const harmonicDamp = [0.35, 0.45, 0.55, 0.65, 0.72];
        for (let h = 0; h < harmonicOffsets.length; h++) {
          const hIdx = chosenMidi + harmonicOffsets[h] - MIN_MIDI;
          if (hIdx >= 0 && hIdx < NUM_NOTES) {
            harmonicScore[hIdx] *= harmonicDamp[h];
          }
        }
      }
    }
    pianoFramePicks[f] = picks;
  }

  options.onProgress?.(0.78, 'Trazando línea melódica vocal y acordes');

  // 5. Viterbi smoothing for continuous, lyrical vocal melody tracking
  const vocalThreshold = globalMaxVocalStrength * 0.28;
  const states = vocalSpan + 1; // index 0 = silent/unvoiced, 1..vocalSpan = MIDI
  const dp = new Float32Array(numFrames * states);
  const backptr = new Int16Array(numFrames * states);

  for (let s = 0; s < states; s++) {
    if (s === 0) {
      dp[0] = vocalThreshold * 0.85;
    } else {
      dp[s] = vocalSalience[s - 1];
    }
  }

  for (let f = 1; f < numFrames; f++) {
    const prevBase = (f - 1) * states;
    const currBase = f * states;
    const salBase = f * vocalSpan;

    for (let s = 0; s < states; s++) {
      const emission = s === 0 ? vocalThreshold * 0.82 : vocalSalience[salBase + (s - 1)];
      let bestPrev = 0;
      let bestScore = -Infinity;

      for (let p = 0; p < states; p++) {
        let transPenalty = 0;
        if (s === p) {
          transPenalty = 0.04 * globalMaxVocalStrength; // reward staying steady on pitch
        } else if (s === 0 || p === 0) {
          transPenalty = -0.05 * globalMaxVocalStrength;
        } else {
          const semitoneJump = Math.abs(s - p);
          // Smooth vocal lines favor stepwise motion or moderate intervals
          transPenalty = -Math.min(semitoneJump, 14) * 0.014 * globalMaxVocalStrength;
        }
        const cand = dp[prevBase + p] + transPenalty;
        if (cand > bestScore) {
          bestScore = cand;
          bestPrev = p;
        }
      }

      dp[currBase + s] = bestScore + emission;
      backptr[currBase + s] = bestPrev;
    }
  }

  // Backtrack best vocal path
  const vocalPath = new Int16Array(numFrames);
  let lastBestState = 0;
  let lastBestVal = -Infinity;
  const lastBase = (numFrames - 1) * states;
  for (let s = 0; s < states; s++) {
    if (dp[lastBase + s] > lastBestVal) {
      lastBestVal = dp[lastBase + s];
      lastBestState = s;
    }
  }
  vocalPath[numFrames - 1] = lastBestState;
  for (let f = numFrames - 2; f >= 0; f--) {
    vocalPath[f] = backptr[(f + 1) * states + vocalPath[f + 1]];
  }

  // Convert vocal path to PianoNoteEvents
  const vocalNotes: PianoNoteEvent[] = [];
  let activeVocalMidi = -1;
  let activeVocalStart = 0;
  let activeVocalPeak = 0;
  let noteIdCounter = 1;

  const avgFlux = onsetFlux.reduce((acc, v) => acc + v, 0) / Math.max(1, numFrames);

  const flushVocalNote = (endFrame: number) => {
    if (activeVocalMidi < 0) return;
    const dur = (endFrame - activeVocalStart) * hopSec;
    if (dur >= 0.085) {
      const vel = Math.max(0.42, Math.min(1.0, activeVocalPeak / globalMaxVocalStrength));
      vocalNotes.push({
        id: `v-${noteIdCounter++}`,
        midi: activeVocalMidi,
        startTime: Number((activeVocalStart * hopSec).toFixed(3)),
        duration: Number(Math.min(4.0, dur).toFixed(3)),
        velocity: Number(vel.toFixed(2)),
        track: 'vocal',
      });
    }
    activeVocalMidi = -1;
  };

  for (let f = 0; f < numFrames; f++) {
    const state = vocalPath[f];
    const midi = state === 0 ? -1 : vocalRangeMin + (state - 1);
    const strength = state === 0 ? 0 : vocalSalience[f * vocalSpan + (state - 1)];

    // Re-articulate repeated vocal syllable on strong spectral onset
    const strongReOnset =
      midi >= 0 &&
      midi === activeVocalMidi &&
      f - activeVocalStart >= 4 &&
      onsetFlux[f] > avgFlux * 2.15 &&
      strength > activeVocalPeak * 0.72;

    if (midi !== activeVocalMidi || strongReOnset) {
      flushVocalNote(f);
      if (midi >= 0 && strength >= vocalThreshold * 0.72) {
        activeVocalMidi = midi;
        activeVocalStart = f;
        activeVocalPeak = strength;
      }
    } else if (midi >= 0) {
      if (strength > activeVocalPeak) activeVocalPeak = strength;
    }
  }
  flushVocalNote(numFrames);

  options.onProgress?.(0.9, 'Segmentando acordes y refinando dinámicas');

  // 6. Segment polyphonic piano accompaniment notes
  const pianoThreshold = globalMaxPianoStrength * 0.24;
  const pianoNotes: PianoNoteEvent[] = [];

  interface ActivePianoVoice {
    startFrame: number;
    lastSeenFrame: number;
    peakStrength: number;
  }
  const activePiano = new Map<number, ActivePianoVoice>();

  const flushPianoVoice = (midi: number, voice: ActivePianoVoice) => {
    const frames = voice.lastSeenFrame - voice.startFrame + 1;
    const dur = frames * hopSec;
    if (dur >= 0.075) {
      const vel = Math.max(0.28, Math.min(0.92, (voice.peakStrength / globalMaxPianoStrength) * 0.92));
      pianoNotes.push({
        id: `p-${noteIdCounter++}`,
        midi,
        startTime: Number((voice.startFrame * hopSec).toFixed(3)),
        duration: Number(Math.min(3.8, dur + hopSec * 0.5).toFixed(3)),
        velocity: Number(vel.toFixed(2)),
        track: 'piano',
      });
    }
  };

  for (let f = 0; f < numFrames; f++) {
    const picks = pianoFramePicks[f];
    const vocalMidiThisFrame = vocalPath[f] === 0 ? -1 : vocalRangeMin + (vocalPath[f] - 1);
    const seen = new Set<number>();

    for (const pick of picks) {
      if (pick.strength < pianoThreshold) continue;
      // Avoid duplicating the exact same MIDI note currently held by the vocal melody line
      if (pick.midi === vocalMidiThisFrame) continue;

      seen.add(pick.midi);
      const existing = activePiano.get(pick.midi);
      if (existing) {
        // Re-strike if sharp energy jump after sustain
        const isReStrike =
          f - existing.startFrame >= 5 &&
          onsetFlux[f] > avgFlux * 2.3 &&
          pick.strength > existing.peakStrength * 0.85;
        if (isReStrike) {
          flushPianoVoice(pick.midi, existing);
          activePiano.set(pick.midi, {
            startFrame: f,
            lastSeenFrame: f,
            peakStrength: pick.strength,
          });
        } else {
          existing.lastSeenFrame = f;
          if (pick.strength > existing.peakStrength) existing.peakStrength = pick.strength;
        }
      } else {
        activePiano.set(pick.midi, {
          startFrame: f,
          lastSeenFrame: f,
          peakStrength: pick.strength,
        });
      }
    }

    // Close voices missing for more than 1 frame (brief 1-frame bridge smooths tremolo)
    for (const [midi, voice] of activePiano.entries()) {
      if (!seen.has(midi) && f - voice.lastSeenFrame >= 2) {
        flushPianoVoice(midi, voice);
        activePiano.delete(midi);
      }
    }
  }
  for (const [midi, voice] of activePiano.entries()) {
    flushPianoVoice(midi, voice);
  }

  // Merge and chronologically sort all notes
  const allNotes = [...pianoNotes, ...vocalNotes].sort(
    (a, b) => a.startTime - b.startTime || a.midi - b.midi,
  );

  const bpm = estimateTempoBpm(onsetFlux, hopSec);
  const keySignature = estimateKeySignature(allNotes);

  options.onProgress?.(1, 'Transcripción completada');

  return {
    id: `upload-${Date.now()}`,
    title,
    subtitle: 'MP3 transcrito (Piano + Melodía vocal)',
    source: 'upload',
    duration: Number(duration.toFixed(2)),
    bpm,
    keySignature,
    notes: allNotes,
    pianoNoteCount: pianoNotes.length,
    vocalNoteCount: vocalNotes.length,
  };
}

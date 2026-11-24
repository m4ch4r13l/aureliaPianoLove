import test from 'node:test';
import assert from 'node:assert/strict';
import {
  build88KeyLayout,
  formatTime,
  isBlackKey,
  midiToFreq,
  midiToSolfege,
  TOTAL_KEYS,
} from './pianoTheory';
import {
  applyHarmonicPercussiveMask,
  fftRadix2,
  filterAndCleanNotes,
  transcribePcmToPianoAndVocal,
} from './pitchDsp';
import { buildMultiTrackMidi } from './midiExporter';

test('88-key grand piano layout generates 52 white and 36 black keys in ascending X order', () => {
  const { keys, totalWidth } = build88KeyLayout();
  assert.equal(keys.length, TOTAL_KEYS);
  assert.equal(keys.filter((k) => !k.isBlack).length, 52);
  assert.equal(keys.filter((k) => k.isBlack).length, 36);
  assert.ok(totalWidth > 1.15 && totalWidth < 1.35);

  for (let i = 1; i < keys.length; i++) {
    assert.ok(
      keys[i].x > keys[i - 1].x,
      `Key ${keys[i].midi} x (${keys[i].x}) should exceed key ${keys[i - 1].midi} x (${keys[i - 1].x})`,
    );
  }
});

test('Spanish solfege naming and time formatting work accurately', () => {
  assert.equal(midiToSolfege(60), 'Do4');
  assert.equal(midiToSolfege(69), 'La4');
  assert.equal(isBlackKey(61), true);
  assert.equal(isBlackKey(60), false);
  assert.equal(formatTime(0), '0:00');
  assert.equal(formatTime(125.7), '2:05');
});

test('Radix-2 FFT detects exact peak bin for 440 Hz sine wave', () => {
  const sampleRate = 16384;
  const n = 2048;
  const real = new Float32Array(n);
  const imag = new Float32Array(n);
  const targetFreq = 440;

  for (let i = 0; i < n; i++) {
    real[i] = Math.sin((2 * Math.PI * targetFreq * i) / sampleRate);
  }
  fftRadix2(real, imag);

  let peakBin = 0;
  let peakMag = 0;
  for (let k = 1; k < n / 2; k++) {
    const mag = Math.hypot(real[k], imag[k]);
    if (mag > peakMag) {
      peakMag = mag;
      peakBin = k;
    }
  }
  const detectedFreq = (peakBin * sampleRate) / n;
  assert.ok(Math.abs(detectedFreq - targetFreq) <= sampleRate / n, `Detected ${detectedFreq} Hz`);
});

test('HPSS suppresses single-frame broadband drum spikes while preserving sustained harmonic notes', () => {
  const frames = 20;
  const bins = 24;
  const mat = new Float32Array(frames * bins);

  // Sustained harmonic note at bin 10 across all frames
  for (let f = 0; f < frames; f++) {
    mat[f * bins + 10] = 1.0;
  }
  // Broadband percussive snare hit across ALL bins on frame 10 only
  for (let b = 0; b < bins; b++) {
    mat[10 * bins + b] = 1.2;
  }

  const cleaned = applyHarmonicPercussiveMask(mat, frames, bins, 3, 3);
  // Sustained harmonic bin survives strongly
  assert.ok(cleaned[6 * bins + 10] > 0.7, `Harmonic bin should survive, got ${cleaned[6 * bins + 10]}`);
  // Broadband drum bin at an unrelated pitch (bin 4, frame 10) is crushed to ~0
  assert.ok(cleaned[10 * bins + 4] < 0.05, `Percussive spike should be suppressed, got ${cleaned[10 * bins + 4]}`);
});

test('transcribePcmToPianoAndVocal separates bass, chords, and vocal melody cleanly even with percussive noise bursts', () => {
  const sampleRate = 22050;
  const durationSec = 2.4;
  const n = Math.floor(sampleRate * durationSec);
  const left = new Float32Array(n);
  const right = new Float32Array(n);

  let seed = 1337;
  const noiseRand = () => {
    seed = (seed * 16807) % 2147483647;
    return ((seed - 1) / 2147483646) * 2 - 1;
  };

  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;

    // Bass note C2 (36)
    const bassF = midiToFreq(36);
    const bass = 0.25 * (Math.sin(2 * Math.PI * bassF * t) + 0.35 * Math.sin(4 * Math.PI * bassF * t));

    // Stereo piano chord C3 (48), E3 (52), G3 (55)
    const chordMidis = [48, 52, 55];
    let cL = 0;
    let cR = 0;
    chordMidis.forEach((m, idx) => {
      const f = midiToFreq(m);
      const w = 0.55 * Math.sin(2 * Math.PI * f * t) + 0.22 * Math.sin(4 * Math.PI * f * t);
      const pL = idx === 0 ? 0.78 : idx === 2 ? 0.22 : 0.5;
      cL += w * pL * 0.2;
      cR += w * (1 - pL) * 0.2;
    });

    // Centered vocal melody C5 (72) -> D5 (74)
    const vm = t < 1.2 ? 72 : 74;
    const vf = midiToFreq(vm);
    const vEnv = Math.sin(Math.PI * ((t % 1.2) / 1.2));
    const vocal =
      vEnv *
      0.38 *
      (Math.sin(2 * Math.PI * vf * t) +
        0.48 * Math.sin(4 * Math.PI * vf * t) +
        0.22 * Math.sin(6 * Math.PI * vf * t));

    // Add brief broadband percussive snare hits every 0.6s (20ms burst)
    const beatPhase = t % 0.6;
    const drum = beatPhase < 0.02 ? noiseRand() * 0.45 * Math.exp(-beatPhase * 120) : 0;

    left[i] = bass + cL + vocal + drum;
    right[i] = bass + cR + vocal + drum;
  }

  const piece = transcribePcmToPianoAndVocal(left, right, sampleRate, 'Prueba 4 Pistas');

  assert.equal(piece.source, 'upload');
  assert.ok(piece.countsByTrack.vocal >= 2, `Expected >=2 vocal notes, got ${piece.countsByTrack.vocal}`);
  assert.ok(piece.countsByTrack.bass >= 1, `Expected >=1 bass note, got ${piece.countsByTrack.bass}`);

  const vocalMidis = new Set(piece.notes.filter((ev) => ev.track === 'vocal').map((ev) => ev.midi));
  assert.ok(vocalMidis.has(72), `Expected vocal C5 (72), got ${[...vocalMidis]}`);
  assert.ok(vocalMidis.has(74), `Expected vocal D5 (74), got ${[...vocalMidis]}`);

  // Verify live cleanliness filter reduces noisy notes at strict level and respects track toggles
  const softNotes = filterAndCleanNotes(
    piece,
    { vocal: true, chords: true, bass: true, lead: true },
    'soft',
  );
  const strictNotes = filterAndCleanNotes(
    piece,
    { vocal: true, chords: true, bass: true, lead: true },
    'strict',
  );
  assert.ok(strictNotes.length <= softNotes.length);

  const vocalOnly = filterAndCleanNotes(
    piece,
    { vocal: true, chords: false, bass: false, lead: false },
    'balanced',
  );
  assert.ok(vocalOnly.every((ev) => ev.track === 'vocal'));

  // Verify multi-track Standard MIDI File export produces valid Format 1 binary header
  const midiBytes = buildMultiTrackMidi(piece, softNotes);
  const headerTag = String.fromCharCode(midiBytes[0], midiBytes[1], midiBytes[2], midiBytes[3]);
  assert.equal(headerTag, 'MThd');
  assert.equal(midiBytes[9], 1); // Format 1 (multi-track)
  assert.ok(midiBytes.length > 60);
});

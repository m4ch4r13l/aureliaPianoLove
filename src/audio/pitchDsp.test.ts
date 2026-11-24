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
  fftRadix2,
  transcribePcmToPianoAndVocal,
} from './pitchDsp';

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

test('transcribePcmToPianoAndVocal extracts both piano chord and vocal melody from synthesized stereo audio', () => {
  const sampleRate = 22050;
  const durationSec = 2.0;
  const n = Math.floor(sampleRate * durationSec);
  const left = new Float32Array(n);
  const right = new Float32Array(n);

  // Synthesize stereo backing piano chord (C3=48, E3=52, G3=55) panned slightly wide
  // + centered vocal melody (C5=72 for 0..1s, D5=74 for 1..2s) with rich harmonics
  const chordMidis = [48, 52, 55];
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;

    // Backing chord with stereo width
    let chordL = 0;
    let chordR = 0;
    for (let c = 0; c < chordMidis.length; c++) {
      const f = midiToFreq(chordMidis[c]);
      const wave =
        0.55 * Math.sin(2 * Math.PI * f * t) +
        0.25 * Math.sin(2 * Math.PI * f * 2 * t) +
        0.1 * Math.sin(2 * Math.PI * f * 3 * t);
      const panL = c === 0 ? 0.78 : c === 2 ? 0.25 : 0.52;
      const panR = 1 - panL;
      chordL += wave * panL * 0.24;
      chordR += wave * panR * 0.24;
    }

    // Centered vocal melody stepping from C5 (72) to D5 (74)
    const vocalMidi = t < 1.0 ? 72 : 74;
    const vf = midiToFreq(vocalMidi);
    const env = Math.sin(Math.PI * (t % 1.0));
    const vocal =
      env *
      0.38 *
      (Math.sin(2 * Math.PI * vf * t) +
        0.5 * Math.sin(2 * Math.PI * vf * 2 * t) +
        0.25 * Math.sin(2 * Math.PI * vf * 3 * t));

    left[i] = chordL + vocal;
    right[i] = chordR + vocal;
  }

  const piece = transcribePcmToPianoAndVocal(left, right, sampleRate, 'Prueba Estéreo');

  assert.equal(piece.source, 'upload');
  assert.ok(piece.vocalNoteCount >= 2, `Expected >=2 vocal notes, got ${piece.vocalNoteCount}`);
  assert.ok(piece.pianoNoteCount >= 2, `Expected >=2 piano notes, got ${piece.pianoNoteCount}`);

  const vocalMidis = new Set(piece.notes.filter((ev) => ev.track === 'vocal').map((ev) => ev.midi));
  assert.ok(vocalMidis.has(72), `Expected vocal C5 (72), got ${[...vocalMidis]}`);
  assert.ok(vocalMidis.has(74), `Expected vocal D5 (74), got ${[...vocalMidis]}`);

  const pianoMidis = new Set(piece.notes.filter((ev) => ev.track === 'piano').map((ev) => ev.midi));
  assert.ok(
    pianoMidis.has(48) || pianoMidis.has(52) || pianoMidis.has(55),
    `Expected piano chord notes (48/52/55), got ${[...pianoMidis]}`,
  );
});

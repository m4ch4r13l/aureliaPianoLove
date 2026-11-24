import test from 'node:test';
import assert from 'node:assert/strict';
import {
  build88KeyLayout,
  formatTime,
  midiToFreq,
  midiToSolfege,
} from './pianoTheory';
import {
  applyHarmonicPercussiveMask,
  decimateBy2,
  fftRadix2,
  filterAndCleanNotes,
  transcribePcmToPianoAndVocal,
  transcribePcmToPianoAndVocalAsync,
} from './pitchDsp';
import { buildMultiTrackMidi } from './midiExporter';

test('88-key grand piano layout generates 52 white and 36 black keys in ascending X order', () => {
  const { keys } = build88KeyLayout();
  assert.equal(keys.length, 88);
  assert.equal(keys.filter((k) => !k.isBlack).length, 52);
  assert.equal(keys.filter((k) => k.isBlack).length, 36);
  assert.equal(keys[0].midi, 21);
  assert.equal(keys[87].midi, 108);

  const whites = keys.filter((k) => !k.isBlack);
  for (let i = 1; i < whites.length; i++) {
    assert.ok(whites[i].x > whites[i - 1].x, 'White keys must increase strictly left-to-right');
  }
});

test('Spanish solfege naming and time formatting work accurately', () => {
  assert.equal(midiToSolfege(60), 'Do4');
  assert.equal(midiToSolfege(69), 'La4');
  assert.equal(midiToSolfege(61), 'Do#4');
  assert.ok(Math.abs(midiToFreq(69) - 440) < 1e-6);
  assert.equal(formatTime(0), '0:00');
  assert.equal(formatTime(75.2), '1:15');
});

test('Radix-2 FFT and anti-aliased decimation detect exact peak frequency', () => {
  const sr = 22050;
  const n = 2048;
  const real = new Float32Array(n);
  const imag = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    real[i] = Math.sin((2 * Math.PI * 440 * i) / sr);
  }
  fftRadix2(real, imag);

  let maxBin = 0;
  let maxMag = 0;
  for (let k = 1; k < n / 2; k++) {
    const mag = Math.hypot(real[k], imag[k]);
    if (mag > maxMag) {
      maxMag = mag;
      maxBin = k;
    }
  }
  const detectedHz = (maxBin * sr) / n;
  assert.ok(Math.abs(detectedHz - 440) < 12, `Expected ~440Hz, got ${detectedHz}`);

  // Verify 2:1 anti-aliased decimation preserves low-frequency sine and attenuates Nyquist
  const raw = new Float32Array(2048);
  for (let i = 0; i < raw.length; i++) {
    raw[i] = Math.sin((2 * Math.PI * 220 * i) / sr);
  }
  const dec = decimateBy2(raw);
  assert.equal(dec.length, 1024);
  const rms = Math.sqrt(dec.reduce((s, v) => s + v * v, 0) / dec.length);
  assert.ok(rms > 0.6 && rms < 0.8, `Expected clean sine RMS ~0.707, got ${rms}`);
});

test('HPSS suppresses broadband percussive drum spikes while preserving sustained harmonic tones', () => {
  const frames = 20;
  const bins = 40;
  const mat = new Float32Array(frames * bins);

  // Sustained harmonic note on bin 18 across all frames
  for (let f = 0; f < frames; f++) {
    mat[f * bins + 18] = 1.0;
  }
  // Single-frame broadband snare hit across ALL bins on frame 10
  for (let b = 0; b < bins; b++) {
    mat[10 * bins + b] += 1.2;
  }

  const cleaned = applyHarmonicPercussiveMask(mat, frames, bins, 3, 4);
  assert.ok(
    cleaned[10 * bins + 18] > 0.55,
    `Sustained harmonic tone must survive HPSS (got ${cleaned[10 * bins + 18]})`,
  );
  assert.ok(
    cleaned[10 * bins + 5] < 0.08,
    `Broadband drum spike at non-harmonic bin must be suppressed (got ${cleaned[10 * bins + 5]})`,
  );
});

function buildRealisticPopMix(): { left: Float32Array; right: Float32Array; sr: number } {
  const sr = 22050;
  const dur = 6.0;
  const n = Math.floor(sr * dur);
  const left = new Float32Array(n);
  const right = new Float32Array(n);

  // Deterministic PRNG for realistic drum snare & hi-hat bursts
  let seed = 4242;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed / 2147483647) * 2 - 1;
  };

  // Progression: 0-2s Am (A2 bass, A3-C4-E4 chord, E5 vocal),
  //              2-4s F  (F2 bass, A3-C4-F4 chord, F5 vocal),
  //              4-6s C  (C2 bass, G3-C4-E4 chord, G5 vocal)
  const sections = [
    { bass: 45, chord: [57, 60, 64], vocal: 76 }, // Am: A2, A3-C4-E4, E5
    { bass: 41, chord: [57, 60, 65], vocal: 77 }, // F:  F2, A3-C4-F4, F5
    { bass: 36, chord: [55, 60, 64], vocal: 79 }, // C:  C2, G3-C4-E4, G5
  ];

  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const secIdx = Math.min(2, Math.floor(t / 2.0));
    const sec = sections[secIdx];
    const localT = t - secIdx * 2.0;
    const env = Math.min(1, localT * 18) * (localT > 1.85 ? Math.max(0, (2.0 - localT) / 0.15) : 1);

    // Bass with realistic 2nd harmonic
    const bf = midiToFreq(sec.bass);
    const bass = env * (0.28 * Math.sin(2 * Math.PI * bf * t) + 0.11 * Math.sin(4 * Math.PI * bf * t));

    // Stereo-spread piano accompaniment chords
    let cL = 0;
    let cR = 0;
    for (let k = 0; k < sec.chord.length; k++) {
      const cf = midiToFreq(sec.chord[k]);
      const tone = env * (0.14 * Math.sin(2 * Math.PI * cf * t) + 0.05 * Math.sin(4 * Math.PI * cf * t));
      cL += tone * (k === 0 ? 0.9 : k === 1 ? 0.65 : 0.35);
      cR += tone * (k === 0 ? 0.35 : k === 1 ? 0.65 : 0.9);
    }

    // Center-panned singing melody with natural harmonics
    const vf = midiToFreq(sec.vocal);
    const vocal =
      env *
      (0.28 * Math.sin(2 * Math.PI * vf * t) +
        0.13 * Math.sin(4 * Math.PI * vf * t) +
        0.05 * Math.sin(6 * Math.PI * vf * t));

    // Loud percussive snare/hi-hat hits every 0.5s (broadband noise!)
    const beatPhase = t % 0.5;
    const drum = beatPhase < 0.025 ? 0.55 * rand() * Math.exp(-beatPhase * 120) : 0;

    left[i] = Math.max(-0.98, Math.min(0.98, bass + cL + vocal + drum));
    right[i] = Math.max(-0.98, Math.min(0.98, bass + cR + vocal - drum * 0.7));
  }

  return { left, right, sr };
}

test('New Constant-Q + Chord HMM engine cleanly separates vocal, bass, and voice-led chords despite drums', async () => {
  const { left, right, sr } = buildRealisticPopMix();
  const piece = transcribePcmToPianoAndVocal(left, right, sr, 'Cancion Prueba Pop');

  assert.equal(piece.source, 'upload');
  assert.ok(piece.notes.length > 0, 'Must extract clean notes');

  const vocals = piece.notes.filter((n) => n.track === 'vocal');
  const basses = piece.notes.filter((n) => n.track === 'bass');
  const chords = piece.notes.filter((n) => n.track === 'chords');

  assert.ok(vocals.length >= 2, `Expected vocal melody notes, got ${vocals.length}`);
  assert.ok(basses.length >= 2, `Expected bassline notes, got ${basses.length}`);
  assert.ok(chords.length >= 3, `Expected clean chord accompaniment notes, got ${chords.length}`);

  // Verify vocal melody detected actual target pitches (E5=76, F5=77, G5=79)
  const expectedVocal = new Set([76, 77, 79]);
  const hitVocal = vocals.some((n) => expectedVocal.has(n.midi));
  assert.ok(
    hitVocal,
    `Expected vocal pitches among 76/77/79, got: ${vocals.map((n) => n.midi).join(',')}`,
  );

  // Verify bass detected low target pitches (C2=36, F2=41, A2=45)
  const expectedBass = new Set([36, 41, 45]);
  const hitBass = basses.some((n) => expectedBass.has(n.midi));
  assert.ok(
    hitBass,
    `Expected bass pitches among 36/41/45, got: ${basses.map((n) => n.midi).join(',')}`,
  );

  // Verify clean note density (no chaotic FFT noise flood!)
  const strict = filterAndCleanNotes(
    piece,
    { vocal: true, chords: true, bass: true, lead: false },
    'strict',
  );
  const soft = filterAndCleanNotes(
    piece,
    { vocal: true, chords: true, bass: true, lead: true },
    'soft',
  );
  assert.ok(strict.length > 0, 'Strict filter must keep real musical notes');
  assert.ok(strict.length <= soft.length, 'Strict filter must be at least as selective as soft');
  assert.ok(strict.length < 70, `Strict notes must stay clean and uncluttered (got ${strict.length})`);

  // Verify async engine produces matching results and calls progress callback cleanly
  const stages: number[] = [];
  const asyncPiece = await transcribePcmToPianoAndVocalAsync(left, right, sr, 'Cancion Async', {
    onProgress: (p) => stages.push(p),
  });
  assert.ok(stages.length >= 4, 'Async engine must emit progress milestones');
  assert.equal(stages[stages.length - 1], 1);
  assert.equal(asyncPiece.notes.length, piece.notes.length);

  // Verify hardened SMF Format-1 multi-track MIDI export (including long title & overlap dedup)
  const longTitlePiece = { ...piece, title: 'A'.repeat(160) };
  const midiBytes = buildMultiTrackMidi(longTitlePiece, strict);
  assert.equal(String.fromCharCode(...midiBytes.slice(0, 4)), 'MThd');
  assert.equal(midiBytes[9], 1, 'MIDI file must be Format 1 (multi-track)');
  assert.ok(midiBytes[11] >= 2, 'MIDI file must include tempo track + instrument track(s)');
});

import { PianoNoteEvent, TranscribedPiece } from './pitchDsp';

interface MeasureChord {
  bass: number[];
  harmony: number[];
  melody: Array<{ beat: number; midi: number; dur: number; vel?: number }>;
}

function compileScore(
  id: string,
  title: string,
  subtitle: string,
  bpm: number,
  keySignature: string,
  beatsPerMeasure: number,
  measures: MeasureChord[],
): TranscribedPiece {
  const beatSec = 60 / bpm;
  const notes: PianoNoteEvent[] = [];
  let seq = 1;

  measures.forEach((m, mIdx) => {
    const barStart = mIdx * beatsPerMeasure * beatSec;

    // Arpeggiated left-hand bass & inner accompaniment
    m.bass.forEach((midi, i) => {
      const startTime = barStart + i * 0.5 * beatSec;
      notes.push({
        id: `${id}-p-${seq++}`,
        midi,
        startTime: Number(startTime.toFixed(3)),
        duration: Number((beatSec * (i === 0 ? beatsPerMeasure * 0.85 : 1.35)).toFixed(3)),
        velocity: i === 0 ? 0.66 : 0.48,
        track: 'piano',
      });
    });

    // Warm chord voicing on middle beats
    m.harmony.forEach((midi, idx) => {
      const startTime = barStart + (idx % 2 === 0 ? 1.0 : 2.5) * beatSec;
      if (startTime < (mIdx + 1) * beatsPerMeasure * beatSec) {
        notes.push({
          id: `${id}-h-${seq++}`,
          midi,
          startTime: Number(startTime.toFixed(3)),
          duration: Number((beatSec * 1.45).toFixed(3)),
          velocity: 0.44,
          track: 'piano',
        });
      }
    });

    // Expressive cantabile vocal melody line
    m.melody.forEach((note) => {
      notes.push({
        id: `${id}-v-${seq++}`,
        midi: note.midi,
        startTime: Number((barStart + note.beat * beatSec).toFixed(3)),
        duration: Number((note.dur * beatSec).toFixed(3)),
        velocity: note.vel ?? 0.86,
        track: 'vocal',
      });
    });
  });

  notes.sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);
  const duration = Number((measures.length * beatsPerMeasure * beatSec + 1.2).toFixed(2));
  const pianoNoteCount = notes.filter((n) => n.track === 'piano').length;
  const vocalNoteCount = notes.filter((n) => n.track === 'vocal').length;

  return {
    id,
    title,
    subtitle,
    source: 'demo',
    duration,
    bpm,
    keySignature,
    notes,
    pianoNoteCount,
    vocalNoteCount,
  };
}

export const DEMO_PIECES: TranscribedPiece[] = [
  compileScore(
    'nocturno-mi-menor',
    'Nocturno en Mi menor',
    'Arreglo de concierto · Piano y línea vocal',
    86,
    'Mi menor',
    4,
    [
      // Em9
      {
        bass: [40, 47, 52, 55, 59, 55, 52, 47],
        harmony: [55, 59, 64],
        melody: [
          { beat: 0, midi: 71, dur: 1.5, vel: 0.84 },
          { beat: 1.5, midi: 74, dur: 0.5, vel: 0.8 },
          { beat: 2, midi: 76, dur: 1.5, vel: 0.9 },
          { beat: 3.5, midi: 74, dur: 0.5, vel: 0.76 },
        ],
      },
      // Cmaj7
      {
        bass: [36, 43, 48, 52, 55, 52, 48, 43],
        harmony: [52, 55, 59],
        melody: [
          { beat: 0, midi: 72, dur: 2.0, vel: 0.86 },
          { beat: 2, midi: 71, dur: 1.0, vel: 0.82 },
          { beat: 3, midi: 67, dur: 1.0, vel: 0.78 },
        ],
      },
      // Am9
      {
        bass: [45, 52, 57, 60, 64, 60, 57, 52],
        harmony: [57, 60, 64],
        melody: [
          { beat: 0, midi: 69, dur: 1.5, vel: 0.84 },
          { beat: 1.5, midi: 71, dur: 0.5, vel: 0.82 },
          { beat: 2, midi: 72, dur: 1.0, vel: 0.88 },
          { beat: 3, midi: 74, dur: 1.0, vel: 0.86 },
        ],
      },
      // B7sus4 -> B7
      {
        bass: [35, 42, 47, 51, 54, 51, 47, 42],
        harmony: [54, 57, 63],
        melody: [
          { beat: 0, midi: 71, dur: 2.5, vel: 0.85 },
          { beat: 2.5, midi: 69, dur: 0.5, vel: 0.78 },
          { beat: 3, midi: 66, dur: 1.0, vel: 0.76 },
        ],
      },
      // Em
      {
        bass: [40, 47, 52, 59, 64, 59, 55, 52],
        harmony: [55, 59, 64],
        melody: [
          { beat: 0, midi: 76, dur: 1.5, vel: 0.92 },
          { beat: 1.5, midi: 78, dur: 0.5, vel: 0.88 },
          { beat: 2, midi: 79, dur: 1.5, vel: 0.95 },
          { beat: 3.5, midi: 76, dur: 0.5, vel: 0.82 },
        ],
      },
      // D/F#
      {
        bass: [42, 50, 54, 57, 62, 57, 54, 50],
        harmony: [54, 57, 62],
        melody: [
          { beat: 0, midi: 74, dur: 2.0, vel: 0.88 },
          { beat: 2, midi: 72, dur: 1.0, vel: 0.82 },
          { beat: 3, midi: 71, dur: 1.0, vel: 0.8 },
        ],
      },
      // Cmaj7 -> B7
      {
        bass: [36, 48, 52, 55, 35, 47, 51, 54],
        harmony: [55, 59, 63],
        melody: [
          { beat: 0, midi: 69, dur: 1.5, vel: 0.84 },
          { beat: 1.5, midi: 67, dur: 0.5, vel: 0.8 },
          { beat: 2, midi: 66, dur: 1.5, vel: 0.84 },
          { beat: 3.5, midi: 69, dur: 0.5, vel: 0.78 },
        ],
      },
      // Em final resolution
      {
        bass: [40, 47, 52, 55, 59, 64],
        harmony: [52, 55, 59, 64],
        melody: [
          { beat: 0, midi: 64, dur: 3.5, vel: 0.85 },
        ],
      },
    ],
  ),
  compileScore(
    'vals-salon-do-mayor',
    'Vals de Salón en Do mayor',
    'Pieza lírica · Acompañamiento y voz cantabile',
    112,
    'Do mayor',
    3,
    [
      {
        bass: [36, 48, 52, 55, 48, 52],
        harmony: [52, 55, 60],
        melody: [
          { beat: 0, midi: 72, dur: 1.5, vel: 0.86 },
          { beat: 1.5, midi: 74, dur: 0.5, vel: 0.8 },
          { beat: 2, midi: 76, dur: 1.0, vel: 0.88 },
        ],
      },
      {
        bass: [43, 50, 55, 59, 50, 55],
        harmony: [55, 59, 62],
        melody: [
          { beat: 0, midi: 79, dur: 2.0, vel: 0.92 },
          { beat: 2, midi: 76, dur: 1.0, vel: 0.82 },
        ],
      },
      {
        bass: [45, 52, 57, 60, 52, 57],
        harmony: [57, 60, 64],
        melody: [
          { beat: 0, midi: 77, dur: 1.0, vel: 0.86 },
          { beat: 1, midi: 76, dur: 1.0, vel: 0.84 },
          { beat: 2, midi: 74, dur: 1.0, vel: 0.82 },
        ],
      },
      {
        bass: [41, 48, 53, 57, 48, 53],
        harmony: [53, 57, 60],
        melody: [
          { beat: 0, midi: 72, dur: 2.5, vel: 0.84 },
        ],
      },
      {
        bass: [38, 45, 50, 53, 45, 50],
        harmony: [53, 57, 62],
        melody: [
          { beat: 0, midi: 74, dur: 1.5, vel: 0.86 },
          { beat: 1.5, midi: 76, dur: 0.5, vel: 0.82 },
          { beat: 2, midi: 77, dur: 1.0, vel: 0.88 },
        ],
      },
      {
        bass: [43, 50, 55, 59, 53, 55],
        harmony: [55, 59, 62],
        melody: [
          { beat: 0, midi: 76, dur: 1.5, vel: 0.86 },
          { beat: 1.5, midi: 74, dur: 0.5, vel: 0.8 },
          { beat: 2, midi: 71, dur: 1.0, vel: 0.82 },
        ],
      },
      {
        bass: [36, 48, 52, 55, 60],
        harmony: [52, 55, 60],
        melody: [
          { beat: 0, midi: 72, dur: 2.8, vel: 0.88 },
        ],
      },
    ],
  ),
  compileScore(
    'balada-la-menor',
    'Balada Vocal en La menor',
    'Cadencia cinematográfica · Acordes graves y soprano',
    76,
    'La menor',
    4,
    [
      {
        bass: [33, 45, 52, 57, 60, 57, 52, 45],
        harmony: [57, 60, 64],
        melody: [
          { beat: 0, midi: 69, dur: 1.0, vel: 0.85 },
          { beat: 1, midi: 72, dur: 1.0, vel: 0.88 },
          { beat: 2, midi: 76, dur: 1.5, vel: 0.92 },
          { beat: 3.5, midi: 74, dur: 0.5, vel: 0.8 },
        ],
      },
      {
        bass: [41, 48, 53, 57, 60, 57, 53, 48],
        harmony: [53, 57, 60],
        melody: [
          { beat: 0, midi: 72, dur: 2.0, vel: 0.86 },
          { beat: 2, midi: 69, dur: 2.0, vel: 0.82 },
        ],
      },
      {
        bass: [36, 43, 48, 55, 60, 55, 52, 48],
        harmony: [52, 55, 60],
        melody: [
          { beat: 0, midi: 74, dur: 1.5, vel: 0.88 },
          { beat: 1.5, midi: 76, dur: 0.5, vel: 0.84 },
          { beat: 2, midi: 79, dur: 2.0, vel: 0.94 },
        ],
      },
      {
        bass: [40, 47, 52, 56, 59, 56, 52, 47],
        harmony: [56, 59, 64],
        melody: [
          { beat: 0, midi: 76, dur: 2.0, vel: 0.88 },
          { beat: 2, midi: 74, dur: 1.0, vel: 0.82 },
          { beat: 3, midi: 71, dur: 1.0, vel: 0.8 },
        ],
      },
      {
        bass: [33, 45, 52, 57, 60, 64],
        harmony: [57, 60, 64],
        melody: [
          { beat: 0, midi: 69, dur: 3.5, vel: 0.88 },
        ],
      },
    ],
  ),
];

import { PianoNoteEvent, TrackRole, TranscribedPiece } from './pitchDsp';

interface MeasureChord {
  bass: number[];
  chords: number[];
  lead: Array<{ beat: number; midi: number; dur: number }>;
  melody: Array<{ beat: number; midi: number; dur: number; vel?: number }>;
}

function compileScore(
  id: string,
  title: string,
  subtitle: string,
  bpm: number,
  keySignature: string,
  keyRoot: number,
  keyMode: 'major' | 'minor',
  beatsPerMeasure: number,
  measures: MeasureChord[],
): TranscribedPiece {
  const beatSec = 60 / bpm;
  const notes: PianoNoteEvent[] = [];
  let seq = 1;

  measures.forEach((m, mIdx) => {
    const barStart = mIdx * beatsPerMeasure * beatSec;

    // Monophonic warm bass foundation
    m.bass.forEach((midi, i) => {
      const startTime = barStart + i * (beatsPerMeasure / Math.max(1, m.bass.length)) * beatSec;
      notes.push({
        id: `${id}-b-${seq++}`,
        midi,
        startTime: Number(startTime.toFixed(3)),
        duration: Number((beatSec * (beatsPerMeasure / Math.max(1, m.bass.length)) * 0.92).toFixed(3)),
        velocity: i === 0 ? 0.68 : 0.56,
        track: 'bass',
        confidence: 1,
        inKey: true,
      });
    });

    // Piano mid-register chord arpeggio / voicing
    m.chords.forEach((midi, idx) => {
      const startTime = barStart + idx * 0.5 * beatSec;
      if (startTime < (mIdx + 1) * beatsPerMeasure * beatSec) {
        notes.push({
          id: `${id}-c-${seq++}`,
          midi,
          startTime: Number(startTime.toFixed(3)),
          duration: Number((beatSec * 1.35).toFixed(3)),
          velocity: 0.46,
          track: 'chords',
          confidence: 1,
          inKey: true,
        });
      }
    });

    // Subtle counter-melody / upper arpeggio detail
    m.lead.forEach((note) => {
      notes.push({
        id: `${id}-l-${seq++}`,
        midi: note.midi,
        startTime: Number((barStart + note.beat * beatSec).toFixed(3)),
        duration: Number((note.dur * beatSec).toFixed(3)),
        velocity: 0.54,
        track: 'lead',
        confidence: 1,
        inKey: true,
      });
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
        confidence: 1,
        inKey: true,
      });
    });
  });

  notes.sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);
  const duration = Number((measures.length * beatsPerMeasure * beatSec + 1.2).toFixed(2));
  const countsByTrack: Record<TrackRole, number> = {
    vocal: notes.filter((n) => n.track === 'vocal').length,
    chords: notes.filter((n) => n.track === 'chords').length,
    bass: notes.filter((n) => n.track === 'bass').length,
    lead: notes.filter((n) => n.track === 'lead').length,
  };

  return {
    id,
    title,
    subtitle,
    source: 'demo',
    duration,
    bpm,
    keySignature,
    keyRoot,
    keyMode,
    notes,
    countsByTrack,
  };
}

export const DEMO_PIECES: TranscribedPiece[] = [
  compileScore(
    'nocturno-mi-menor',
    'Nocturno en Mi menor',
    'Arreglo de concierto · 4 pistas separadas',
    86,
    'Mi menor',
    4,
    'minor',
    4,
    [
      {
        bass: [40, 47],
        chords: [52, 55, 59, 55, 52, 55, 59, 64],
        lead: [{ beat: 2.5, midi: 83, dur: 1.0 }],
        melody: [
          { beat: 0, midi: 71, dur: 1.5, vel: 0.84 },
          { beat: 1.5, midi: 74, dur: 0.5, vel: 0.8 },
          { beat: 2, midi: 76, dur: 1.5, vel: 0.9 },
          { beat: 3.5, midi: 74, dur: 0.5, vel: 0.76 },
        ],
      },
      {
        bass: [36, 43],
        chords: [48, 52, 55, 59, 55, 52, 48, 52],
        lead: [{ beat: 1.5, midi: 79, dur: 1.0 }],
        melody: [
          { beat: 0, midi: 72, dur: 2.0, vel: 0.86 },
          { beat: 2, midi: 71, dur: 1.0, vel: 0.82 },
          { beat: 3, midi: 67, dur: 1.0, vel: 0.78 },
        ],
      },
      {
        bass: [45, 52],
        chords: [52, 57, 60, 64, 60, 57, 52, 57],
        lead: [{ beat: 2.5, midi: 81, dur: 1.0 }],
        melody: [
          { beat: 0, midi: 69, dur: 1.5, vel: 0.84 },
          { beat: 1.5, midi: 71, dur: 0.5, vel: 0.82 },
          { beat: 2, midi: 72, dur: 1.0, vel: 0.88 },
          { beat: 3, midi: 74, dur: 1.0, vel: 0.86 },
        ],
      },
      {
        bass: [35, 42],
        chords: [47, 51, 54, 59, 54, 51, 47, 51],
        lead: [{ beat: 1.0, midi: 78, dur: 1.5 }],
        melody: [
          { beat: 0, midi: 71, dur: 2.5, vel: 0.85 },
          { beat: 2.5, midi: 69, dur: 0.5, vel: 0.78 },
          { beat: 3, midi: 66, dur: 1.0, vel: 0.76 },
        ],
      },
      {
        bass: [40, 47],
        chords: [52, 55, 59, 64, 59, 55, 52, 55],
        lead: [{ beat: 2.5, midi: 83, dur: 1.0 }],
        melody: [
          { beat: 0, midi: 76, dur: 1.5, vel: 0.92 },
          { beat: 1.5, midi: 78, dur: 0.5, vel: 0.88 },
          { beat: 2, midi: 79, dur: 1.5, vel: 0.95 },
          { beat: 3.5, midi: 76, dur: 0.5, vel: 0.82 },
        ],
      },
      {
        bass: [42, 50],
        chords: [50, 54, 57, 62, 57, 54, 50, 54],
        lead: [{ beat: 1.5, midi: 81, dur: 1.0 }],
        melody: [
          { beat: 0, midi: 74, dur: 2.0, vel: 0.88 },
          { beat: 2, midi: 72, dur: 1.0, vel: 0.82 },
          { beat: 3, midi: 71, dur: 1.0, vel: 0.8 },
        ],
      },
      {
        bass: [36, 35],
        chords: [48, 52, 55, 52, 47, 51, 54, 51],
        lead: [{ beat: 2.0, midi: 78, dur: 1.5 }],
        melody: [
          { beat: 0, midi: 69, dur: 1.5, vel: 0.84 },
          { beat: 1.5, midi: 67, dur: 0.5, vel: 0.8 },
          { beat: 2, midi: 66, dur: 1.5, vel: 0.84 },
          { beat: 3.5, midi: 69, dur: 0.5, vel: 0.78 },
        ],
      },
      {
        bass: [40],
        chords: [52, 55, 59, 64],
        lead: [{ beat: 1.5, midi: 79, dur: 2.0 }],
        melody: [{ beat: 0, midi: 64, dur: 3.5, vel: 0.85 }],
      },
    ],
  ),
  compileScore(
    'vals-salon-do-mayor',
    'Vals de Salón en Do mayor',
    'Pieza lírica · Voz, acordes, bajo y arreglos',
    112,
    'Do mayor',
    0,
    'major',
    3,
    [
      {
        bass: [36],
        chords: [48, 52, 55, 52, 55, 60],
        lead: [{ beat: 1.0, midi: 79, dur: 1.0 }],
        melody: [
          { beat: 0, midi: 72, dur: 1.5, vel: 0.86 },
          { beat: 1.5, midi: 74, dur: 0.5, vel: 0.8 },
          { beat: 2, midi: 76, dur: 1.0, vel: 0.88 },
        ],
      },
      {
        bass: [43],
        chords: [50, 55, 59, 55, 59, 62],
        lead: [{ beat: 1.5, midi: 83, dur: 1.0 }],
        melody: [
          { beat: 0, midi: 79, dur: 2.0, vel: 0.92 },
          { beat: 2, midi: 76, dur: 1.0, vel: 0.82 },
        ],
      },
      {
        bass: [45],
        chords: [52, 57, 60, 57, 60, 64],
        lead: [{ beat: 2.0, midi: 81, dur: 0.8 }],
        melody: [
          { beat: 0, midi: 77, dur: 1.0, vel: 0.86 },
          { beat: 1, midi: 76, dur: 1.0, vel: 0.84 },
          { beat: 2, midi: 74, dur: 1.0, vel: 0.82 },
        ],
      },
      {
        bass: [41],
        chords: [48, 53, 57, 53, 57, 60],
        lead: [{ beat: 1.0, midi: 77, dur: 1.5 }],
        melody: [{ beat: 0, midi: 72, dur: 2.5, vel: 0.84 }],
      },
      {
        bass: [38],
        chords: [45, 50, 53, 50, 53, 57],
        lead: [{ beat: 2.0, midi: 81, dur: 0.9 }],
        melody: [
          { beat: 0, midi: 74, dur: 1.5, vel: 0.86 },
          { beat: 1.5, midi: 76, dur: 0.5, vel: 0.82 },
          { beat: 2, midi: 77, dur: 1.0, vel: 0.88 },
        ],
      },
      {
        bass: [43],
        chords: [50, 55, 59, 53, 55, 59],
        lead: [{ beat: 1.0, midi: 79, dur: 1.0 }],
        melody: [
          { beat: 0, midi: 76, dur: 1.5, vel: 0.86 },
          { beat: 1.5, midi: 74, dur: 0.5, vel: 0.8 },
          { beat: 2, midi: 71, dur: 1.0, vel: 0.82 },
        ],
      },
      {
        bass: [36],
        chords: [48, 52, 55, 60],
        lead: [{ beat: 1.0, midi: 84, dur: 1.8 }],
        melody: [{ beat: 0, midi: 72, dur: 2.8, vel: 0.88 }],
      },
    ],
  ),
  compileScore(
    'balada-la-menor',
    'Balada Vocal en La menor',
    'Cadencia cinematográfica · 4 pistas separadas',
    76,
    'La menor',
    9,
    'minor',
    4,
    [
      {
        bass: [33, 45],
        chords: [52, 57, 60, 57, 52, 57, 60, 64],
        lead: [{ beat: 2.5, midi: 81, dur: 1.2 }],
        melody: [
          { beat: 0, midi: 69, dur: 1.0, vel: 0.85 },
          { beat: 1, midi: 72, dur: 1.0, vel: 0.88 },
          { beat: 2, midi: 76, dur: 1.5, vel: 0.92 },
          { beat: 3.5, midi: 74, dur: 0.5, vel: 0.8 },
        ],
      },
      {
        bass: [41, 48],
        chords: [53, 57, 60, 57, 53, 57, 60, 57],
        lead: [{ beat: 1.5, midi: 77, dur: 1.5 }],
        melody: [
          { beat: 0, midi: 72, dur: 2.0, vel: 0.86 },
          { beat: 2, midi: 69, dur: 2.0, vel: 0.82 },
        ],
      },
      {
        bass: [36, 43],
        chords: [48, 55, 60, 55, 52, 55, 60, 55],
        lead: [{ beat: 2.5, midi: 84, dur: 1.2 }],
        melody: [
          { beat: 0, midi: 74, dur: 1.5, vel: 0.88 },
          { beat: 1.5, midi: 76, dur: 0.5, vel: 0.84 },
          { beat: 2, midi: 79, dur: 2.0, vel: 0.94 },
        ],
      },
      {
        bass: [40, 47],
        chords: [52, 56, 59, 56, 52, 56, 59, 64],
        lead: [{ beat: 1.5, midi: 80, dur: 1.2 }],
        melody: [
          { beat: 0, midi: 76, dur: 2.0, vel: 0.88 },
          { beat: 2, midi: 74, dur: 1.0, vel: 0.82 },
          { beat: 3, midi: 71, dur: 1.0, vel: 0.8 },
        ],
      },
      {
        bass: [33],
        chords: [45, 52, 57, 60, 64],
        lead: [{ beat: 1.5, midi: 81, dur: 2.0 }],
        melody: [{ beat: 0, midi: 69, dur: 3.5, vel: 0.88 }],
      },
    ],
  ),
];

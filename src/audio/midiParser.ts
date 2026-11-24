import * as ToneMidiPkg from '@tonejs/midi';
import { MAX_MIDI, MIN_MIDI } from './pianoTheory';

const Midi = (ToneMidiPkg.Midi ??
  (ToneMidiPkg as unknown as { default?: { Midi?: typeof ToneMidiPkg.Midi } }).default?.Midi)!;
import {
  estimateKeyFromChroma,
  isPitchInKey,
  PianoNoteEvent,
  TrackRole,
  TranscribedPiece,
} from './pitchDsp';

const KEY_NAME_TO_ROOT: Record<string, number> = {
  C: 0,
  'C#': 1,
  Db: 1,
  D: 2,
  'D#': 3,
  Eb: 3,
  E: 4,
  F: 5,
  'F#': 6,
  Gb: 6,
  G: 7,
  'G#': 8,
  Ab: 8,
  A: 9,
  'A#': 10,
  Bb: 10,
  B: 11,
  Cb: 11,
};

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

function readAscii(data: Uint8Array, offset: number, length: number): string {
  let out = '';
  const end = Math.min(data.length, offset + length);
  for (let i = offset; i < end; i++) {
    out += String.fromCharCode(data[i]);
  }
  return out;
}

/**
 * Checks if a File looks like a Standard MIDI File (.mid / .midi / .kar).
 */
export function isMidiFile(file: File): boolean {
  const name = file.name.toLowerCase();
  const type = file.type.toLowerCase();
  return (
    name.endsWith('.mid') ||
    name.endsWith('.midi') ||
    name.endsWith('.kar') ||
    type === 'audio/midi' ||
    type === 'audio/x-midi' ||
    type === 'audio/mid' ||
    type === 'application/x-midi'
  );
}

/**
 * Checks raw bytes for a Standard MIDI File header (`MThd` or RIFF RMID wrapper).
 */
export function hasMidiHeader(bytes: Uint8Array): boolean {
  if (bytes.length < 14) return false;
  if (readAscii(bytes, 0, 4) === 'MThd') return true;
  if (readAscii(bytes, 0, 4) === 'RIFF') {
    const scanLimit = Math.min(bytes.length - 14, 128);
    for (let i = 12; i < scanLimit; i++) {
      if (
        bytes[i] === 0x4d &&
        bytes[i + 1] === 0x54 &&
        bytes[i + 2] === 0x68 &&
        bytes[i + 3] === 0x64
      ) {
        return true;
      }
    }
  }
  return false;
}

function stripRiffWrapperIfNeeded(bytes: Uint8Array): Uint8Array {
  if (bytes.length >= 14 && readAscii(bytes, 0, 4) === 'MThd') return bytes;
  const scanLimit = Math.min(bytes.length - 14, 256);
  for (let i = 0; i < scanLimit; i++) {
    if (
      bytes[i] === 0x4d &&
      bytes[i + 1] === 0x54 &&
      bytes[i + 2] === 0x68 &&
      bytes[i + 3] === 0x64
    ) {
      return bytes.subarray(i);
    }
  }
  return bytes;
}

function isDrumTrackName(name: string): boolean {
  return /\b(drum|drums|perc|percussion|bateria|batería|kit|hihat|snare|kick)\b/i.test(name);
}

function clampTo88Keys(midi: number): number {
  let m = Math.round(midi);
  while (m < MIN_MIDI) m += 12;
  while (m > MAX_MIDI) m -= 12;
  return m;
}

function classifyTrackRole(
  trackName: string,
  gmProgram: number,
  medianMidi: number,
  isSingleMelodicTrack: boolean,
): TrackRole | null {
  const lower = trackName.toLowerCase();

  if (/\b(vocal|voz|melod|melody|voice|sing|cant|soprano|tenor|alto|lead\s*vox|flute|sax|violin)\b/i.test(lower)) {
    return 'vocal';
  }
  if (/\b(bass|bajo|contrabajo|sub|left\s*hand|mano\s*izq|lh)\b/i.test(lower)) {
    return 'bass';
  }
  if (/\b(guitar|guitarra|arregl|arp|arpeggio|synth|strings|cuerdas|lead|solo|bell)\b/i.test(lower)) {
    return 'lead';
  }
  if (/\b(chord|acord|piano|keys|teclado|pad|organ|rhodes|harpsichord|right\s*hand|mano\s*der|rh)\b/i.test(lower)) {
    return isSingleMelodicTrack ? null : 'chords';
  }

  if (isSingleMelodicTrack) return null;

  if (gmProgram >= 32 && gmProgram <= 39) return 'bass';
  if ((gmProgram >= 52 && gmProgram <= 54) || (gmProgram >= 64 && gmProgram <= 87)) return 'vocal';
  if ((gmProgram >= 24 && gmProgram <= 31) || (gmProgram >= 40 && gmProgram <= 51)) return 'lead';
  if (medianMidi < 50) return 'bass';
  if (medianMidi >= 70) return 'vocal';
  return 'chords';
}

/**
 * Parses any Standard MIDI File (.mid / .midi) using @tonejs/midi (`Midi` class),
 * preserving exact tempo maps, key signatures, sustain pedal extensions (CC64),
 * skipping channel 10 percussion, and classifying notes into Voz, Acordes, Bajo & Arreglos.
 */
export function parseMidiToPiece(
  bytes: Uint8Array,
  fallbackTitle = 'Archivo MIDI',
): TranscribedPiece {
  const normalized = stripRiffWrapperIfNeeded(bytes);
  if (!hasMidiHeader(normalized)) {
    throw new Error('El archivo no contiene una cabecera MIDI (MThd) válida.');
  }

  const midi = new Midi(normalized);

  // Filter out empty tracks and percussion tracks (GM channel 9 / drum track names)
  const melodicTracks = midi.tracks.filter(
    (t) =>
      t.notes.length > 0 &&
      t.channel !== 9 &&
      !t.instrument.percussion &&
      !isDrumTrackName(t.name || ''),
  );

  if (melodicTracks.length === 0) {
    throw new Error('El archivo MIDI no contiene notas melódicas reproducibles.');
  }

  const isSingleTrackPiano = melodicTracks.length === 1;

  interface ExtractedNote {
    trackIdx: number;
    role: TrackRole | null;
    midi: number;
    startTime: number;
    duration: number;
    velocity: number;
  }

  const extracted: ExtractedNote[] = [];

  for (let ti = 0; ti < melodicTracks.length; ti++) {
    const track = melodicTracks[ti];
    const sortedMidis = track.notes.map((n) => n.midi).sort((a, b) => a - b);
    const medianMidi = sortedMidis[sortedMidis.length >> 1];
    const trackRole = classifyTrackRole(
      track.name || '',
      track.instrument.number ?? 0,
      medianMidi,
      isSingleTrackPiano,
    );

    // Extract CC64 sustain pedal intervals so piano MIDI sustain sounds rich & accurate
    const sustainCCs = (track.controlChanges[64] ?? []).slice().sort((a, b) => a.time - b.time);
    const sustainIntervals: Array<{ start: number; end: number }> = [];
    let pedalDownAt: number | null = null;
    for (const cc of sustainCCs) {
      if (cc.value >= 0.5 && pedalDownAt === null) {
        pedalDownAt = cc.time;
      } else if (cc.value < 0.5 && pedalDownAt !== null) {
        sustainIntervals.push({ start: pedalDownAt, end: cc.time });
        pedalDownAt = null;
      }
    }

    for (const n of track.notes) {
      let endTime = n.time + n.duration;
      // Extend note release to sustain pedal lift if key was released while pedal was down
      for (const sInt of sustainIntervals) {
        if (endTime >= sInt.start - 0.005 && endTime <= sInt.end) {
          endTime = Math.max(endTime, sInt.end);
          break;
        }
      }
      extracted.push({
        trackIdx: ti,
        role: trackRole,
        midi: clampTo88Keys(n.midi),
        startTime: Math.max(0, n.time),
        duration: Math.max(0.06, Math.min(6.5, endTime - n.time)),
        velocity: Math.max(0.2, Math.min(1, n.velocity || 0.75)),
      });
    }
  }

  extracted.sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);
  const firstStart = extracted[0].startTime;
  const trimOffset = firstStart > 1.5 ? firstStart - 0.25 : 0;

  // Read embedded MIDI key signature or estimate from chroma
  let keyInfo: { root: number; mode: 'major' | 'minor'; label: string } | null = null;
  if (midi.header.keySignatures.length > 0) {
    const ks = midi.header.keySignatures[0];
    const root = KEY_NAME_TO_ROOT[ks.key];
    if (root !== undefined) {
      const mode: 'major' | 'minor' = ks.scale === 'minor' ? 'minor' : 'major';
      keyInfo = {
        root,
        mode,
        label: `${NOTE_NAMES_ES[root]} ${mode === 'major' ? 'mayor' : 'menor'}`,
      };
    }
  }
  if (!keyInfo) {
    const chroma = new Float64Array(12);
    for (const n of extracted) chroma[n.midi % 12] += n.duration * n.velocity;
    keyInfo = estimateKeyFromChroma(chroma);
  }

  // For single-track piano MIDIs, identify top melody voice per ~62ms onset bucket
  const topMidiByBucket = new Map<number, number>();
  if (isSingleTrackPiano) {
    for (const n of extracted) {
      const bucket = Math.round((n.startTime - trimOffset) * 16);
      const prev = topMidiByBucket.get(bucket) ?? -1;
      if (n.midi > prev) topMidiByBucket.set(bucket, n.midi);
    }
  }

  const countsByTrack: Record<TrackRole, number> = {
    vocal: 0,
    chords: 0,
    bass: 0,
    lead: 0,
  };

  let maxEnd = 0;
  const notes: PianoNoteEvent[] = extracted.map((n, idx) => {
    const startTime = Math.max(0, Number((n.startTime - trimOffset).toFixed(3)));
    const duration = Number(n.duration.toFixed(3));
    if (startTime + duration > maxEnd) maxEnd = startTime + duration;

    let role = n.role;
    if (!role) {
      const bucket = Math.round(startTime * 16);
      const isTopVoice = (topMidiByBucket.get(bucket) ?? -1) === n.midi;
      if (n.midi < 50) role = 'bass';
      else if (isTopVoice && n.midi >= 64) role = 'vocal';
      else role = 'chords';
    }

    countsByTrack[role]++;
    return {
      id: `m-${idx + 1}`,
      midi: n.midi,
      startTime,
      duration,
      velocity: Number(n.velocity.toFixed(2)),
      track: role,
      confidence: 1.0,
      inKey: isPitchInKey(n.midi, keyInfo.root, keyInfo.mode),
    };
  });

  const primaryBpm = midi.header.tempos[0]?.bpm ?? 120;
  const bpm = Math.max(40, Math.min(240, Math.round(primaryBpm)));

  const cleanFallback = fallbackTitle.trim();
  const title =
    cleanFallback && cleanFallback !== 'Archivo MIDI'
      ? cleanFallback
      : midi.header.name?.trim() || 'Archivo MIDI';

  return {
    id: `midi-${Date.now()}`,
    title,
    subtitle: 'Partitura MIDI importada con @tonejs/midi',
    source: 'midi',
    duration: Number(Math.max(1, maxEnd + 0.4).toFixed(2)),
    bpm,
    keySignature: keyInfo.label,
    keyRoot: keyInfo.root,
    keyMode: keyInfo.mode,
    notes,
    countsByTrack,
  };
}

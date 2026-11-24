import * as ToneMidiPkg from '@tonejs/midi';
import { PianoNoteEvent, TrackRole, TranscribedPiece } from './pitchDsp';

const Midi = (ToneMidiPkg.Midi ??
  (ToneMidiPkg as unknown as { default?: { Midi?: typeof ToneMidiPkg.Midi } }).default?.Midi)!;

const TRACK_SPECS: Array<{
  role: TrackRole;
  name: string;
  channel: number;
  gmProgram: number;
  volume: number; // 0..1 normalized for @tonejs/midi CC7
}> = [
  { role: 'vocal', name: 'Melodia Vocal', channel: 0, gmProgram: 0, volume: 0.93 },
  { role: 'chords', name: 'Acordes Piano', channel: 1, gmProgram: 0, volume: 0.76 },
  { role: 'bass', name: 'Bajo', channel: 2, gmProgram: 32, volume: 0.82 },
  { role: 'lead', name: 'Arreglos Guitarra', channel: 3, gmProgram: 24, volume: 0.7 },
];

/**
 * Deduplicates overlapping same-pitch notes within a single track so DAWs and
 * MIDI players (Synthesia, MuseScore, FL Studio, Logic) play cleanly with zero stuck notes.
 */
function sanitizeTrackNotes(
  notes: PianoNoteEvent[],
): Array<{ midi: number; time: number; duration: number; velocity: number }> {
  const mapped = notes
    .map((n) => ({
      midi: Math.max(21, Math.min(108, Math.round(n.midi))),
      time: Math.max(0, n.startTime),
      duration: Math.max(0.05, n.duration),
      velocity: Math.max(0.2, Math.min(1, n.velocity)),
    }))
    .sort((a, b) => a.time - b.time || a.midi - b.midi);

  const lastByPitch = new Map<
    number,
    { midi: number; time: number; duration: number; velocity: number }
  >();
  const clean: Array<{ midi: number; time: number; duration: number; velocity: number }> = [];

  for (const item of mapped) {
    const prev = lastByPitch.get(item.midi);
    if (prev && prev.time + prev.duration > item.time) {
      if (item.time - prev.time < 0.025) {
        const end = Math.max(prev.time + prev.duration, item.time + item.duration);
        prev.duration = end - prev.time;
        prev.velocity = Math.max(prev.velocity, item.velocity);
        continue;
      }
      prev.duration = Math.max(0.03, item.time - prev.time - 0.008);
    }
    clean.push(item);
    lastByPitch.set(item.midi, item);
  }

  return clean;
}

/**
 * Encodes a cleaned multi-instrument transcription into a Standard MIDI File (Format 1)
 * using @tonejs/midi, with separate named tracks, channels, GM instruments, and balanced volume.
 */
export function buildMultiTrackMidi(
  piece: TranscribedPiece,
  filteredNotes: PianoNoteEvent[],
): Uint8Array {
  const midi = new Midi();
  const bpm = Math.max(40, Math.min(240, Math.round(piece.bpm || 100)));
  const safeTitle =
    piece.title.replace(/[^\x20-\x7e]/g, '').trim().slice(0, 80) || 'Aurelia Piano';

  midi.header.name = safeTitle;
  midi.header.setTempo(bpm);
  midi.header.timeSignatures.push({ ticks: 0, timeSignature: [4, 4] });

  for (const spec of TRACK_SPECS) {
    const rawNotes = filteredNotes.filter((n) => n.track === spec.role);
    if (rawNotes.length === 0) continue;

    const track = midi.addTrack();
    track.name = spec.name;
    track.channel = spec.channel;
    track.instrument.number = spec.gmProgram;
    track.addCC({ number: 7, time: 0, value: spec.volume });

    const cleanNotes = sanitizeTrackNotes(rawNotes);
    for (const n of cleanNotes) {
      track.addNote({
        midi: n.midi,
        time: n.time,
        duration: n.duration,
        velocity: n.velocity,
      });
    }
  }

  return midi.toArray();
}

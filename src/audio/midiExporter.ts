import { PianoNoteEvent, TrackRole, TranscribedPiece } from './pitchDsp';

function writeVarLen(value: number, out: number[]): void {
  let v = Math.max(0, Math.round(value));
  const bytes: number[] = [v & 0x7f];
  v >>= 7;
  while (v > 0) {
    bytes.push((v & 0x7f) | 0x80);
    v >>= 7;
  }
  for (let i = bytes.length - 1; i >= 0; i--) {
    out.push(bytes[i]);
  }
}

function pushUint32BE(out: number[], val: number): void {
  out.push((val >>> 24) & 0xff, (val >>> 16) & 0xff, (val >>> 8) & 0xff, val & 0xff);
}

function pushUint16BE(out: number[], val: number): void {
  out.push((val >>> 8) & 0xff, val & 0xff);
}

function pushAscii(out: number[], str: string): void {
  for (let i = 0; i < str.length; i++) {
    out.push(str.charCodeAt(i) & 0x7f);
  }
}

const TRACK_SPECS: Array<{
  role: TrackRole;
  name: string;
  channel: number;
  gmProgram: number;
  volume: number;
}> = [
  { role: 'vocal', name: 'Melodia Vocal', channel: 0, gmProgram: 0, volume: 118 }, // Concert Grand Lead
  { role: 'chords', name: 'Acordes Piano', channel: 1, gmProgram: 0, volume: 96 }, // Grand Piano Harmony
  { role: 'bass', name: 'Bajo', channel: 2, gmProgram: 32, volume: 102 }, // Acoustic Bass
  { role: 'lead', name: 'Arreglos Guitarra', channel: 3, gmProgram: 24, volume: 88 }, // Acoustic Guitar
];

/**
 * Removes overlapping same-MIDI notes within a single instrument track so every DAW
 * and MIDI player (Synthesia, MuseScore, FL Studio, Logic, Ableton) plays back cleanly
 * with zero stuck or truncated notes.
 */
function sanitizeTrackNotes(
  notes: PianoNoteEvent[],
  secToTicks: (sec: number) => number,
): Array<{ midi: number; startTick: number; endTick: number; vel: number }> {
  const mapped = notes
    .map((n) => {
      const startTick = secToTicks(n.startTime);
      const endTick = Math.max(startTick + 32, secToTicks(n.startTime + n.duration));
      return {
        midi: Math.max(21, Math.min(108, Math.round(n.midi))),
        startTick,
        endTick,
        vel: Math.max(24, Math.min(127, Math.round(n.velocity * 120))),
      };
    })
    .sort((a, b) => a.startTick - b.startTick || a.midi - b.midi);

  const lastByPitch = new Map<number, { midi: number; startTick: number; endTick: number; vel: number }>();
  const clean: Array<{ midi: number; startTick: number; endTick: number; vel: number }> = [];

  for (const item of mapped) {
    const prev = lastByPitch.get(item.midi);
    if (prev && prev.endTick > item.startTick) {
      // Trim previous note cleanly 4 ticks before re-strike (or skip duplicate at identical tick)
      if (item.startTick - prev.startTick < 16) {
        prev.endTick = Math.max(prev.endTick, item.endTick);
        prev.vel = Math.max(prev.vel, item.vel);
        continue;
      }
      prev.endTick = Math.max(prev.startTick + 12, item.startTick - 4);
    }
    clean.push(item);
    lastByPitch.set(item.midi, item);
  }

  return clean;
}

/**
 * Encodes a cleaned multi-instrument transcription into a valid Standard MIDI File
 * (Format 1) with separate named tracks, channels, volume mix, and tempo map.
 */
export function buildMultiTrackMidi(
  piece: TranscribedPiece,
  filteredNotes: PianoNoteEvent[],
): Uint8Array {
  const ticksPerBeat = 480;
  const bpm = Math.max(40, Math.min(240, Math.round(piece.bpm || 100)));
  const usPerBeat = Math.round(60_000_000 / bpm);
  const secToTicks = (sec: number) => Math.max(0, Math.round((sec * bpm * ticksPerBeat) / 60));

  const chunks: number[][] = [];

  // Track 0: Title, Time Signature (4/4) & Tempo Meta Track
  const metaBody: number[] = [];
  const safeTitle =
    piece.title.replace(/[^\x20-\x7e]/g, '').trim().slice(0, 80) || 'Aurelia Piano';
  writeVarLen(0, metaBody);
  metaBody.push(0xff, 0x03);
  writeVarLen(safeTitle.length, metaBody);
  pushAscii(metaBody, safeTitle);

  // Time Signature: 4/4, 24 clocks/click, 8 32nd-notes per quarter
  writeVarLen(0, metaBody);
  metaBody.push(0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08);

  // Set Tempo
  writeVarLen(0, metaBody);
  metaBody.push(
    0xff,
    0x51,
    0x03,
    (usPerBeat >>> 16) & 0xff,
    (usPerBeat >>> 8) & 0xff,
    usPerBeat & 0xff,
  );

  // End of Track
  writeVarLen(0, metaBody);
  metaBody.push(0xff, 0x2f, 0x00);
  chunks.push(metaBody);

  // Separated Instrument Tracks
  for (const spec of TRACK_SPECS) {
    const rawNotes = filteredNotes.filter((n) => n.track === spec.role);
    if (rawNotes.length === 0) continue;

    const cleanNotes = sanitizeTrackNotes(rawNotes, secToTicks);
    interface MidiEvt {
      tick: number;
      order: number;
      bytes: number[];
    }
    const events: MidiEvt[] = [];

    for (const n of cleanNotes) {
      events.push({ tick: n.startTick, order: 1, bytes: [0x90 | spec.channel, n.midi, n.vel] });
      events.push({ tick: n.endTick, order: 0, bytes: [0x80 | spec.channel, n.midi, 0] });
    }

    events.sort((a, b) => a.tick - b.tick || a.order - b.order);

    const body: number[] = [];
    // Track Name
    writeVarLen(0, body);
    body.push(0xff, 0x03);
    writeVarLen(spec.name.length, body);
    pushAscii(body, spec.name);
    // Program Change
    writeVarLen(0, body);
    body.push(0xc0 | spec.channel, spec.gmProgram);
    // Channel Volume (CC7)
    writeVarLen(0, body);
    body.push(0xb0 | spec.channel, 0x07, spec.volume);

    let prevTick = 0;
    for (const ev of events) {
      const delta = Math.max(0, ev.tick - prevTick);
      prevTick = ev.tick;
      writeVarLen(delta, body);
      for (const b of ev.bytes) body.push(b);
    }

    // End of Track
    writeVarLen(0, body);
    body.push(0xff, 0x2f, 0x00);
    chunks.push(body);
  }

  // Assemble SMF Format 1 Header + MTrk chunks
  const out: number[] = [];
  pushAscii(out, 'MThd');
  pushUint32BE(out, 6);
  pushUint16BE(out, 1); // Format 1 (multi-track)
  pushUint16BE(out, chunks.length);
  pushUint16BE(out, ticksPerBeat);

  for (const body of chunks) {
    pushAscii(out, 'MTrk');
    pushUint32BE(out, body.length);
    for (const b of body) out.push(b);
  }

  return new Uint8Array(out);
}

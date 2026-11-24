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
}> = [
  { role: 'vocal', name: 'Melodia Vocal', channel: 0, gmProgram: 0 }, // Acoustic Grand Piano (Lead)
  { role: 'chords', name: 'Acordes Piano', channel: 1, gmProgram: 0 }, // Acoustic Grand Piano
  { role: 'bass', name: 'Bajo', channel: 2, gmProgram: 32 }, // Acoustic Bass
  { role: 'lead', name: 'Arreglos Guitarra', channel: 3, gmProgram: 24 }, // Nylon Acoustic Guitar
];

/**
 * Encodes a cleaned multi-instrument transcription into a valid Standard MIDI File
 * (Format 1) with separate named tracks and MIDI channels for Voz, Acordes, Bajo & Arreglos.
 */
export function buildMultiTrackMidi(
  piece: TranscribedPiece,
  filteredNotes: PianoNoteEvent[],
): Uint8Array {
  const ticksPerBeat = 480;
  const bpm = Math.max(40, Math.min(240, piece.bpm || 100));
  const usPerBeat = Math.round(60_000_000 / bpm);
  const secToTicks = (sec: number) => Math.max(0, Math.round((sec * bpm * ticksPerBeat) / 60));

  const chunks: number[][] = [];

  // Track 0: Tempo & Title Meta Track
  const metaBody: number[] = [];
  // Track name
  const safeTitle = piece.title.replace(/[^\x20-\x7e]/g, '') || 'Aurelia Piano';
  writeVarLen(0, metaBody);
  metaBody.push(0xff, 0x03, safeTitle.length);
  pushAscii(metaBody, safeTitle);
  // Set tempo
  writeVarLen(0, metaBody);
  metaBody.push(0xff, 0x51, 0x03, (usPerBeat >>> 16) & 0xff, (usPerBeat >>> 8) & 0xff, usPerBeat & 0xff);
  // End of track
  writeVarLen(0, metaBody);
  metaBody.push(0xff, 0x2f, 0x00);
  chunks.push(metaBody);

  // Separated instrument tracks
  for (const spec of TRACK_SPECS) {
    const trackNotes = filteredNotes.filter((n) => n.track === spec.role);
    if (trackNotes.length === 0) continue;

    interface MidiEvt {
      tick: number;
      order: number;
      bytes: number[];
    }
    const events: MidiEvt[] = [];

    for (const n of trackNotes) {
      const startTick = secToTicks(n.startTime);
      const endTick = Math.max(startTick + 30, secToTicks(n.startTime + n.duration));
      const vel = Math.max(24, Math.min(127, Math.round(n.velocity * 120)));
      const note = Math.max(21, Math.min(108, Math.round(n.midi)));

      events.push({ tick: startTick, order: 1, bytes: [0x90 | spec.channel, note, vel] });
      events.push({ tick: endTick, order: 0, bytes: [0x80 | spec.channel, note, 0] });
    }

    events.sort((a, b) => a.tick - b.tick || a.order - b.order);

    const body: number[] = [];
    // Track Name Meta Event
    writeVarLen(0, body);
    body.push(0xff, 0x03, spec.name.length);
    pushAscii(body, spec.name);
    // Program Change
    writeVarLen(0, body);
    body.push(0xc0 | spec.channel, spec.gmProgram);

    let prevTick = 0;
    for (const ev of events) {
      const delta = Math.max(0, ev.tick - prevTick);
      prevTick = ev.tick;
      writeVarLen(delta, body);
      for (const b of ev.bytes) body.push(b);
    }

    // End of track
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

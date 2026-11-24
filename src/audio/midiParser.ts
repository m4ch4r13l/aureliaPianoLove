import { MAX_MIDI, MIN_MIDI } from './pianoTheory';
import {
  estimateKeyFromChroma,
  isPitchInKey,
  PianoNoteEvent,
  TrackRole,
  TranscribedPiece,
} from './pitchDsp';

interface RawMidiNote {
  trackIndex: number;
  trackName: string;
  channel: number;
  gmProgram: number;
  midi: number;
  startTick: number;
  endTick: number;
  velocity: number;
}

interface TempoEvent {
  tick: number;
  usPerBeat: number;
}

const KEY_SIG_MAJOR_ROOTS: Record<number, number> = {
  [-7]: 11, // Cb / B
  [-6]: 6, // Gb / F#
  [-5]: 1, // Db / C#
  [-4]: 8, // Ab
  [-3]: 3, // Eb
  [-2]: 10, // Bb
  [-1]: 5, // F
  0: 0, // C
  1: 7, // G
  2: 2, // D
  3: 9, // A
  4: 4, // E
  5: 11, // B
  6: 6, // F#
  7: 1, // C#
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

function readUint16BE(data: Uint8Array, offset: number): number {
  return (data[offset] << 8) | data[offset + 1];
}

function readUint32BE(data: Uint8Array, offset: number): number {
  return (
    ((data[offset] << 24) |
      (data[offset + 1] << 16) |
      (data[offset + 2] << 8) |
      data[offset + 3]) >>>
    0
  );
}

function readAscii(data: Uint8Array, offset: number, length: number): string {
  let out = '';
  const end = Math.min(data.length, offset + length);
  for (let i = offset; i < end; i++) {
    out += String.fromCharCode(data[i]);
  }
  return out;
}

function readVarLen(
  data: Uint8Array,
  offset: number,
  limit: number,
): { value: number; next: number } {
  let value = 0;
  let pos = offset;
  for (let i = 0; i < 4 && pos < limit; i++) {
    const byte = data[pos++];
    value = (value << 7) | (byte & 0x7f);
    if ((byte & 0x80) === 0) break;
  }
  return { value, next: pos };
}

/**
 * Checks if a File or binary buffer is a Standard MIDI File (.mid / .midi or MThd magic header).
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

export function hasMidiHeader(bytes: Uint8Array): boolean {
  if (bytes.length < 14) return false;
  if (readAscii(bytes, 0, 4) === 'MThd') return true;
  // Support RIFF RMID wrapper (MThd appears within first 64 bytes)
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

function findHeaderOffset(bytes: Uint8Array): number {
  if (bytes.length >= 14 && readAscii(bytes, 0, 4) === 'MThd') return 0;
  const scanLimit = Math.min(bytes.length - 14, 256);
  for (let i = 0; i < scanLimit; i++) {
    if (
      bytes[i] === 0x4d &&
      bytes[i + 1] === 0x54 &&
      bytes[i + 2] === 0x68 &&
      bytes[i + 3] === 0x64
    ) {
      return i;
    }
  }
  return -1;
}

function isDrumTrackName(name: string): boolean {
  return /\b(drum|drums|perc|percussion|bateria|batería|kit|hihat|snare|kick)\b/i.test(name);
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

  if (isSingleMelodicTrack) {
    // Let pitch-aware hand/melody splitting classify single-track piano files nicely
    return null;
  }

  // General MIDI program heuristics for multi-track files
  if (gmProgram >= 32 && gmProgram <= 39) return 'bass';
  if ((gmProgram >= 52 && gmProgram <= 54) || (gmProgram >= 64 && gmProgram <= 87)) return 'vocal';
  if ((gmProgram >= 24 && gmProgram <= 31) || (gmProgram >= 40 && gmProgram <= 51)) return 'lead';
  if (medianMidi < 50) return 'bass';
  if (medianMidi >= 70) return 'vocal';
  return 'chords';
}

function clampTo88Keys(midi: number): number {
  let m = Math.round(midi);
  while (m < MIN_MIDI) m += 12;
  while (m > MAX_MIDI) m -= 12;
  return m;
}

/**
 * Pure Standard MIDI File (SMF Format 0, 1, and 2) parser.
 * Supports tempo maps, running status, Note On vel=0 as Note Off, sustain pedal (CC64),
 * percussion filtering (GM ch 10), and automatic 4-track instrument classification.
 */
export function parseMidiToPiece(
  bytes: Uint8Array,
  fallbackTitle = 'Archivo MIDI',
): TranscribedPiece {
  const headerStart = findHeaderOffset(bytes);
  if (headerStart < 0 || headerStart + 14 > bytes.length) {
    throw new Error('El archivo no contiene una cabecera MIDI (MThd) válida.');
  }

  const headerLen = readUint32BE(bytes, headerStart + 4);
  if (headerLen < 6 || headerStart + 8 + headerLen > bytes.length) {
    throw new Error('La cabecera del archivo MIDI está incompleta o dañada.');
  }

  const numTracks = readUint16BE(bytes, headerStart + 10);
  const divisionRaw = readUint16BE(bytes, headerStart + 12);

  const isSmpte = (divisionRaw & 0x8000) !== 0;
  const ticksPerBeat = isSmpte ? 480 : Math.max(1, divisionRaw);
  const smpteFps = isSmpte ? Math.max(24, 256 - ((divisionRaw >> 8) & 0xff)) : 0;
  const smpteTicksPerFrame = isSmpte ? Math.max(1, divisionRaw & 0xff) : 0;

  let cursor = headerStart + 8 + headerLen;
  const tempoEvents: TempoEvent[] = [{ tick: 0, usPerBeat: 500_000 }];
  const rawNotes: RawMidiNote[] = [];
  let embeddedTitle = '';
  let explicitKeySig: { root: number; mode: 'major' | 'minor'; label: string } | null = null;

  let parsedTracks = 0;
  while (cursor + 8 <= bytes.length && parsedTracks < Math.max(numTracks, 128)) {
    const chunkId = readAscii(bytes, cursor, 4);
    const chunkLen = readUint32BE(bytes, cursor + 4);
    const chunkStart = cursor + 8;
    const chunkEnd = Math.min(bytes.length, chunkStart + chunkLen);
    cursor = chunkEnd;

    if (chunkId !== 'MTrk') continue;

    const trackIndex = parsedTracks++;
    let pos = chunkStart;
    let absTick = 0;
    let runningStatus = 0;
    let trackName = '';

    const programByChannel = new Int16Array(16).fill(0);
    const sustainDown = new Uint8Array(16);
    const activeNotes = new Map<
      number,
      Array<{ startTick: number; velocity: number; gmProgram: number; releasedByKey: boolean }>
    >();

    const finishNote = (ch: number, note: number, endTick: number, force = false) => {
      const key = (ch << 7) | note;
      const stack = activeNotes.get(key);
      if (!stack || stack.length === 0) return;

      if (!force && sustainDown[ch] === 1) {
        // Mark as key-released; sustain pedal holds note until CC64 releases
        for (const entry of stack) {
          entry.releasedByKey = true;
        }
        return;
      }

      const item = stack.shift()!;
      if (stack.length === 0) activeNotes.delete(key);

      if (ch === 9 || isDrumTrackName(trackName)) return;
      if (endTick <= item.startTick) return;

      rawNotes.push({
        trackIndex,
        trackName,
        channel: ch,
        gmProgram: item.gmProgram,
        midi: clampTo88Keys(note),
        startTick: item.startTick,
        endTick,
        velocity: item.velocity,
      });
    };

    const releaseSustainedForChannel = (ch: number, tick: number) => {
      for (const [key, stack] of activeNotes.entries()) {
        if ((key >> 7) !== ch) continue;
        const note = key & 0x7f;
        let i = 0;
        while (i < stack.length) {
          if (stack[i].releasedByKey) {
            const item = stack.splice(i, 1)[0];
            if (ch !== 9 && !isDrumTrackName(trackName) && tick > item.startTick) {
              rawNotes.push({
                trackIndex,
                trackName,
                channel: ch,
                gmProgram: item.gmProgram,
                midi: clampTo88Keys(note),
                startTick: item.startTick,
                endTick: tick,
                velocity: item.velocity,
              });
            }
          } else {
            i++;
          }
        }
        if (stack.length === 0) activeNotes.delete(key);
      }
    };

    while (pos < chunkEnd) {
      const delta = readVarLen(bytes, pos, chunkEnd);
      absTick += delta.value;
      pos = delta.next;
      if (pos >= chunkEnd) break;

      let status = bytes[pos];
      if (status < 0x80) {
        if (runningStatus === 0) break;
        status = runningStatus;
      } else {
        pos++;
        if (status < 0xf0) {
          runningStatus = status;
        }
      }

      if (status === 0xff) {
        // Meta event
        if (pos >= chunkEnd) break;
        const metaType = bytes[pos++];
        const metaLenRes = readVarLen(bytes, pos, chunkEnd);
        const metaStart = metaLenRes.next;
        const metaEnd = Math.min(chunkEnd, metaStart + metaLenRes.value);
        pos = metaEnd;

        if (metaType === 0x2f) {
          break; // End of track
        } else if (metaType === 0x03 && metaLenRes.value > 0) {
          const text = readAscii(bytes, metaStart, metaLenRes.value).replace(/\0/g, '').trim();
          if (text) {
            if (!trackName) trackName = text;
            if (trackIndex === 0 && !embeddedTitle) embeddedTitle = text;
          }
        } else if (metaType === 0x51 && metaLenRes.value === 3 && metaStart + 3 <= chunkEnd) {
          const us =
            (bytes[metaStart] << 16) | (bytes[metaStart + 1] << 8) | bytes[metaStart + 2];
          if (us > 10_000 && us < 5_000_000) {
            tempoEvents.push({ tick: absTick, usPerBeat: us });
          }
        } else if (metaType === 0x59 && metaLenRes.value === 2 && metaStart + 2 <= chunkEnd && !explicitKeySig) {
          const sfRaw = bytes[metaStart];
          const sf = sfRaw > 127 ? sfRaw - 256 : sfRaw;
          const mi = bytes[metaStart + 1];
          const majRoot = KEY_SIG_MAJOR_ROOTS[sf];
          if (majRoot !== undefined) {
            const mode: 'major' | 'minor' = mi === 1 ? 'minor' : 'major';
            const root = mode === 'minor' ? (majRoot + 9) % 12 : majRoot;
            explicitKeySig = {
              root,
              mode,
              label: `${NOTE_NAMES_ES[root]} ${mode === 'major' ? 'mayor' : 'menor'}`,
            };
          }
        }
        continue;
      }

      if (status === 0xf0 || status === 0xf7) {
        // SysEx event
        const sysLen = readVarLen(bytes, pos, chunkEnd);
        pos = Math.min(chunkEnd, sysLen.next + sysLen.value);
        continue;
      }

      const msgType = status & 0xf0;
      const ch = status & 0x0f;

      if (msgType === 0x80) {
        if (pos + 2 > chunkEnd) break;
        const note = bytes[pos] & 0x7f;
        pos += 2;
        finishNote(ch, note, absTick, false);
      } else if (msgType === 0x90) {
        if (pos + 2 > chunkEnd) break;
        const note = bytes[pos] & 0x7f;
        const vel = bytes[pos + 1] & 0x7f;
        pos += 2;

        if (vel === 0) {
          finishNote(ch, note, absTick, false);
        } else {
          // Close any already-sounding note on same channel & pitch cleanly
          finishNote(ch, note, absTick, true);
          const key = (ch << 7) | note;
          const stack = activeNotes.get(key) ?? [];
          stack.push({
            startTick: absTick,
            velocity: Math.max(0.2, Math.min(1, vel / 127)),
            gmProgram: programByChannel[ch],
            releasedByKey: false,
          });
          activeNotes.set(key, stack);
        }
      } else if (msgType === 0xa0 || msgType === 0xe0) {
        pos += 2;
      } else if (msgType === 0xb0) {
        if (pos + 2 > chunkEnd) break;
        const cc = bytes[pos] & 0x7f;
        const val = bytes[pos + 1] & 0x7f;
        pos += 2;
        if (cc === 64) {
          const nextDown = val >= 64 ? 1 : 0;
          if (sustainDown[ch] === 1 && nextDown === 0) {
            releaseSustainedForChannel(ch, absTick);
          }
          sustainDown[ch] = nextDown;
        } else if (cc === 120 || cc === 123) {
          // All Sound Off / All Notes Off
          sustainDown[ch] = 0;
          for (const key of Array.from(activeNotes.keys())) {
            if ((key >> 7) === ch) finishNote(ch, key & 0x7f, absTick, true);
          }
        }
      } else if (msgType === 0xc0) {
        if (pos + 1 > chunkEnd) break;
        programByChannel[ch] = bytes[pos] & 0x7f;
        pos += 1;
      } else if (msgType === 0xd0) {
        pos += 1;
      } else {
        break;
      }
    }

    // Flush any unterminated notes at track end
    for (const [key, stack] of activeNotes.entries()) {
      const ch = key >> 7;
      const note = key & 0x7f;
      for (const item of stack) {
        if (ch !== 9 && !isDrumTrackName(trackName)) {
          rawNotes.push({
            trackIndex,
            trackName,
            channel: ch,
            gmProgram: item.gmProgram,
            midi: clampTo88Keys(note),
            startTick: item.startTick,
            endTick: Math.max(item.startTick + Math.round(ticksPerBeat * 0.25), absTick),
            velocity: item.velocity,
          });
        }
      }
    }
  }

  if (rawNotes.length === 0) {
    throw new Error('El archivo MIDI no contiene notas melódicas reproducibles.');
  }

  // Build deduplicated sorted tempo map
  tempoEvents.sort((a, b) => a.tick - b.tick);
  const tempoMap: Array<{ tick: number; timeSec: number; usPerBeat: number }> = [];
  for (const ev of tempoEvents) {
    if (tempoMap.length === 0) {
      tempoMap.push({ tick: 0, timeSec: 0, usPerBeat: ev.usPerBeat });
      continue;
    }
    const prev = tempoMap[tempoMap.length - 1];
    if (ev.tick === prev.tick) {
      prev.usPerBeat = ev.usPerBeat;
    } else {
      const dtTicks = ev.tick - prev.tick;
      const dtSec = (dtTicks / ticksPerBeat) * (prev.usPerBeat / 1_000_000);
      tempoMap.push({
        tick: ev.tick,
        timeSec: prev.timeSec + dtSec,
        usPerBeat: ev.usPerBeat,
      });
    }
  }

  const tickToSec = (tick: number): number => {
    if (isSmpte) {
      return tick / (smpteFps * smpteTicksPerFrame);
    }
    let lo = 0;
    let hi = tempoMap.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (tempoMap[mid].tick <= tick) lo = mid;
      else hi = mid - 1;
    }
    const seg = tempoMap[lo];
    return seg.timeSec + ((tick - seg.tick) / ticksPerBeat) * (seg.usPerBeat / 1_000_000);
  };

  // Group notes by (trackIndex, channel) to inspect multi-track structure
  const groups = new Map<string, RawMidiNote[]>();
  for (const n of rawNotes) {
    const key = `${n.trackIndex}:${n.channel}`;
    const list = groups.get(key);
    if (list) list.push(n);
    else groups.set(key, [n]);
  }

  const isSingleTrackPiano = groups.size === 1;
  const groupRoles = new Map<string, TrackRole | null>();
  for (const [key, list] of groups.entries()) {
    const sortedMidis = list.map((n) => n.midi).sort((a, b) => a - b);
    const medianMidi = sortedMidis[sortedMidis.length >> 1];
    const sample = list[0];
    groupRoles.set(
      key,
      classifyTrackRole(sample.trackName, sample.gmProgram, medianMidi, isSingleTrackPiano),
    );
  }

  // Convert ticks to seconds and normalize start offset if MIDI has long leading silence (> 1.5s)
  const timed = rawNotes.map((n) => {
    const startSec = tickToSec(n.startTick);
    const endSec = tickToSec(n.endTick);
    return {
      ...n,
      startTime: startSec,
      duration: Math.max(0.06, Math.min(6.5, endSec - startSec)),
    };
  });

  timed.sort((a, b) => a.startTime - b.startTime || a.midi - b.midi);
  const firstStart = timed[0].startTime;
  const trimOffset = firstStart > 1.5 ? firstStart - 0.25 : 0;

  // Estimate key signature if MIDI didn't carry a Key Signature meta event
  const chroma = new Float64Array(12);
  for (const n of timed) {
    chroma[n.midi % 12] += n.duration * n.velocity;
  }
  const keyInfo = explicitKeySig ?? estimateKeyFromChroma(chroma);

  // For unlabelled single-track piano files, identify top melody voice per onset window
  const topMidiByBucket = new Map<number, number>();
  if (isSingleTrackPiano) {
    for (const n of timed) {
      const bucket = Math.round((n.startTime - trimOffset) * 16); // ~62ms onset grouping
      const prevTop = topMidiByBucket.get(bucket) ?? -1;
      if (n.midi > prevTop) topMidiByBucket.set(bucket, n.midi);
    }
  }

  const countsByTrack: Record<TrackRole, number> = {
    vocal: 0,
    chords: 0,
    bass: 0,
    lead: 0,
  };

  let maxEndTime = 0;
  const notes: PianoNoteEvent[] = timed.map((n, idx) => {
    const startTime = Math.max(0, Number((n.startTime - trimOffset).toFixed(3)));
    const duration = Number(n.duration.toFixed(3));
    if (startTime + duration > maxEndTime) maxEndTime = startTime + duration;

    const groupKey = `${n.trackIndex}:${n.channel}`;
    let role = groupRoles.get(groupKey) ?? null;

    if (!role) {
      // Single-track concert piano heuristic: Bass (<50), Top Melody (>=64 & highest at onset), Chords (middle)
      const bucket = Math.round(startTime * 16);
      const isTopVoice = (topMidiByBucket.get(bucket) ?? -1) === n.midi;
      if (n.midi < 50) {
        role = 'bass';
      } else if (isTopVoice && n.midi >= 64) {
        role = 'vocal';
      } else {
        role = 'chords';
      }
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

  // Compute primary BPM from initial (or first active) tempo event
  const primaryUsPerBeat = tempoMap[0]?.usPerBeat ?? 500_000;
  const bpm = Math.max(40, Math.min(240, Math.round(60_000_000 / primaryUsPerBeat)));

  const cleanFallback = fallbackTitle.trim();
  const title =
    cleanFallback && cleanFallback !== 'Archivo MIDI'
      ? cleanFallback
      : embeddedTitle || 'Archivo MIDI';

  return {
    id: `midi-${Date.now()}`,
    title,
    subtitle: 'Partitura MIDI importada con fidelidad exacta',
    source: 'midi',
    duration: Number(Math.max(1, maxEndTime + 0.4).toFixed(2)),
    bpm,
    keySignature: keyInfo.label,
    keyRoot: keyInfo.root,
    keyMode: keyInfo.mode,
    notes,
    countsByTrack,
  };
}

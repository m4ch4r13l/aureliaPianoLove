export const MIN_MIDI = 21; // A0
export const MAX_MIDI = 108; // C8
export const TOTAL_KEYS = MAX_MIDI - MIN_MIDI + 1; // 88 keys

const SOLFEGE_SHARP = [
  'Do',
  'Do#',
  'Re',
  'Re#',
  'Mi',
  'Fa',
  'Fa#',
  'Sol',
  'Sol#',
  'La',
  'La#',
  'Si',
] as const;

const SCIENTIFIC_SHARP = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B',
] as const;

const BLACK_KEY_PCS = new Set([1, 3, 6, 8, 10]);

export function isBlackKey(midi: number): boolean {
  return BLACK_KEY_PCS.has(((midi % 12) + 12) % 12);
}

export function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function freqToMidiFloat(freq: number): number {
  return 69 + 12 * Math.log2(freq / 440);
}

export function midiToSolfege(midi: number): string {
  const pc = ((midi % 12) + 12) % 12;
  const octave = Math.floor(midi / 12) - 1;
  return `${SOLFEGE_SHARP[pc]}${octave}`;
}

export function midiToScientific(midi: number): string {
  const pc = ((midi % 12) + 12) % 12;
  const octave = Math.floor(midi / 12) - 1;
  return `${SCIENTIFIC_SHARP[pc]}${octave}`;
}

export function formatTime(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export interface PianoKeyLayout {
  midi: number;
  isBlack: boolean;
  /** Center X position in meters along the keyboard (centered at 0) */
  x: number;
  width: number;
  length: number;
  height: number;
  whiteIndex: number;
}

/**
 * Generates exact physical 88-key concert grand layout in meters.
 * Standard 52 white keys span ~1.225 m across X.
 */
export function build88KeyLayout(): {
  keys: PianoKeyLayout[];
  totalWidth: number;
  whiteKeyWidth: number;
  whiteKeyPitch: number;
} {
  const whiteKeyPitch = 0.0236; // 23.6 mm center-to-center
  const whiteKeyWidth = 0.0222; // 1.4 mm clean gap between ivory keys
  const whiteKeyLength = 0.148;
  const whiteKeyHeight = 0.022;

  const blackKeyWidth = 0.0132;
  const blackKeyLength = 0.094;
  const blackKeyHeight = 0.015;

  const whiteCount = 52;
  const totalWidth = whiteCount * whiteKeyPitch;
  const startX = -totalWidth / 2 + whiteKeyPitch / 2;

  // Fine acoustic offsets for black keys relative to the boundary between adjacent white keys
  const blackOffsetByPc: Record<number, number> = {
    1: -0.0014, // C# slightly left
    3: 0.0014,  // D# slightly right
    6: -0.0018, // F# slightly left
    8: 0.0,     // G# centered
    10: 0.0018, // A# slightly right
  };

  const keys: PianoKeyLayout[] = [];
  let whiteIdx = 0;

  for (let midi = MIN_MIDI; midi <= MAX_MIDI; midi++) {
    const black = isBlackKey(midi);
    if (!black) {
      keys.push({
        midi,
        isBlack: false,
        x: startX + whiteIdx * whiteKeyPitch,
        width: whiteKeyWidth,
        length: whiteKeyLength,
        height: whiteKeyHeight,
        whiteIndex: whiteIdx,
      });
      whiteIdx++;
    } else {
      const pc = midi % 12;
      // Boundary between previous white key (whiteIdx - 1) and next white key (whiteIdx)
      const boundaryX = startX + (whiteIdx - 0.5) * whiteKeyPitch;
      keys.push({
        midi,
        isBlack: true,
        x: boundaryX + (blackOffsetByPc[pc] ?? 0),
        width: blackKeyWidth,
        length: blackKeyLength,
        height: blackKeyHeight,
        whiteIndex: whiteIdx - 1,
      });
    }
  }

  return { keys, totalWidth, whiteKeyWidth, whiteKeyPitch };
}

/** Computer keyboard bindings for live playing around middle C (C4 = 60) */
export const KEYBOARD_KEY_TO_MIDI: Record<string, number> = {
  KeyZ: 48, // C3
  KeyS: 49,
  KeyX: 50,
  KeyD: 51,
  KeyC: 52,
  KeyV: 53,
  KeyG: 54,
  KeyB: 55,
  KeyH: 56,
  KeyN: 57,
  KeyJ: 58,
  KeyM: 59,
  KeyQ: 60, // C4
  Digit2: 61,
  KeyW: 62,
  Digit3: 63,
  KeyE: 64,
  KeyR: 65,
  Digit5: 66,
  KeyT: 67,
  Digit6: 68,
  KeyY: 69,
  Digit7: 70,
  KeyU: 71,
  KeyI: 72, // C5
  Digit9: 73,
  KeyO: 74,
  Digit0: 75,
  KeyP: 76,
};

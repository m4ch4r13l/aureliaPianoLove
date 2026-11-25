import { transcribeAudioBufferWithBasicPitch } from './basicPitchTranscriber';
import { transcribeAudioBufferWithByteDancePiano } from './bytedancePianoTranscriber';
import {
  transcribePcmToPianoAndVocalAsync,
  TranscribedPiece,
  TranscriptionMode,
} from './pitchDsp';

export interface TranscribedUploadResult {
  piece: TranscribedPiece;
  audioBuffer: AudioBuffer;
}

/**
 * Re-transcribes an already-decoded AudioBuffer in a chosen mode:
 *   - `'piano-hd'`: ByteDance High-Resolution Piano CRNN (96.7% MAESTRO F1, ONNX WebGPU/WASM)
 *   - `'piano'`:    Spotify Basic Pitch calibrated for fast solo piano
 *   - `'song'`:     HPSS drum-removal + Stereo Center-Vocal + Basic Pitch + Viterbi
 */
export async function retranscribeAudioBuffer(
  audioBuffer: AudioBuffer,
  title: string,
  mode: TranscriptionMode,
  onProgress?: (progress: number, stage: string) => void,
): Promise<TranscribedPiece> {
  if (mode === 'piano-hd') {
    try {
      const piece = await transcribeAudioBufferWithByteDancePiano(
        audioBuffer,
        title,
        onProgress,
      );
      if (piece.notes.length > 0) {
        return { ...piece, transcriptionMode: 'piano-hd' };
      }
    } catch (bdErr) {
      console.warn('ByteDance Piano CRNN unavailable, falling back to Basic Pitch:', bdErr);
      onProgress?.(0.15, 'Usando Basic Pitch (respaldo local)...');
    }
  }

  try {
    const piece = await transcribeAudioBufferWithBasicPitch(
      audioBuffer,
      title,
      onProgress,
      mode === 'piano-hd' ? 'piano' : mode,
    );
    if (piece.notes.length > 0) {
      return piece;
    }
  } catch (bpErr) {
    console.warn('Basic Pitch fallback to Constant-Q engine:', bpErr);
  }

  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : null;
  const fallback = await transcribePcmToPianoAndVocalAsync(
    left,
    right,
    audioBuffer.sampleRate,
    title,
    {
      maxDurationSec: 210,
      targetSampleRate: 22050,
      onProgress,
    },
  );
  return { ...fallback, transcriptionMode: mode };
}

/**
 * Decodes an uploaded MP3 / WAV / OGG / M4A File via Web Audio API and transcribes it
 * using ByteDance High-Resolution Piano CRNN (default `'piano-hd'`), Spotify Basic Pitch
 * (`'piano'`), or full-song HPSS + Stereo Vocal mode (`'song'`).
 */
export async function decodeAndTranscribeAudioFile(
  file: File,
  onProgress?: (progress: number, stage: string) => void,
  mode: TranscriptionMode = 'piano-hd',
): Promise<TranscribedUploadResult> {
  onProgress?.(0.02, 'Leyendo archivo de audio...');
  const arrayBuffer = await file.arrayBuffer();
  if (arrayBuffer.byteLength < 64) {
    throw new Error('El archivo de audio está vacío o dañado.');
  }

  const AudioCtx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AudioCtx();

  try {
    onProgress?.(0.04, 'Decodificando audio con Web Audio...');
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    if (audioBuffer.length < 1024 || audioBuffer.duration < 0.2) {
      throw new Error('El audio es demasiado corto para extraer notas MIDI.');
    }

    await new Promise((resolve) => setTimeout(resolve, 16));

    const cleanName =
      file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim() || 'Pista subida';

    const piece = await retranscribeAudioBuffer(audioBuffer, cleanName, mode, onProgress);
    return { piece, audioBuffer };
  } finally {
    await ctx.close().catch(() => {});
  }
}

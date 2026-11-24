import { transcribeAudioBufferWithBasicPitch } from './basicPitchTranscriber';
import { transcribePcmToPianoAndVocalAsync, TranscribedPiece } from './pitchDsp';

export interface TranscribedUploadResult {
  piece: TranscribedPiece;
  audioBuffer: AudioBuffer;
}

/**
 * Decodes an uploaded MP3 / WAV / OGG / M4A File via Web Audio API and transcribes it
 * using Spotify Basic Pitch (@spotify/basic-pitch over TensorFlow.js) with graceful
 * fallback to the local Constant-Q engine if neural inference is unavailable.
 */
export async function decodeAndTranscribeAudioFile(
  file: File,
  onProgress?: (progress: number, stage: string) => void,
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
    onProgress?.(0.06, 'Decodificando audio con Web Audio...');
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    if (audioBuffer.length < 1024 || audioBuffer.duration < 0.2) {
      throw new Error('El audio es demasiado corto para extraer notas MIDI.');
    }

    // Yield one frame so the progress overlay paints cleanly before model inference starts
    await new Promise((resolve) => setTimeout(resolve, 16));

    const cleanName =
      file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim() || 'Pista subida';

    try {
      const piece = await transcribeAudioBufferWithBasicPitch(audioBuffer, cleanName, onProgress);
      if (piece.notes.length > 0) {
        return { piece, audioBuffer };
      }
    } catch (bpErr) {
      console.warn('Basic Pitch fallback to Constant-Q engine:', bpErr);
    }

    // Resilient fallback if Basic Pitch returns 0 notes or model cannot load in environment
    const left = audioBuffer.getChannelData(0);
    const right = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : null;
    const piece = await transcribePcmToPianoAndVocalAsync(
      left,
      right,
      audioBuffer.sampleRate,
      cleanName,
      {
        maxDurationSec: 210,
        targetSampleRate: 22050,
        onProgress,
      },
    );

    return { piece, audioBuffer };
  } finally {
    await ctx.close().catch(() => {});
  }
}

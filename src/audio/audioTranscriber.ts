import { transcribePcmToPianoAndVocal, TranscribedPiece } from './pitchDsp';

export interface TranscribedUploadResult {
  piece: TranscribedPiece;
  audioBuffer: AudioBuffer;
}

/**
 * Decodes an uploaded MP3 / WAV / OGG / M4A File via Web Audio API and runs
 * the harmonic + vocal melody DSP transcription engine asynchronously.
 */
export async function decodeAndTranscribeAudioFile(
  file: File,
  onProgress?: (progress: number, stage: string) => void,
): Promise<TranscribedUploadResult> {
  onProgress?.(0.02, 'Leyendo archivo de audio...');
  const arrayBuffer = await file.arrayBuffer();

  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AudioCtx();

  try {
    onProgress?.(0.08, 'Decodificando MP3 con Web Audio...');
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));

    // Yield one frame so the progress overlay paints cleanly before DSP crunch
    await new Promise((resolve) => setTimeout(resolve, 24));

    const left = audioBuffer.getChannelData(0);
    const right = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : null;
    const cleanName = file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim() || 'Pista subida';

    const piece = transcribePcmToPianoAndVocal(left, right, audioBuffer.sampleRate, cleanName, {
      maxDurationSec: 210,
      targetSampleRate: 22050,
      fftSize: 4096,
      hopSize: 1024,
      onProgress,
    });

    return { piece, audioBuffer };
  } finally {
    await ctx.close().catch(() => {});
  }
}

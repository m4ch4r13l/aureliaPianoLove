import { MAX_MIDI, MIN_MIDI, midiToFreq } from './pianoTheory';
import { TrackRole } from './pitchDsp';

interface ActiveAudioVoice {
  stopAt: (when: number) => void;
}

/**
 * Procedural Concert Grand Piano Synthesizer using Web Audio API.
 * Uses stretched harmonic partials, felt-hammer transient, velocity-sensitive lowpass,
 * stereo key-position panning, and a procedural concert-hall convolution reverb.
 */
export class GrandPianoSynth {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private dryBus: GainNode | null = null;
  private wetBus: GainNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;

  private activeVoices = new Map<string, ActiveAudioVoice>();
  private originalSource: AudioBufferSourceNode | null = null;
  private originalGain: GainNode | null = null;

  private masterVolume = 0.85;
  private originalAudioMix = 0.0; // 0 = pure grand piano, >0 = blend with original MP3

  public ensureContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    if (!this.ctx) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return null;
      this.ctx = new AudioCtx();
      this.setupGraph(this.ctx);
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  private setupGraph(ctx: AudioContext): void {
    this.compressor = ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -14;
    this.compressor.knee.value = 18;
    this.compressor.ratio.value = 4;
    this.compressor.attack.value = 0.004;
    this.compressor.release.value = 0.22;

    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = this.masterVolume;

    this.dryBus = ctx.createGain();
    this.dryBus.gain.value = 0.82;

    this.wetBus = ctx.createGain();
    this.wetBus.gain.value = 0.34;

    // Procedural warm concert-hall impulse response
    const convolver = ctx.createConvolver();
    convolver.buffer = this.createHallImpulse(ctx, 2.2, 2.6);

    this.dryBus.connect(this.compressor);
    this.wetBus.connect(convolver);
    convolver.connect(this.compressor);

    this.compressor.connect(this.masterGain);
    this.masterGain.connect(ctx.destination);

    // Pre-generate short filtered hammer felt noise buffer
    const noiseLen = Math.floor(ctx.sampleRate * 0.045);
    this.noiseBuffer = ctx.createBuffer(1, noiseLen, ctx.sampleRate);
    const data = this.noiseBuffer.getChannelData(0);
    for (let i = 0; i < noiseLen; i++) {
      const env = Math.exp(-i / (ctx.sampleRate * 0.008));
      data[i] = (Math.random() * 2 - 1) * env;
    }
  }

  private createHallImpulse(ctx: AudioContext, duration: number, decay: number): AudioBuffer {
    const length = Math.floor(ctx.sampleRate * duration);
    const impulse = ctx.createBuffer(2, length, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const channel = impulse.getChannelData(ch);
      let lp = 0;
      for (let i = 0; i < length; i++) {
        const t = i / length;
        const env = Math.pow(1 - t, decay);
        const white = (Math.random() * 2 - 1) * env;
        // Smooth high-frequency absorption typical of wood concert halls
        lp = lp * 0.72 + white * 0.28;
        channel[i] = lp * (i < 400 ? i / 400 : 1);
      }
    }
    return impulse;
  }

  public setVolume(vol: number): void {
    this.masterVolume = Math.max(0, Math.min(1, vol));
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(this.masterVolume, this.ctx.currentTime, 0.03);
    }
  }

  public setOriginalAudioMix(mix: number): void {
    this.originalAudioMix = Math.max(0, Math.min(1, mix));
    if (this.originalGain && this.ctx) {
      this.originalGain.gain.setTargetAtTime(this.originalAudioMix * 0.65, this.ctx.currentTime, 0.04);
    }
  }

  public startOriginalAudio(buffer: AudioBuffer, offsetSec: number, playbackRate: number): void {
    this.stopOriginalAudio();
    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) return;

    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.playbackRate.value = playbackRate;

    const gain = ctx.createGain();
    gain.gain.value = this.originalAudioMix * 0.65;

    src.connect(gain);
    gain.connect(this.masterGain);

    const safeOffset = Math.max(0, Math.min(buffer.duration - 0.05, offsetSec));
    src.start(ctx.currentTime, safeOffset);

    this.originalSource = src;
    this.originalGain = gain;
  }

  public stopOriginalAudio(): void {
    if (this.originalSource) {
      try {
        this.originalSource.stop();
        this.originalSource.disconnect();
      } catch {
        // already stopped
      }
      this.originalSource = null;
    }
    if (this.originalGain) {
      try {
        this.originalGain.disconnect();
      } catch {
        // ignore
      }
      this.originalGain = null;
    }
  }

  public playNote(
    midi: number,
    velocity = 0.75,
    durationSec = 1.2,
    role: TrackRole = 'piano',
  ): void {
    const ctx = this.ensureContext();
    if (!ctx || !this.dryBus || !this.wetBus) return;

    const clampedMidi = Math.max(MIN_MIDI, Math.min(MAX_MIDI, Math.round(midi)));
    const freq = midiToFreq(clampedMidi);
    const now = ctx.currentTime;
    const vel = Math.max(0.12, Math.min(1.0, velocity));

    // Voice envelope and harmonic richness vary naturally across the 88-key range
    const pitchNorm = (clampedMidi - MIN_MIDI) / (MAX_MIDI - MIN_MIDI); // 0 bass -> 1 treble
    const naturalDecay = (4.2 - pitchNorm * 3.1) * (role === 'vocal' ? 1.15 : 1.0);
    const holdDur = Math.max(0.09, Math.min(durationSec, 4.5));
    const releaseTime = 0.28 + (1 - pitchNorm) * 0.35;
    const stopTime = now + holdDur + releaseTime;

    // Velocity-sensitive lowpass brightness filter
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    const peakCutoff = Math.min(
      11500,
      freq * (role === 'vocal' ? 6.2 : 4.6) * (0.45 + vel * 0.85),
    );
    const sustainCutoff = Math.min(4800, freq * (role === 'vocal' ? 2.8 : 2.1));
    filter.frequency.setValueAtTime(peakCutoff, now);
    filter.frequency.exponentialRampToValueAtTime(
      Math.max(180, sustainCutoff),
      now + Math.min(holdDur, 0.9),
    );
    filter.Q.value = role === 'vocal' ? 1.15 : 0.7;

    // Stereo position mirroring physical grand piano keyboard (bass left -> treble right)
    const panner = ctx.createStereoPanner();
    panner.pan.value = (pitchNorm - 0.5) * 0.72;

    const voiceGain = ctx.createGain();
    const peakGain = vel * (role === 'vocal' ? 0.24 : 0.17) * (1.12 - pitchNorm * 0.28);
    voiceGain.gain.setValueAtTime(0.0001, now);
    voiceGain.gain.linearRampToValueAtTime(peakGain, now + 0.006);
    voiceGain.gain.exponentialRampToValueAtTime(
      Math.max(0.0005, peakGain * 0.42),
      now + Math.min(holdDur, naturalDecay * 0.5),
    );
    voiceGain.gain.setTargetAtTime(0.0001, now + holdDur, releaseTime * 0.32);

    filter.connect(voiceGain);
    voiceGain.connect(panner);
    panner.connect(this.dryBus);
    panner.connect(this.wetBus);

    // Stretched piano string partials (slight inharmonicity coefficient B)
    const B = 0.00014 + pitchNorm * 0.00035;
    const partials: Array<{ n: number; amp: number; type: OscillatorType }> =
      role === 'vocal'
        ? [
            { n: 1, amp: 1.0, type: 'triangle' },
            { n: 2, amp: 0.48, type: 'sine' },
            { n: 3, amp: 0.22, type: 'sine' },
            { n: 4, amp: 0.09, type: 'sine' },
          ]
        : [
            { n: 1, amp: 1.0, type: 'triangle' },
            { n: 2, amp: 0.36, type: 'sine' },
            { n: 3, amp: 0.16, type: 'sine' },
            { n: 4, amp: 0.07, type: 'sine' },
          ];

    const oscs: OscillatorNode[] = [];
    for (const p of partials) {
      const partialFreq = freq * p.n * Math.sqrt(1 + B * p.n * p.n);
      if (partialFreq > ctx.sampleRate * 0.45) continue;

      const osc = ctx.createOscillator();
      osc.type = p.type;
      osc.frequency.setValueAtTime(partialFreq, now);
      // Tiny chorus detune on 2nd partial for warm unison grand piano strings
      if (p.n === 2) osc.detune.value = 1.6;

      const pGain = ctx.createGain();
      pGain.gain.value = p.amp;
      osc.connect(pGain);
      pGain.connect(filter);
      osc.start(now);
      osc.stop(stopTime + 0.08);
      oscs.push(osc);
    }

    // Subtle felt-hammer percussive thump
    if (this.noiseBuffer) {
      const noise = ctx.createBufferSource();
      noise.buffer = this.noiseBuffer;
      const nFilter = ctx.createBiquadFilter();
      nFilter.type = 'bandpass';
      nFilter.frequency.value = Math.min(3400, 650 + pitchNorm * 2200);
      nFilter.Q.value = 1.4;
      const nGain = ctx.createGain();
      nGain.gain.value = vel * 0.09;
      noise.connect(nFilter);
      nFilter.connect(nGain);
      nGain.connect(panner);
      noise.start(now);
    }

    const voiceId = `${clampedMidi}-${now.toFixed(4)}-${Math.random()}`;
    const cleanupTimer = window.setTimeout(() => {
      this.activeVoices.delete(voiceId);
    }, Math.ceil((stopTime - now + 0.15) * 1000));

    this.activeVoices.set(voiceId, {
      stopAt: (when: number) => {
        window.clearTimeout(cleanupTimer);
        try {
          voiceGain.gain.cancelScheduledValues(when);
          voiceGain.gain.setTargetAtTime(0.0001, when, 0.04);
          for (const o of oscs) {
            o.stop(when + 0.15);
          }
        } catch {
          // ignore already stopped
        }
      },
    });
  }

  public stopAll(): void {
    this.stopOriginalAudio();
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    for (const v of this.activeVoices.values()) {
      v.stopAt(now);
    }
    this.activeVoices.clear();
  }

  public dispose(): void {
    this.stopAll();
    if (this.ctx) {
      this.ctx.close().catch(() => {});
      this.ctx = null;
    }
  }
}

import { MAX_MIDI, MIN_MIDI, midiToFreq } from './pianoTheory';
import { TrackRole } from './pitchDsp';

interface ActiveAudioVoice {
  startedAt: number;
  stopAt: (when: number) => void;
}

const MAX_POLYPHONY = 14;

/**
 * Procedural Concert Grand Piano Synthesizer using Web Audio API.
 * Features stretched partials, felt-hammer attack, per-role musical voicing
 * (vocal cantabile, warm chords, clean bass, crystalline lead), voice-stealing
 * anti-clipping protection, and procedural concert-hall reverb.
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
  private originalAudioMix = 0.0;

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
    // Sub-rumble clean high-pass to prevent low-end distortion
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 34;

    this.compressor = ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -16;
    this.compressor.knee.value = 20;
    this.compressor.ratio.value = 5;
    this.compressor.attack.value = 0.003;
    this.compressor.release.value = 0.2;

    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = this.masterVolume;

    this.dryBus = ctx.createGain();
    this.dryBus.gain.value = 0.84;

    this.wetBus = ctx.createGain();
    this.wetBus.gain.value = 0.28;

    const convolver = ctx.createConvolver();
    convolver.buffer = this.createHallImpulse(ctx, 2.0, 2.8);

    this.dryBus.connect(hp);
    this.wetBus.connect(convolver);
    convolver.connect(hp);

    hp.connect(this.compressor);
    this.compressor.connect(this.masterGain);
    this.masterGain.connect(ctx.destination);

    const noiseLen = Math.floor(ctx.sampleRate * 0.035);
    this.noiseBuffer = ctx.createBuffer(1, noiseLen, ctx.sampleRate);
    const data = this.noiseBuffer.getChannelData(0);
    for (let i = 0; i < noiseLen; i++) {
      const env = Math.exp(-i / (ctx.sampleRate * 0.006));
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
        lp = lp * 0.76 + white * 0.24;
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

  private enforcePolyphonyLimit(now: number): void {
    while (this.activeVoices.size >= MAX_POLYPHONY) {
      let oldestKey: string | null = null;
      let oldestTime = Infinity;
      for (const [k, v] of this.activeVoices.entries()) {
        if (v.startedAt < oldestTime) {
          oldestTime = v.startedAt;
          oldestKey = k;
        }
      }
      if (!oldestKey) break;
      const victim = this.activeVoices.get(oldestKey);
      this.activeVoices.delete(oldestKey);
      victim?.stopAt(now);
    }
  }

  public playNote(
    midi: number,
    velocity = 0.75,
    durationSec = 1.2,
    role: TrackRole = 'chords',
  ): void {
    const ctx = this.ensureContext();
    if (!ctx || !this.dryBus || !this.wetBus) return;

    const now = ctx.currentTime;
    this.enforcePolyphonyLimit(now);

    const clampedMidi = Math.max(MIN_MIDI, Math.min(MAX_MIDI, Math.round(midi)));
    const freq = midiToFreq(clampedMidi);
    const vel = Math.max(0.12, Math.min(1.0, velocity));

    const pitchNorm = (clampedMidi - MIN_MIDI) / (MAX_MIDI - MIN_MIDI);
    const naturalDecay = (3.8 - pitchNorm * 2.8) * (role === 'vocal' ? 1.15 : role === 'bass' ? 1.2 : 0.95);
    const holdDur = Math.max(0.09, Math.min(durationSec, 3.8));
    const releaseTime = 0.24 + (1 - pitchNorm) * 0.28;
    const stopTime = now + holdDur + releaseTime;

    // Role-tailored concert grand voicing (clear separation between singing voice, bass & chords)
    const roleBright =
      role === 'vocal' ? 5.6 : role === 'lead' ? 5.0 : role === 'bass' ? 2.6 : 3.8;
    const roleGainScale =
      role === 'vocal' ? 0.22 : role === 'bass' ? 0.17 : role === 'lead' ? 0.12 : 0.13;

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    const peakCutoff = Math.min(9800, freq * roleBright * (0.48 + vel * 0.78));
    const sustainCutoff = Math.min(4200, freq * (role === 'vocal' ? 2.6 : 1.85));
    filter.frequency.setValueAtTime(peakCutoff, now);
    filter.frequency.exponentialRampToValueAtTime(
      Math.max(150, sustainCutoff),
      now + Math.min(holdDur, 0.85),
    );
    filter.Q.value = role === 'vocal' ? 1.05 : 0.6;

    const panner = ctx.createStereoPanner();
    panner.pan.value =
      role === 'vocal'
        ? 0
        : role === 'bass'
          ? -0.22
          : (pitchNorm - 0.5) * 0.68;

    // Gentle automatic gain headroom when many voices sound together
    const polyScale = 1 / Math.sqrt(Math.max(1, this.activeVoices.size * 0.45 + 1));

    const voiceGain = ctx.createGain();
    const peakGain = vel * roleGainScale * polyScale * (1.08 - pitchNorm * 0.24);
    voiceGain.gain.setValueAtTime(0.0001, now);
    voiceGain.gain.linearRampToValueAtTime(peakGain, now + 0.007);
    voiceGain.gain.exponentialRampToValueAtTime(
      Math.max(0.0005, peakGain * 0.4),
      now + Math.min(holdDur, naturalDecay * 0.48),
    );
    voiceGain.gain.setTargetAtTime(0.0001, now + holdDur, releaseTime * 0.3);

    filter.connect(voiceGain);
    voiceGain.connect(panner);
    panner.connect(this.dryBus);
    panner.connect(this.wetBus);

    const B = 0.00013 + pitchNorm * 0.00032;
    const partials: Array<{ n: number; amp: number; type: OscillatorType }> =
      role === 'vocal'
        ? [
            { n: 1, amp: 1.0, type: 'triangle' },
            { n: 2, amp: 0.44, type: 'sine' },
            { n: 3, amp: 0.18, type: 'sine' },
            { n: 4, amp: 0.07, type: 'sine' },
          ]
        : role === 'bass'
          ? [
              { n: 1, amp: 1.0, type: 'sine' },
              { n: 2, amp: 0.42, type: 'triangle' },
              { n: 3, amp: 0.14, type: 'sine' },
            ]
          : [
              { n: 1, amp: 1.0, type: 'triangle' },
              { n: 2, amp: 0.32, type: 'sine' },
              { n: 3, amp: 0.13, type: 'sine' },
            ];

    const oscs: OscillatorNode[] = [];
    for (const p of partials) {
      const partialFreq = freq * p.n * Math.sqrt(1 + B * p.n * p.n);
      if (partialFreq > ctx.sampleRate * 0.45) continue;

      const osc = ctx.createOscillator();
      osc.type = p.type;
      osc.frequency.setValueAtTime(partialFreq, now);
      if (p.n === 2) osc.detune.value = 1.4;

      const pGain = ctx.createGain();
      pGain.gain.value = p.amp;
      osc.connect(pGain);
      pGain.connect(filter);
      osc.start(now);
      osc.stop(stopTime + 0.08);
      oscs.push(osc);
    }

    if (this.noiseBuffer && role !== 'bass') {
      const noise = ctx.createBufferSource();
      noise.buffer = this.noiseBuffer;
      const nFilter = ctx.createBiquadFilter();
      nFilter.type = 'bandpass';
      nFilter.frequency.value = Math.min(3000, 620 + pitchNorm * 1900);
      nFilter.Q.value = 1.6;
      const nGain = ctx.createGain();
      nGain.gain.value = vel * 0.055 * polyScale;
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
      startedAt: now,
      stopAt: (when: number) => {
        window.clearTimeout(cleanupTimer);
        try {
          voiceGain.gain.cancelScheduledValues(when);
          voiceGain.gain.setTargetAtTime(0.0001, when, 0.035);
          for (const o of oscs) {
            o.stop(when + 0.14);
          }
        } catch {
          // already stopped
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

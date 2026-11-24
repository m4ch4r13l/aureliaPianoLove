import * as Tone from 'tone';
import { MAX_MIDI, MIN_MIDI, midiToFreq } from './pianoTheory';
import { TrackRole } from './pitchDsp';

/**
 * Concert Grand Piano Synthesizer powered by Tone.js.
 * Routes 4 role-calibrated PolySynth voices (vocal, chords, bass, lead) through
 * a high-pass anti-rumble filter, concert-hall reverb, dynamics compressor,
 * and brickwall limiter for lush, distortion-free playback.
 */
export class GrandPianoSynth {
  private initialized = false;
  private masterVolumeNode: Tone.Gain | null = null;
  private highpass: Tone.Filter | null = null;
  private reverb: Tone.Reverb | null = null;
  private compressor: Tone.Compressor | null = null;
  private limiter: Tone.Limiter | null = null;

  private synths: Partial<Record<TrackRole, Tone.PolySynth>> = {};
  private originalPlayer: Tone.Player | null = null;
  private originalGain: Tone.Gain | null = null;

  private masterVolume = 0.85;
  private originalAudioMix = 0.0;

  public ensureContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;

    if (!this.initialized) {
      this.setupToneGraph();
    }

    // Unlock Tone.js audio context on user interaction
    Tone.start().catch(() => {});

    const rawCtx = Tone.getContext().rawContext as AudioContext | undefined;
    if (rawCtx && rawCtx.state === 'suspended') {
      rawCtx.resume().catch(() => {});
    }
    return rawCtx ?? null;
  }

  private setupToneGraph(): void {
    this.initialized = true;

    this.highpass = new Tone.Filter({
      type: 'highpass',
      frequency: 36,
      rolloff: -12,
    });

    this.reverb = new Tone.Reverb({
      decay: 2.2,
      preDelay: 0.018,
      wet: 0.24,
    });

    this.compressor = new Tone.Compressor({
      threshold: -16,
      ratio: 4.5,
      attack: 0.004,
      release: 0.22,
    });

    this.limiter = new Tone.Limiter(-2);

    this.masterVolumeNode = new Tone.Gain(this.masterVolume);

    this.highpass.connect(this.reverb);
    this.reverb.connect(this.compressor);
    this.compressor.connect(this.limiter);
    this.limiter.connect(this.masterVolumeNode);
    this.masterVolumeNode.toDestination();

    // 1. Vocal Melody — expressive singing concert grand right-hand voice
    const vocalSynth = new Tone.PolySynth(Tone.Synth, {
      oscillator: {
        type: 'custom',
        partials: [1.0, 0.48, 0.22, 0.11, 0.04],
      },
      envelope: {
        attack: 0.008,
        decay: 0.55,
        sustain: 0.36,
        release: 0.42,
      },
      volume: -6,
    });
    vocalSynth.maxPolyphony = 6;
    vocalSynth.connect(this.highpass);

    // 2. Chords — warm mellow felt-hammer accompaniment
    const chordsSynth = new Tone.PolySynth(Tone.Synth, {
      oscillator: {
        type: 'custom',
        partials: [1.0, 0.34, 0.14, 0.05],
      },
      envelope: {
        attack: 0.012,
        decay: 0.65,
        sustain: 0.28,
        release: 0.55,
      },
      volume: -10,
    });
    chordsSynth.maxPolyphony = 10;
    chordsSynth.connect(this.highpass);

    // 3. Bass — deep warm concert grand left-hand strings
    const bassSynth = new Tone.PolySynth(Tone.Synth, {
      oscillator: {
        type: 'custom',
        partials: [1.0, 0.42, 0.12, 0.03],
      },
      envelope: {
        attack: 0.014,
        decay: 0.85,
        sustain: 0.34,
        release: 0.65,
      },
      volume: -8,
    });
    bassSynth.maxPolyphony = 4;
    bassSynth.connect(this.highpass);

    // 4. Lead / Arreglos — crystalline upper register articulation
    const leadSynth = new Tone.PolySynth(Tone.Synth, {
      oscillator: {
        type: 'custom',
        partials: [1.0, 0.44, 0.24, 0.09],
      },
      envelope: {
        attack: 0.006,
        decay: 0.42,
        sustain: 0.22,
        release: 0.38,
      },
      volume: -11,
    });
    leadSynth.maxPolyphony = 6;
    leadSynth.connect(this.highpass);

    this.synths = {
      vocal: vocalSynth,
      chords: chordsSynth,
      bass: bassSynth,
      lead: leadSynth,
    };
  }

  public setVolume(vol: number): void {
    this.masterVolume = Math.max(0, Math.min(1, vol));
    if (this.masterVolumeNode) {
      this.masterVolumeNode.gain.rampTo(this.masterVolume, 0.04);
    }
  }

  public setOriginalAudioMix(mix: number): void {
    this.originalAudioMix = Math.max(0, Math.min(1, mix));
    if (this.originalGain) {
      this.originalGain.gain.rampTo(this.originalAudioMix * 0.65, 0.04);
    }
  }

  public startOriginalAudio(buffer: AudioBuffer, offsetSec: number, playbackRate: number): void {
    this.stopOriginalAudio();
    this.ensureContext();
    if (!this.masterVolumeNode) return;

    try {
      const toneBuf = new Tone.ToneAudioBuffer(buffer);
      const gain = new Tone.Gain(this.originalAudioMix * 0.65);
      const player = new Tone.Player(toneBuf);
      player.playbackRate = playbackRate;
      player.connect(gain);
      gain.connect(this.masterVolumeNode);

      const safeOffset = Math.max(0, Math.min(buffer.duration - 0.05, offsetSec));
      player.start(Tone.now(), safeOffset);

      this.originalPlayer = player;
      this.originalGain = gain;
    } catch {
      // Non-fatal if browser blocks audio before gesture
    }
  }

  public stopOriginalAudio(): void {
    if (this.originalPlayer) {
      try {
        this.originalPlayer.stop();
        this.originalPlayer.dispose();
      } catch {
        // already stopped
      }
      this.originalPlayer = null;
    }
    if (this.originalGain) {
      try {
        this.originalGain.dispose();
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
    role: TrackRole = 'chords',
  ): void {
    this.ensureContext();
    const synth = this.synths[role] ?? this.synths.chords;
    if (!synth) return;

    const clampedMidi = Math.max(MIN_MIDI, Math.min(MAX_MIDI, Math.round(midi)));
    const freq = midiToFreq(clampedMidi);
    const vel = Math.max(0.14, Math.min(1.0, velocity));
    const dur = Math.max(0.08, Math.min(4.2, durationSec));

    try {
      synth.triggerAttackRelease(freq, dur, Tone.now(), vel);
    } catch {
      // Prevent any single voice scheduling glitch from interrupting playback
    }
  }

  public stopAll(): void {
    this.stopOriginalAudio();
    for (const synth of Object.values(this.synths)) {
      try {
        synth?.releaseAll(Tone.now());
      } catch {
        // ignore
      }
    }
  }

  public dispose(): void {
    this.stopAll();
    for (const synth of Object.values(this.synths)) {
      try {
        synth?.dispose();
      } catch {
        // ignore
      }
    }
    this.synths = {};
    this.highpass?.dispose();
    this.reverb?.dispose();
    this.compressor?.dispose();
    this.limiter?.dispose();
    this.masterVolumeNode?.dispose();
    this.highpass = null;
    this.reverb = null;
    this.compressor = null;
    this.limiter = null;
    this.masterVolumeNode = null;
    this.initialized = false;
  }
}

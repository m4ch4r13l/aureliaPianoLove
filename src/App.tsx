import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from 'react';
import { Loader2, Upload } from 'lucide-react';
import { DEMO_PIECES } from './audio/demoPieces';
import { decodeAndTranscribeAudioFile } from './audio/audioTranscriber';
import { GrandPianoSynth } from './audio/pianoSynth';
import { KEYBOARD_KEY_TO_MIDI } from './audio/pianoTheory';
import { PianoNoteEvent, TrackRole, TranscribedPiece } from './audio/pitchDsp';
import {
  CameraPresetId,
  GrandPianoViewport,
  PianoViewportHandle,
} from './components/GrandPianoViewport';
import { ControlDock, TrackFilterMode } from './components/ControlDock';
import { LidState } from './three/grandPianoBuilder';

export default function App() {
  const [pieces, setPieces] = useState<TranscribedPiece[]>(DEMO_PIECES);
  const [currentPieceId, setCurrentPieceId] = useState<string>(DEMO_PIECES[0].id);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [trackFilter, setTrackFilter] = useState<TrackFilterMode>('both');
  const [playbackRate, setPlaybackRate] = useState(1);
  const [cameraPreset, setCameraPreset] = useState<CameraPresetId>('concert');
  const [lidState, setLidState] = useState<LidState>('open');
  const [showRibbons, setShowRibbons] = useState(true);
  const [isMuted, setIsMuted] = useState(false);
  const [mixOriginalAudio, setMixOriginalAudio] = useState(false);
  const [dockCollapsed, setDockCollapsed] = useState(false);

  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [analysisState, setAnalysisState] = useState<{
    active: boolean;
    progress: number;
    stage: string;
    error: string | null;
  }>({ active: false, progress: 0, stage: '', error: null });

  const [activeVocalMidis, setActiveVocalMidis] = useState<number[]>([]);
  const [activePianoMidis, setActivePianoMidis] = useState<number[]>([]);

  const currentPiece = pieces.find((p) => p.id === currentPieceId) ?? pieces[0];

  const viewportRef = useRef<PianoViewportHandle | null>(null);
  const synthRef = useRef<GrandPianoSynth | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadedBuffersRef = useRef<Map<string, AudioBuffer>>(new Map());

  // Playback engine mutable refs (avoids 60Hz React re-renders)
  const playStateRef = useRef({
    playing: false,
    songTime: 0,
    lastFrameMs: 0,
    triggeredIds: new Set<string>(),
    activeNotes: new Map<number, { endTime: number; role: TrackRole }>(),
    piece: currentPiece,
    trackFilter,
    playbackRate,
    mixOriginalAudio,
  });

  playStateRef.current.piece = currentPiece;
  playStateRef.current.trackFilter = trackFilter;
  playStateRef.current.playbackRate = playbackRate;
  playStateRef.current.mixOriginalAudio = mixOriginalAudio;

  // Initialize procedural grand piano synthesizer once
  useEffect(() => {
    const synth = new GrandPianoSynth();
    synthRef.current = synth;
    return () => {
      synth.dispose();
      synthRef.current = null;
    };
  }, []);

  useEffect(() => {
    synthRef.current?.setVolume(isMuted ? 0 : 0.85);
  }, [isMuted]);

  useEffect(() => {
    synthRef.current?.setOriginalAudioMix(mixOriginalAudio ? 0.55 : 0);
  }, [mixOriginalAudio]);

  const stopPlayback = useCallback((resetTime = false) => {
    const ps = playStateRef.current;
    ps.playing = false;
    ps.triggeredIds.clear();
    ps.activeNotes.clear();
    if (resetTime) {
      ps.songTime = 0;
      setCurrentTime(0);
    }
    synthRef.current?.stopAll();
    viewportRef.current?.clearAllKeys();
    viewportRef.current?.setSustainActive(false);
    setIsPlaying(false);
    setActiveVocalMidis([]);
    setActivePianoMidis([]);
  }, []);

  const syncOriginalAudioPlayback = useCallback((pieceId: string, timeSec: number, rate: number) => {
    const buf = uploadedBuffersRef.current.get(pieceId);
    if (buf) {
      synthRef.current?.startOriginalAudio(buf, timeSec, rate);
    } else {
      synthRef.current?.stopOriginalAudio();
    }
  }, []);

  const startPlayback = useCallback(
    (fromTime?: number) => {
      const ps = playStateRef.current;
      const piece = ps.piece;
      const startAt =
        fromTime !== undefined
          ? fromTime
          : ps.songTime >= piece.duration - 0.15
            ? 0
            : ps.songTime;

      ps.songTime = startAt;
      ps.lastFrameMs = performance.now();
      ps.triggeredIds.clear();
      ps.activeNotes.clear();

      // Mark earlier notes as already triggered when resuming mid-song
      for (const note of piece.notes) {
        if (note.startTime < startAt - 0.04) {
          ps.triggeredIds.add(note.id);
        }
      }

      synthRef.current?.ensureContext();
      syncOriginalAudioPlayback(piece.id, startAt, ps.playbackRate);

      ps.playing = true;
      setCurrentTime(startAt);
      setIsPlaying(true);
      viewportRef.current?.requestFrame();
    },
    [syncOriginalAudioPlayback],
  );

  const handleTogglePlay = useCallback(() => {
    if (playStateRef.current.playing) {
      stopPlayback(false);
    } else {
      startPlayback();
    }
  }, [startPlayback, stopPlayback]);

  const handleRestart = useCallback(() => {
    stopPlayback(true);
    startPlayback(0);
  }, [startPlayback, stopPlayback]);

  const handleSeek = useCallback(
    (timeSec: number) => {
      const clamped = Math.max(0, Math.min(playStateRef.current.piece.duration, timeSec));
      const wasPlaying = playStateRef.current.playing;
      stopPlayback(false);
      playStateRef.current.songTime = clamped;
      setCurrentTime(clamped);
      if (wasPlaying) {
        startPlayback(clamped);
      }
    },
    [startPlayback, stopPlayback],
  );

  const handleSelectPiece = useCallback(
    (pieceId: string) => {
      stopPlayback(true);
      setCurrentPieceId(pieceId);
      const hasBuf = uploadedBuffersRef.current.has(pieceId);
      if (!hasBuf) setMixOriginalAudio(false);
    },
    [stopPlayback],
  );

  // Sync playback rate into original audio buffer if playing
  useEffect(() => {
    if (playStateRef.current.playing) {
      syncOriginalAudioPlayback(
        playStateRef.current.piece.id,
        playStateRef.current.songTime,
        playbackRate,
      );
    }
  }, [playbackRate, syncOriginalAudioPlayback]);

  // High-precision playback scheduler driving Web Audio + 3D keys (throttled React HUD updates <= 10Hz)
  useEffect(() => {
    if (!isPlaying) return;

    let rafId = 0;
    let lastHudSync = 0;

    const noteMatchesFilter = (note: PianoNoteEvent, filter: TrackFilterMode): boolean => {
      if (filter === 'both') return true;
      return note.track === filter;
    };

    const tick = (nowMs: number) => {
      const ps = playStateRef.current;
      if (!ps.playing) return;

      const dtSec = Math.min(0.1, Math.max(0, (nowMs - ps.lastFrameMs) / 1000));
      ps.lastFrameMs = nowMs;
      ps.songTime += dtSec * ps.playbackRate;

      const piece = ps.piece;

      // Trigger notes crossing current playback head
      for (const note of piece.notes) {
        if (note.startTime > ps.songTime + 0.02) break;
        if (ps.triggeredIds.has(note.id)) continue;
        ps.triggeredIds.add(note.id);

        if (!noteMatchesFilter(note, ps.trackFilter)) continue;

        const scaledDur = note.duration / ps.playbackRate;
        synthRef.current?.playNote(note.midi, note.velocity, scaledDur, note.track);
        viewportRef.current?.setKeyActive(note.midi, true, note.track, note.velocity);
        ps.activeNotes.set(note.midi, {
          endTime: note.startTime + note.duration,
          role: note.track,
        });
      }

      // Release finished notes
      let anyPianoActive = false;
      for (const [midi, info] of ps.activeNotes.entries()) {
        if (ps.songTime >= info.endTime) {
          ps.activeNotes.delete(midi);
          viewportRef.current?.setKeyActive(midi, false, null, 0);
        } else if (info.role === 'piano') {
          anyPianoActive = true;
        }
      }
      viewportRef.current?.setSustainActive(anyPianoActive);

      // Throttled HUD update (~10Hz)
      if (nowMs - lastHudSync >= 100) {
        lastHudSync = nowMs;
        setCurrentTime(Math.min(ps.songTime, piece.duration));
        const vocals: number[] = [];
        const pianos: number[] = [];
        for (const [midi, info] of ps.activeNotes.entries()) {
          if (info.role === 'vocal') vocals.push(midi);
          else pianos.push(midi);
        }
        vocals.sort((a, b) => a - b);
        pianos.sort((a, b) => a - b);
        setActiveVocalMidis(vocals);
        setActivePianoMidis(pianos);
      }

      if (ps.songTime >= piece.duration) {
        stopPlayback(false);
        setCurrentTime(piece.duration);
        return;
      }

      rafId = requestAnimationFrame(tick);
    };

    playStateRef.current.lastFrameMs = performance.now();
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [isPlaying, stopPlayback]);

  // Direct interactive playing (clicking 3D keys or computer keyboard)
  const handleManualNotePlay = useCallback((midi: number) => {
    synthRef.current?.ensureContext();
    synthRef.current?.playNote(midi, 0.85, 1.4, 'vocal');
    viewportRef.current?.setKeyActive(midi, true, 'vocal', 0.88);
    setActiveVocalMidis((prev) => (prev.includes(midi) ? prev : [...prev.slice(-2), midi]));
    window.setTimeout(() => {
      if (!playStateRef.current.activeNotes.has(midi)) {
        viewportRef.current?.setKeyActive(midi, false, null, 0);
        setActiveVocalMidis((prev) => prev.filter((m) => m !== midi));
      }
    }, 340);
  }, []);

  // Global keyboard shortcuts: Space = Play/Pause, H = Hide/Show dock, Z-M / Q-P = Live piano keys
  useEffect(() => {
    const heldCodes = new Set<string>();

    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA')
      ) {
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        handleTogglePlay();
        return;
      }
      if (e.key === 'h' || e.key === 'H') {
        setDockCollapsed((prev) => !prev);
        return;
      }

      const midi = KEYBOARD_KEY_TO_MIDI[e.code];
      if (midi !== undefined && !heldCodes.has(e.code)) {
        heldCodes.add(e.code);
        handleManualNotePlay(midi);
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      heldCodes.delete(e.code);
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [handleManualNotePlay, handleTogglePlay]);

  // MP3 / Audio File Upload & Real Transcription Flow
  const processAudioFile = useCallback(
    async (file: File) => {
      stopPlayback(true);
      setAnalysisState({
        active: true,
        progress: 0.03,
        stage: `Cargando "${file.name}"...`,
        error: null,
      });

      try {
        const result = await decodeAndTranscribeAudioFile(file, (progress, stage) => {
          setAnalysisState((prev) => ({ ...prev, progress, stage }));
        });

        uploadedBuffersRef.current.set(result.piece.id, result.audioBuffer);
        setPieces((prev) => [result.piece, ...prev]);
        setCurrentPieceId(result.piece.id);
        playStateRef.current.piece = result.piece;
        setAnalysisState({ active: false, progress: 1, stage: '', error: null });

        // Automatically start playing the newly transcribed piece on the grand piano!
        window.setTimeout(() => {
          startPlayback(0);
        }, 60);
      } catch (err) {
        const msg =
          err instanceof Error && err.message
            ? err.message
            : 'No se pudo decodificar el archivo de audio. Prueba con un MP3 o WAV válido.';
        setAnalysisState({
          active: false,
          progress: 0,
          stage: '',
          error: msg,
        });
      }
    },
    [startPlayback, stopPlayback],
  );

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (file) {
      void processAudioFile(file);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isDraggingFile) setIsDraggingFile(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    if (e.currentTarget === e.target) {
      setIsDraggingFile(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingFile(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      void processAudioFile(file);
    }
  };

  const hasOriginalAudio = uploadedBuffersRef.current.has(currentPiece.id);

  return (
    <main
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      data-playing={isPlaying ? 'true' : 'false'}
      data-track-source={currentPiece.source}
      data-camera-preset={cameraPreset}
      data-track-filter={trackFilter}
      data-lid-state={lidState}
      data-note-count={currentPiece.notes.length}
      className="relative h-full w-full overflow-hidden bg-[#0c0e13]"
    >
      {/* Full-Viewport 3D Concert Grand Piano Stage */}
      <GrandPianoViewport
        ref={viewportRef}
        cameraPreset={cameraPreset}
        lidState={lidState}
        showRibbons={showRibbons}
        isPlaying={isPlaying}
        onUserKeyPlay={handleManualNotePlay}
      />

      {/* Hidden MP3 File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.ogg,.m4a"
        onChange={handleFileInputChange}
        aria-label="Seleccionar archivo MP3 o de audio"
        data-testid="mp3-file-input"
        className="sr-only"
      />

      {/* Minimal Top-Left Brand & Unboxed Piece Metadata (zero pills, zero side clutter) */}
      <header className="pointer-events-none fixed left-5 top-5 z-20 max-w-[min(460px,82vw)] sm:left-7 sm:top-6">
        <h1 className="font-display text-3xl font-medium tracking-tight text-[#f6f1e6] sm:text-4xl">
          Aurelia
        </h1>
        <div
          data-testid="piece-metadata"
          className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-[#b7b0a3]"
        >
          <span className="font-medium text-[#e4dccd]">{currentPiece.title}</span>
          <span aria-hidden="true">·</span>
          <span>{currentPiece.keySignature}</span>
          <span aria-hidden="true">·</span>
          <span className="font-mono-tabular">{currentPiece.bpm} BPM</span>
          <span aria-hidden="true">·</span>
          <span className="font-mono-tabular">
            {currentPiece.pianoNoteCount} piano / {currentPiece.vocalNoteCount} voz
          </span>
        </div>
      </header>

      {/* Analysis Error Toast (only when an invalid file fails to decode) */}
      {analysisState.error && (
        <div
          role="alert"
          className="fixed left-1/2 top-6 z-30 flex max-w-md -translate-x-1/2 items-center gap-3 rounded-lg border border-[#f26b7a]/40 bg-[#1a1114]/95 px-4 py-2.5 text-xs text-[#f7c5cb] shadow-xl"
        >
          <span>{analysisState.error}</span>
          <button
            type="button"
            onClick={() => setAnalysisState((prev) => ({ ...prev, error: null }))}
            className="font-medium text-[#f3efe7] underline underline-offset-2"
          >
            Cerrar
          </button>
        </div>
      )}

      {/* Audio Transcription Progress Overlay */}
      {analysisState.active && (
        <div
          role="status"
          aria-live="polite"
          className="fixed inset-0 z-30 flex items-center justify-center bg-[#0c0e13]/80 p-6"
        >
          <div className="w-full max-w-sm rounded-xl border border-white/12 bg-[#12151c] p-6 shadow-2xl">
            <div className="flex items-center gap-3">
              <Loader2 className="h-5 w-5 animate-spin text-[#d4a353]" />
              <h2 className="font-display text-2xl font-medium text-[#f6f1e6]">
                Calculando partitura
              </h2>
            </div>
            <p className="mt-2 text-xs text-[#b7b0a3]">{analysisState.stage}</p>
            <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full bg-[#d4a353] transition-all duration-150"
                style={{ width: `${Math.max(6, Math.round(analysisState.progress * 100))}%` }}
              />
            </div>
            <p className="font-mono-tabular mt-2 text-right text-[11px] text-[#9c968b]">
              {Math.round(analysisState.progress * 100)}%
            </p>
          </div>
        </div>
      )}

      {/* Drag & Drop MP3 Overlay */}
      {isDraggingFile && (
        <div className="pointer-events-none fixed inset-0 z-30 flex items-center justify-center bg-[#0c0e13]/80 p-6">
          <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-[#d4a353] bg-[#14171f] px-10 py-8 text-center">
            <Upload className="h-8 w-8 text-[#d4a353]" />
            <p className="font-display text-2xl text-[#f6f1e6]">Suelta tu archivo MP3</p>
            <p className="text-xs text-[#b7b0a3]">
              Calcularemos las notas del piano y la melodía de voz automáticamente
            </p>
          </div>
        </div>
      )}

      {/* Unified Collapsible Bottom Control Dock */}
      <ControlDock
        pieces={pieces}
        currentPiece={currentPiece}
        onSelectPiece={handleSelectPiece}
        onUploadClick={() => fileInputRef.current?.click()}
        isAnalyzing={analysisState.active}
        isPlaying={isPlaying}
        onTogglePlay={handleTogglePlay}
        onRestart={handleRestart}
        currentTime={currentTime}
        onSeek={handleSeek}
        trackFilter={trackFilter}
        onTrackFilterChange={setTrackFilter}
        playbackRate={playbackRate}
        onPlaybackRateChange={setPlaybackRate}
        cameraPreset={cameraPreset}
        onCameraPresetChange={setCameraPreset}
        lidState={lidState}
        onLidStateChange={setLidState}
        showRibbons={showRibbons}
        onToggleRibbons={() => setShowRibbons((prev) => !prev)}
        isMuted={isMuted}
        onToggleMute={() => setIsMuted((prev) => !prev)}
        hasOriginalAudio={hasOriginalAudio}
        mixOriginalAudio={mixOriginalAudio}
        onToggleMixOriginalAudio={() => setMixOriginalAudio((prev) => !prev)}
        activeVocalMidis={activeVocalMidis}
        activePianoMidis={activePianoMidis}
        collapsed={dockCollapsed}
        onToggleCollapsed={() => setDockCollapsed((prev) => !prev)}
      />
    </main>
  );
}

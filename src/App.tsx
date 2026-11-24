import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, Upload } from 'lucide-react';
import { DEMO_PIECES } from './audio/demoPieces';
import { decodeAndTranscribeAudioFile, retranscribeAudioBuffer } from './audio/audioTranscriber';
import { GrandPianoSynth } from './audio/pianoSynth';
import { KEYBOARD_KEY_TO_MIDI } from './audio/pianoTheory';
import {
  CleanlinessLevel,
  filterAndCleanNotes,
  PianoNoteEvent,
  TrackRole,
  TranscribedPiece,
  TranscriptionMode,
} from './audio/pitchDsp';
import { buildMultiTrackMidi } from './audio/midiExporter';
import { hasMidiHeader, isMidiFile, parseMidiToPiece } from './audio/midiParser';
import {
  CameraPresetId,
  GrandPianoViewport,
  PianoViewportHandle,
} from './components/GrandPianoViewport';
import { ControlDock } from './components/ControlDock';
import { LidState } from './three/grandPianoBuilder';

const EMPTY_ACTIVE_BY_ROLE: Record<TrackRole, number[]> = {
  vocal: [],
  chords: [],
  bass: [],
  lead: [],
};

export default function App() {
  const [pieces, setPieces] = useState<TranscribedPiece[]>(DEMO_PIECES);
  const [currentPieceId, setCurrentPieceId] = useState<string>(DEMO_PIECES[0].id);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  // 4 separated instrument tracks — independently toggleable at any moment
  const [enabledTracks, setEnabledTracks] = useState<Record<TrackRole, boolean>>({
    vocal: true,
    chords: true,
    bass: true,
    lead: true,
  });
  // Cleanliness filter defaults to 'strict' (Alta) for pop demos; solo piano uploads auto-select 'soft'
  const [cleanliness, setCleanliness] = useState<CleanlinessLevel>('strict');
  // MP3 transcription preset: defaults to 'piano' (optimized for solo piano recordings)
  const [transcriptionMode, setTranscriptionMode] = useState<TranscriptionMode>('piano');

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

  const [activeNotesByRole, setActiveNotesByRole] =
    useState<Record<TrackRole, number[]>>(EMPTY_ACTIVE_BY_ROLE);

  const currentPiece = pieces.find((p) => p.id === currentPieceId) ?? pieces[0];

  // Live cleaned & separated note list (updates instantaneously when toggling tracks or cleanliness)
  const filteredNotes = useMemo(
    () => filterAndCleanNotes(currentPiece, enabledTracks, cleanliness),
    [currentPiece, enabledTracks, cleanliness],
  );

  const viewportRef = useRef<PianoViewportHandle | null>(null);
  const synthRef = useRef<GrandPianoSynth | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadedBuffersRef = useRef<Map<string, AudioBuffer>>(new Map());

  const playStateRef = useRef<{
    playing: boolean;
    songTime: number;
    lastFrameMs: number;
    triggeredIds: Set<string>;
    activeNotes: Map<number, { endTime: number; role: TrackRole }>;
    piece: TranscribedPiece;
    notes: PianoNoteEvent[];
    playbackRate: number;
  }>({
    playing: false,
    songTime: 0,
    lastFrameMs: 0,
    triggeredIds: new Set<string>(),
    activeNotes: new Map(),
    piece: currentPiece,
    notes: filteredNotes,
    playbackRate,
  });

  playStateRef.current.piece = currentPiece;
  playStateRef.current.notes = filteredNotes;
  playStateRef.current.playbackRate = playbackRate;

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
    setActiveNotesByRole(EMPTY_ACTIVE_BY_ROLE);
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

      for (const note of ps.notes) {
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
      if (!uploadedBuffersRef.current.has(pieceId)) {
        setMixOriginalAudio(false);
      }
    },
    [stopPlayback],
  );

  const handleToggleTrack = useCallback((role: TrackRole) => {
    setEnabledTracks((prev) => ({ ...prev, [role]: !prev[role] }));
  }, []);

  // When cleanliness or tracks change mid-playback, mark already-past notes as triggered so we don't burst old notes
  useEffect(() => {
    const ps = playStateRef.current;
    if (!ps.playing) return;
    for (const note of filteredNotes) {
      if (note.startTime < ps.songTime - 0.05) {
        ps.triggeredIds.add(note.id);
      }
    }
  }, [filteredNotes]);

  useEffect(() => {
    if (playStateRef.current.playing) {
      syncOriginalAudioPlayback(
        playStateRef.current.piece.id,
        playStateRef.current.songTime,
        playbackRate,
      );
    }
  }, [playbackRate, syncOriginalAudioPlayback]);

  // High-precision playback scheduler (throttled React HUD sync at 10Hz)
  useEffect(() => {
    if (!isPlaying) return;

    let rafId = 0;
    let lastHudSync = 0;

    const tick = (nowMs: number) => {
      const ps = playStateRef.current;
      if (!ps.playing) return;

      const dtSec = Math.min(0.1, Math.max(0, (nowMs - ps.lastFrameMs) / 1000));
      ps.lastFrameMs = nowMs;
      ps.songTime += dtSec * ps.playbackRate;

      const piece = ps.piece;
      const notes = ps.notes;

      for (const note of notes) {
        if (note.startTime > ps.songTime + 0.02) break;
        if (ps.triggeredIds.has(note.id)) continue;
        ps.triggeredIds.add(note.id);

        const scaledDur = note.duration / ps.playbackRate;
        synthRef.current?.playNote(note.midi, note.velocity, scaledDur, note.track);
        viewportRef.current?.setKeyActive(note.midi, true, note.track, note.velocity);
        ps.activeNotes.set(note.midi, {
          endTime: note.startTime + note.duration,
          role: note.track,
        });
      }

      let anyHarmonicActive = false;
      for (const [midi, info] of ps.activeNotes.entries()) {
        if (ps.songTime >= info.endTime) {
          ps.activeNotes.delete(midi);
          viewportRef.current?.setKeyActive(midi, false, null, 0);
        } else if (info.role === 'chords' || info.role === 'bass') {
          anyHarmonicActive = true;
        }
      }
      viewportRef.current?.setSustainActive(anyHarmonicActive);

      if (nowMs - lastHudSync >= 100) {
        lastHudSync = nowMs;
        setCurrentTime(Math.min(ps.songTime, piece.duration));
        const grouped: Record<TrackRole, number[]> = {
          vocal: [],
          chords: [],
          bass: [],
          lead: [],
        };
        for (const [midi, info] of ps.activeNotes.entries()) {
          grouped[info.role].push(midi);
        }
        grouped.vocal.sort((a, b) => a - b);
        grouped.chords.sort((a, b) => a - b);
        grouped.bass.sort((a, b) => a - b);
        grouped.lead.sort((a, b) => a - b);
        setActiveNotesByRole(grouped);
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

  // Direct interactive playing
  const handleManualNotePlay = useCallback((midi: number) => {
    synthRef.current?.ensureContext();
    synthRef.current?.playNote(midi, 0.85, 1.4, 'vocal');
    viewportRef.current?.setKeyActive(midi, true, 'vocal', 0.88);
    setActiveNotesByRole((prev) => ({
      ...prev,
      vocal: prev.vocal.includes(midi) ? prev.vocal : [...prev.vocal.slice(-2), midi],
    }));
    window.setTimeout(() => {
      if (!playStateRef.current.activeNotes.has(midi)) {
        viewportRef.current?.setKeyActive(midi, false, null, 0);
        setActiveNotesByRole((prev) => ({
          ...prev,
          vocal: prev.vocal.filter((m) => m !== midi),
        }));
      }
    }, 340);
  }, []);

  // Export cleaned multi-track MIDI (.mid) separated into 4 instrument tracks
  const handleExportMidi = useCallback(() => {
    const bytes = buildMultiTrackMidi(currentPiece, filteredNotes);
    const blob = new Blob([bytes.buffer as ArrayBuffer], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const safeName = currentPiece.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/gi, '-')
      .replace(/^-|-$/g, '');
    a.href = url;
    a.download = `${safeName || 'aurelia-piano'}-multipista.mid`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [currentPiece, filteredNotes]);

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

  const processAudioFile = useCallback(
    async (file: File) => {
      synthRef.current?.ensureContext();
      stopPlayback(true);
      setAnalysisState({
        active: true,
        progress: 0.02,
        stage: `Cargando "${file.name}"...`,
        error: null,
      });

      try {
        const headerBytes = new Uint8Array(await file.slice(0, 128).arrayBuffer());
        const treatAsMidi = isMidiFile(file) || hasMidiHeader(headerBytes);

        if (treatAsMidi) {
          setAnalysisState({
            active: true,
            progress: 0.45,
            stage: `Leyendo pistas MIDI de "${file.name}"...`,
            error: null,
          });
          const fullBytes = new Uint8Array(await file.arrayBuffer());
          const cleanName =
            file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim() || 'Archivo MIDI';
          const midiPiece = parseMidiToPiece(fullBytes, cleanName);
          const allTracks: Record<TrackRole, boolean> = {
            vocal: true,
            chords: true,
            bass: true,
            lead: true,
          };
          const initialFiltered = filterAndCleanNotes(midiPiece, allTracks, cleanliness);

          setEnabledTracks(allTracks);
          setMixOriginalAudio(false);
          setPieces((prev) => [midiPiece, ...prev]);
          setCurrentPieceId(midiPiece.id);
          playStateRef.current.piece = midiPiece;
          playStateRef.current.notes = initialFiltered;
          setAnalysisState({ active: false, progress: 1, stage: '', error: null });

          window.setTimeout(() => {
            startPlayback(0);
          }, 60);
          return;
        }

        const result = await decodeAndTranscribeAudioFile(
          file,
          (progress, stage) => {
            setAnalysisState((prev) => ({ ...prev, progress, stage }));
          },
          transcriptionMode,
        );

        uploadedBuffersRef.current.set(result.piece.id, result.audioBuffer);
        const defaultTracks: Record<TrackRole, boolean> = {
          vocal: true,
          chords: true,
          bass: true,
          lead: true,
        };
        const initialCleanliness: CleanlinessLevel =
          transcriptionMode === 'piano' ? 'soft' : 'balanced';
        const initialFiltered = filterAndCleanNotes(
          result.piece,
          defaultTracks,
          initialCleanliness,
        );

        setEnabledTracks(defaultTracks);
        setCleanliness(initialCleanliness);
        setPieces((prev) => [result.piece, ...prev]);
        setCurrentPieceId(result.piece.id);
        playStateRef.current.piece = result.piece;
        playStateRef.current.notes = initialFiltered;
        setAnalysisState({ active: false, progress: 1, stage: '', error: null });

        window.setTimeout(() => {
          startPlayback(0);
        }, 60);
      } catch (err) {
        const msg =
          err instanceof Error && err.message
            ? err.message
            : 'No se pudo leer el archivo. Prueba con un MP3, WAV o MIDI (.mid) válido.';
        setAnalysisState({
          active: false,
          progress: 0,
          stage: '',
          error: msg,
        });
      }
    },
    [cleanliness, startPlayback, stopPlayback, transcriptionMode],
  );

  // Switch between Piano (solo piano MP3s) and Canción (full mix with drums/vocals).
  // If an uploaded MP3 is currently active, re-transcribe it automatically in the new mode.
  const handleTranscriptionModeChange = useCallback(
    async (nextMode: TranscriptionMode) => {
      setTranscriptionMode(nextMode);
      const activePiece = playStateRef.current.piece;
      const buf = uploadedBuffersRef.current.get(activePiece.id);
      if (!buf || activePiece.source !== 'upload' || activePiece.transcriptionMode === nextMode) {
        return;
      }

      stopPlayback(true);
      setAnalysisState({
        active: true,
        progress: 0.08,
        stage:
          nextMode === 'piano'
            ? 'Re-analizando grabación en modo Piano (88 teclas)...'
            : 'Re-analizando pista en modo Canción (HPSS + Voz)...',
        error: null,
      });

      try {
        const retranscribed = await retranscribeAudioBuffer(
          buf,
          activePiece.title,
          nextMode,
          (progress, stage) => {
            setAnalysisState((prev) => ({ ...prev, progress, stage }));
          },
        );
        const updatedPiece: TranscribedPiece = {
          ...retranscribed,
          id: activePiece.id,
        };
        const nextClean: CleanlinessLevel = nextMode === 'piano' ? 'soft' : 'balanced';
        const nextFiltered = filterAndCleanNotes(updatedPiece, enabledTracks, nextClean);

        setCleanliness(nextClean);
        setPieces((prev) => prev.map((p) => (p.id === activePiece.id ? updatedPiece : p)));
        playStateRef.current.piece = updatedPiece;
        playStateRef.current.notes = nextFiltered;
        setAnalysisState({ active: false, progress: 1, stage: '', error: null });

        window.setTimeout(() => {
          startPlayback(0);
        }, 60);
      } catch (err) {
        const msg =
          err instanceof Error && err.message
            ? err.message
            : 'No se pudo re-analizar el audio en el nuevo modo.';
        setAnalysisState({ active: false, progress: 0, stage: '', error: msg });
      }
    },
    [enabledTracks, startPlayback, stopPlayback],
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
      data-transcription-mode={currentPiece.transcriptionMode ?? transcriptionMode}
      data-camera-preset={cameraPreset}
      data-cleanliness={cleanliness}
      data-lid-state={lidState}
      data-note-count={filteredNotes.length}
      className="relative h-full w-full overflow-hidden bg-[#161923]"
    >
      <GrandPianoViewport
        ref={viewportRef}
        cameraPreset={cameraPreset}
        lidState={lidState}
        showRibbons={showRibbons}
        isPlaying={isPlaying}
        onUserKeyPlay={handleManualNotePlay}
      />

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.ogg,.m4a,.mid,.midi,.kar,audio/midi,audio/x-midi"
        onChange={handleFileInputChange}
        aria-label="Seleccionar archivo MP3, audio o MIDI"
        data-testid="mp3-file-input"
        className="sr-only"
      />

      {/* Minimal Top-Left Brand & Unboxed Metadata */}
      <header className="pointer-events-none fixed left-5 top-5 z-20 max-w-[min(520px,86vw)] sm:left-7 sm:top-6">
        <h1 className="font-display text-3xl font-medium tracking-tight text-[#f6f1e6] sm:text-4xl">
          Aurelia
        </h1>
        <div
          data-testid="piece-metadata"
          className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-[#cfc8bc]"
        >
          <span className="font-medium text-[#f5efe4]">{currentPiece.title}</span>
          <span aria-hidden="true">·</span>
          <span>{currentPiece.keySignature}</span>
          <span aria-hidden="true">·</span>
          <span className="font-mono-tabular">{currentPiece.bpm} BPM</span>
          <span aria-hidden="true">·</span>
          <span className="font-mono-tabular">{filteredNotes.length} notas activas</span>
        </div>
      </header>

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
                Limpiando y separando pistas
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

      {isDraggingFile && (
        <div className="pointer-events-none fixed inset-0 z-30 flex items-center justify-center bg-[#0c0e13]/80 p-6">
          <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-[#d4a353] bg-[#14171f] px-10 py-8 text-center">
            <Upload className="h-8 w-8 text-[#d4a353]" />
            <p className="font-display text-2xl text-[#f6f1e6]">Suelta tu archivo MP3 o MIDI</p>
            <p className="text-xs text-[#b7b0a3]">
              Reproduce archivos .mid directamente o convierte tu MP3 en 4 pistas limpias
            </p>
          </div>
        </div>
      )}

      <ControlDock
        pieces={pieces}
        currentPiece={currentPiece}
        filteredNotes={filteredNotes}
        onSelectPiece={handleSelectPiece}
        onUploadClick={() => fileInputRef.current?.click()}
        onExportMidi={handleExportMidi}
        isAnalyzing={analysisState.active}
        isPlaying={isPlaying}
        onTogglePlay={handleTogglePlay}
        onRestart={handleRestart}
        currentTime={currentTime}
        onSeek={handleSeek}
        enabledTracks={enabledTracks}
        onToggleTrack={handleToggleTrack}
        cleanliness={cleanliness}
        onCleanlinessChange={setCleanliness}
        transcriptionMode={transcriptionMode}
        onTranscriptionModeChange={handleTranscriptionModeChange}
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
        activeNotesByRole={activeNotesByRole}
        collapsed={dockCollapsed}
        onToggleCollapsed={() => setDockCollapsed((prev) => !prev)}
      />
    </main>
  );
}

import { useEffect, useRef } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Download,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  Upload,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { formatTime, midiToSolfege } from '../audio/pianoTheory';
import {
  CleanlinessLevel,
  PianoNoteEvent,
  TrackRole,
  TranscribedPiece,
} from '../audio/pitchDsp';
import { CameraPresetId } from './GrandPianoViewport';
import { LidState } from '../three/grandPianoBuilder';

const INSTRUMENT_TRACKS: Array<{
  role: TrackRole;
  label: string;
  color: string;
  rollColor: string;
}> = [
  { role: 'vocal', label: 'Voz', color: '#f26b7a', rollColor: 'rgba(242, 107, 122, 0.9)' },
  { role: 'chords', label: 'Acordes', color: '#f2b84b', rollColor: 'rgba(242, 184, 75, 0.62)' },
  { role: 'bass', label: 'Bajo', color: '#d9823b', rollColor: 'rgba(217, 130, 59, 0.72)' },
  { role: 'lead', label: 'Arreglos', color: '#7db4e6', rollColor: 'rgba(125, 180, 230, 0.75)' },
];

interface ControlDockProps {
  pieces: TranscribedPiece[];
  currentPiece: TranscribedPiece;
  filteredNotes: PianoNoteEvent[];
  onSelectPiece: (pieceId: string) => void;
  onUploadClick: () => void;
  onExportMidi: () => void;
  isAnalyzing: boolean;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onRestart: () => void;
  currentTime: number;
  onSeek: (timeSec: number) => void;
  enabledTracks: Record<TrackRole, boolean>;
  onToggleTrack: (role: TrackRole) => void;
  cleanliness: CleanlinessLevel;
  onCleanlinessChange: (level: CleanlinessLevel) => void;
  playbackRate: number;
  onPlaybackRateChange: (rate: number) => void;
  cameraPreset: CameraPresetId;
  onCameraPresetChange: (preset: CameraPresetId) => void;
  lidState: LidState;
  onLidStateChange: (state: LidState) => void;
  showRibbons: boolean;
  onToggleRibbons: () => void;
  isMuted: boolean;
  onToggleMute: () => void;
  hasOriginalAudio: boolean;
  mixOriginalAudio: boolean;
  onToggleMixOriginalAudio: () => void;
  activeNotesByRole: Record<TrackRole, number[]>;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

export function ControlDock({
  pieces,
  currentPiece,
  filteredNotes,
  onSelectPiece,
  onUploadClick,
  onExportMidi,
  isAnalyzing,
  isPlaying,
  onTogglePlay,
  onRestart,
  currentTime,
  onSeek,
  enabledTracks,
  onToggleTrack,
  cleanliness,
  onCleanlinessChange,
  playbackRate,
  onPlaybackRateChange,
  cameraPreset,
  onCameraPresetChange,
  lidState,
  onLidStateChange,
  showRibbons,
  onToggleRibbons,
  isMuted,
  onToggleMute,
  hasOriginalAudio,
  mixOriginalAudio,
  onToggleMixOriginalAudio,
  activeNotesByRole,
  collapsed,
  onToggleCollapsed,
}: ControlDockProps) {
  const miniRollRef = useRef<HTMLCanvasElement | null>(null);

  // Render crisp 4-instrument mini piano roll reflecting live cleanliness & muted tracks
  useEffect(() => {
    const canvas = miniRollRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.fillRect(0, 0, w, h);

    const dur = Math.max(0.1, currentPiece.duration);
    const colorByRole: Record<TrackRole, string> = {
      vocal: 'rgba(242, 107, 122, 0.9)',
      chords: 'rgba(242, 184, 75, 0.58)',
      bass: 'rgba(217, 130, 59, 0.72)',
      lead: 'rgba(125, 180, 230, 0.78)',
    };

    for (const note of filteredNotes) {
      const x = (note.startTime / dur) * w;
      const nw = Math.max(1.8, (note.duration / dur) * w);
      const normY = 1 - Math.max(0, Math.min(1, (note.midi - 28) / 68));
      const y = 3 + normY * (h - 6);

      ctx.fillStyle = colorByRole[note.track];
      ctx.fillRect(x, y, nw, note.track === 'vocal' ? 2.4 : 1.7);
    }
  }, [currentPiece.duration, filteredNotes]);

  const progressPct =
    currentPiece.duration > 0 ? Math.min(100, (currentTime / currentPiece.duration) * 100) : 0;

  const formatRoleNotes = (midis: number[], limit: number) =>
    midis.length > 0 ? midis.slice(0, limit).map(midiToSolfege).join(', ') : '—';

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-20 flex flex-col items-center px-3 pb-4 sm:px-5 sm:pb-5">
      {/* Collapse / Expand Handle (Zen / Cinema Mode) */}
      <button
        type="button"
        onClick={onToggleCollapsed}
        aria-expanded={!collapsed}
        aria-label={collapsed ? 'Mostrar controles del piano' : 'Ocultar controles del piano'}
        className="pointer-events-auto mb-1.5 inline-flex items-center gap-1.5 rounded-t-lg border border-b-0 border-white/12 bg-[#0f1116]/95 px-3.5 py-1.5 text-xs font-medium text-[#cfc8bc] transition-colors hover:bg-[#191c24] hover:text-[#f8f3e8] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d4a353] whitespace-nowrap"
      >
        {collapsed ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
        <span>{collapsed ? 'Mostrar controles' : 'Ocultar controles'}</span>
      </button>

      {/* Unified Bottom Control Dock — solid/translucent surface without backdrop-blur */}
      <div
        className={`pointer-events-auto w-full max-w-[1080px] rounded-xl border border-white/12 bg-[#0f1116]/95 shadow-2xl transition-transform duration-200 ease-out ${
          collapsed ? 'translate-y-[120%] hidden' : 'translate-y-0 block'
        }`}
      >
        <div className="flex flex-col gap-2.5 p-3.5 sm:p-4">
          {/* Row 1: Primary Transport, 4-Track Piano Roll Scrubber, Upload MP3 & Export MIDI */}
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="flex items-center gap-1.5 shrink-0">
              <button
                type="button"
                onClick={onTogglePlay}
                disabled={isAnalyzing}
                aria-label={isPlaying ? 'Pausar reproducción' : 'Reproducir en el piano'}
                className="inline-flex h-10 items-center gap-2 rounded-lg bg-[#d4a353] px-4 text-sm font-semibold text-[#0e0f13] transition-colors hover:bg-[#e1b56a] disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f5e6c8] whitespace-nowrap shrink-0"
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                <span>{isPlaying ? 'Pausar' : 'Reproducir'}</span>
              </button>

              <button
                type="button"
                onClick={onRestart}
                disabled={isAnalyzing}
                aria-label="Reiniciar obra"
                className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-[#171a22] text-[#d9d2c5] transition-colors hover:bg-[#222632] hover:text-white disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d4a353]"
              >
                <RotateCcw className="h-4 w-4" />
              </button>

              <button
                type="button"
                onClick={onToggleMute}
                aria-label={isMuted ? 'Activar sonido' : 'Silenciar piano'}
                className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-[#171a22] text-[#d9d2c5] transition-colors hover:bg-[#222632] hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d4a353]"
              >
                {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
              </button>
            </div>

            {/* Interactive 4-Track Piano Roll Scrubber */}
            <div className="flex min-w-[200px] flex-1 items-center gap-2.5">
              <span
                data-testid="current-time"
                className="font-mono-tabular w-10 text-right text-xs text-[#cfc8bc]"
              >
                {formatTime(currentTime)}
              </span>

              <div className="relative flex h-9 flex-1 items-center overflow-hidden rounded-md border border-white/10 bg-[#14171f]">
                <canvas
                  ref={miniRollRef}
                  width={480}
                  height={36}
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0 h-full w-full"
                />
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-y-0 left-0 bg-white/8"
                  style={{ width: `${progressPct}%` }}
                />
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-y-0 w-[2px] bg-[#f5e6c8]"
                  style={{ left: `${progressPct}%` }}
                />
                <input
                  type="range"
                  min={0}
                  max={Math.max(0.1, currentPiece.duration)}
                  step={0.05}
                  value={Math.min(currentTime, currentPiece.duration)}
                  onChange={(e) => onSeek(parseFloat(e.target.value))}
                  aria-label="Línea de tiempo de la obra"
                  className="piano-scrubber relative z-10 h-full w-full opacity-0 hover:opacity-100 focus-visible:opacity-100"
                />
              </div>

              <span className="font-mono-tabular w-10 text-xs text-[#9c968b]">
                {formatTime(currentPiece.duration)}
              </span>
            </div>

            {/* Upload MP3 + Export Separated MIDI */}
            <div className="flex items-center gap-1.5 shrink-0">
              <button
                type="button"
                onClick={onUploadClick}
                disabled={isAnalyzing}
                aria-label="Subir archivo MP3 para calcular notas"
                className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#d4a353]/50 bg-[#1e1a14] px-3.5 text-sm font-medium text-[#f5dfb8] transition-colors hover:border-[#d4a353] hover:bg-[#2a2318] disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d4a353] whitespace-nowrap shrink-0"
              >
                <Upload className="h-4 w-4 text-[#d4a353]" />
                <span>Subir MP3</span>
              </button>

              <button
                type="button"
                onClick={onExportMidi}
                disabled={isAnalyzing || filteredNotes.length === 0}
                aria-label="Descargar archivo MIDI multipista separado por instrumentos"
                className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-white/12 bg-[#171a22] px-3 text-xs font-medium text-[#d9d2c5] transition-colors hover:bg-[#222632] hover:text-white disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d4a353] whitespace-nowrap shrink-0"
              >
                <Download className="h-3.5 w-3.5" />
                <span>MIDI</span>
              </button>
            </div>
          </div>

          <div className="h-px w-full bg-white/8" />

          {/* Row 2: Separated MIDI Instrument Mixer + Noise Cleanliness Filter */}
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 text-xs">
            {/* 4 Independent Separated Instrument Track Toggles */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[#9c968b] whitespace-nowrap">Pistas MIDI</span>
              <div
                role="group"
                aria-label="Separación de pistas por instrumento"
                className="flex items-center gap-1"
              >
                {INSTRUMENT_TRACKS.map((inst) => {
                  const active = enabledTracks[inst.role];
                  const count = filteredNotes.filter((n) => n.track === inst.role).length;
                  return (
                    <button
                      key={inst.role}
                      type="button"
                      onClick={() => onToggleTrack(inst.role)}
                      aria-pressed={active}
                      aria-label={`${active ? 'Silenciar' : 'Activar'} pista ${inst.label}`}
                      className={`inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                        active
                          ? 'border-white/18 bg-[#222632] text-[#f6f1e6]'
                          : 'border-white/8 bg-[#14171f] text-[#6f6a61] hover:text-[#a8a296]'
                      }`}
                    >
                      <span
                        aria-hidden="true"
                        className="inline-block h-2 w-2 rounded-xs transition-opacity"
                        style={{
                          backgroundColor: inst.color,
                          opacity: active ? 1 : 0.25,
                        }}
                      />
                      <span>{inst.label}</span>
                      <span className="font-mono-tabular text-[10px] text-[#8d877d]">{count}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Live Noise Cleanliness Filter (Suave / Media / Alta) */}
            <div className="flex items-center gap-1.5">
              <span className="text-[#9c968b] whitespace-nowrap">Limpieza</span>
              <div
                role="group"
                aria-label="Nivel de limpieza de ruido de notas"
                className="flex items-center rounded-lg border border-white/10 bg-[#14171f] p-0.5"
              >
                {(
                  [
                    { id: 'soft', label: 'Suave' },
                    { id: 'balanced', label: 'Media' },
                    { id: 'strict', label: 'Alta' },
                  ] as const
                ).map((lvl) => (
                  <button
                    key={lvl.id}
                    type="button"
                    onClick={() => onCleanlinessChange(lvl.id)}
                    aria-pressed={cleanliness === lvl.id}
                    className={`rounded-md px-2 py-1 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                      cleanliness === lvl.id
                        ? 'bg-[#262a36] text-[#f8f3e8]'
                        : 'text-[#9c968b] hover:text-[#e4ded3]'
                    }`}
                  >
                    {lvl.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Repertoire Selector */}
            <div className="flex items-center gap-1.5">
              <label htmlFor="piece-select" className="text-[#9c968b] whitespace-nowrap">
                Obra
              </label>
              <select
                id="piece-select"
                value={currentPiece.id}
                onChange={(e) => onSelectPiece(e.target.value)}
                className="h-7 max-w-[165px] truncate rounded-md border border-white/10 bg-[#171a22] px-2 text-xs font-medium text-[#f3efe7] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d4a353]"
              >
                {pieces.map((p) => (
                  <option key={p.id} value={p.id} className="bg-[#14171f] text-[#f3efe7]">
                    {p.source === 'upload' ? `${p.title} (MP3)` : p.title}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Row 3: Camera, Lid, Speed, Visual & Original MP3 + Quiet Live Notes */}
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 border-t border-white/6 pt-2 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <div
                role="group"
                aria-label="Vistas de cámara"
                className="flex items-center rounded-lg border border-white/10 bg-[#14171f] p-0.5"
              >
                {(
                  [
                    { id: 'concert', label: 'Concierto' },
                    { id: 'keyboard', label: 'Teclado' },
                    { id: 'harp', label: 'Arpa' },
                    { id: 'overhead', label: 'Superior' },
                  ] as const
                ).map((cam) => (
                  <button
                    key={cam.id}
                    type="button"
                    onClick={() => onCameraPresetChange(cam.id)}
                    aria-pressed={cameraPreset === cam.id}
                    className={`rounded-md px-2 py-0.5 text-[11px] font-medium transition-colors whitespace-nowrap shrink-0 ${
                      cameraPreset === cam.id
                        ? 'bg-[#262a36] text-[#f8f3e8]'
                        : 'text-[#9c968b] hover:text-[#e4ded3]'
                    }`}
                  >
                    {cam.label}
                  </button>
                ))}
              </div>

              <div
                role="group"
                aria-label="Apertura de tapa del piano"
                className="flex items-center rounded-lg border border-white/10 bg-[#14171f] p-0.5"
              >
                {(
                  [
                    { id: 'open', label: 'Abierta' },
                    { id: 'half', label: 'Media' },
                    { id: 'closed', label: 'Cerrada' },
                  ] as const
                ).map((lid) => (
                  <button
                    key={lid.id}
                    type="button"
                    onClick={() => onLidStateChange(lid.id)}
                    aria-pressed={lidState === lid.id}
                    className={`rounded-md px-2 py-0.5 text-[11px] font-medium transition-colors whitespace-nowrap shrink-0 ${
                      lidState === lid.id
                        ? 'bg-[#262a36] text-[#f8f3e8]'
                        : 'text-[#9c968b] hover:text-[#e4ded3]'
                    }`}
                  >
                    {lid.label}
                  </button>
                ))}
              </div>

              <button
                type="button"
                onClick={() => {
                  const next = playbackRate === 1 ? 0.75 : playbackRate === 0.75 ? 1.25 : 1;
                  onPlaybackRateChange(next);
                }}
                aria-label={`Velocidad actual ${playbackRate}x, cambiar velocidad`}
                className="font-mono-tabular h-6 rounded-md border border-white/10 bg-[#171a22] px-2 text-[11px] text-[#d9d2c5] transition-colors hover:bg-[#222632] whitespace-nowrap shrink-0"
              >
                {playbackRate}×
              </button>

              <button
                type="button"
                onClick={onToggleRibbons}
                aria-pressed={showRibbons}
                aria-label="Alternar cascada de notas 3D"
                className={`inline-flex h-6 items-center gap-1 rounded-md border px-2 text-[11px] font-medium transition-colors whitespace-nowrap shrink-0 ${
                  showRibbons
                    ? 'border-[#d4a353]/50 bg-[#251f15] text-[#f2d08b]'
                    : 'border-white/10 bg-[#171a22] text-[#9c968b] hover:text-[#e4ded3]'
                }`}
              >
                <Sparkles className="h-3 w-3" />
                <span>Cascada</span>
              </button>

              {hasOriginalAudio && (
                <button
                  type="button"
                  onClick={onToggleMixOriginalAudio}
                  aria-pressed={mixOriginalAudio}
                  aria-label="Mezclar con audio MP3 original"
                  className={`h-6 rounded-md border px-2 text-[11px] font-medium transition-colors whitespace-nowrap shrink-0 ${
                    mixOriginalAudio
                      ? 'border-[#f26b7a]/55 bg-[#2a171b] text-[#f7b1ba]'
                      : 'border-white/10 bg-[#171a22] text-[#9c968b] hover:text-[#e4ded3]'
                  }`}
                >
                  MP3 original
                </button>
              )}
            </div>

            {/* Quiet unboxed live notes readout */}
            <div className="flex flex-wrap items-center gap-2.5 text-[11px] text-[#9c968b]">
              <span>
                Voz:{' '}
                <span data-testid="active-vocal-notes" className="font-mono-tabular text-[#f4d8dc]">
                  {formatRoleNotes(activeNotesByRole.vocal, 2)}
                </span>
              </span>
              <span aria-hidden="true">·</span>
              <span>
                Acordes:{' '}
                <span data-testid="active-chords-notes" className="font-mono-tabular text-[#f2dfbc]">
                  {formatRoleNotes(activeNotesByRole.chords, 3)}
                </span>
              </span>
              <span aria-hidden="true">·</span>
              <span>
                Bajo:{' '}
                <span data-testid="active-bass-notes" className="font-mono-tabular text-[#eccbb0]">
                  {formatRoleNotes(activeNotesByRole.bass, 2)}
                </span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

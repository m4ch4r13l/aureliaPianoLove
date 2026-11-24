import { useEffect, useRef } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  Upload,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { formatTime, midiToSolfege } from '../audio/pianoTheory';
import { TranscribedPiece } from '../audio/pitchDsp';
import { CameraPresetId } from './GrandPianoViewport';
import { LidState } from '../three/grandPianoBuilder';

export type TrackFilterMode = 'both' | 'vocal' | 'piano';

interface ControlDockProps {
  pieces: TranscribedPiece[];
  currentPiece: TranscribedPiece;
  onSelectPiece: (pieceId: string) => void;
  onUploadClick: () => void;
  isAnalyzing: boolean;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onRestart: () => void;
  currentTime: number;
  onSeek: (timeSec: number) => void;
  trackFilter: TrackFilterMode;
  onTrackFilterChange: (mode: TrackFilterMode) => void;
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
  activeVocalMidis: number[];
  activePianoMidis: number[];
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

export function ControlDock({
  pieces,
  currentPiece,
  onSelectPiece,
  onUploadClick,
  isAnalyzing,
  isPlaying,
  onTogglePlay,
  onRestart,
  currentTime,
  onSeek,
  trackFilter,
  onTrackFilterChange,
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
  activeVocalMidis,
  activePianoMidis,
  collapsed,
  onToggleCollapsed,
}: ControlDockProps) {
  const miniRollRef = useRef<HTMLCanvasElement | null>(null);

  // Render crisp mini piano-roll overview inside the scrubber showing piano (gold) & vocal (rose) notes
  useEffect(() => {
    const canvas = miniRollRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Subtle baseline
    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.fillRect(0, 0, w, h);

    const dur = Math.max(0.1, currentPiece.duration);
    for (const note of currentPiece.notes) {
      if (trackFilter === 'vocal' && note.track !== 'vocal') continue;
      if (trackFilter === 'piano' && note.track !== 'piano') continue;

      const x = (note.startTime / dur) * w;
      const nw = Math.max(1.8, (note.duration / dur) * w);
      const normY = 1 - Math.max(0, Math.min(1, (note.midi - 28) / 68));
      const y = 3 + normY * (h - 6);

      ctx.fillStyle =
        note.track === 'vocal' ? 'rgba(242, 107, 122, 0.85)' : 'rgba(242, 184, 75, 0.52)';
      ctx.fillRect(x, y, nw, note.track === 'vocal' ? 2.4 : 1.7);
    }
  }, [currentPiece, trackFilter]);

  const progressPct =
    currentPiece.duration > 0 ? Math.min(100, (currentTime / currentPiece.duration) * 100) : 0;

  const vocalLabel =
    activeVocalMidis.length > 0
      ? activeVocalMidis.slice(0, 3).map(midiToSolfege).join(', ')
      : '—';
  const pianoLabel =
    activePianoMidis.length > 0
      ? activePianoMidis.slice(0, 4).map(midiToSolfege).join(', ')
      : '—';

  return (
    <div
      className="pointer-events-none fixed inset-x-0 bottom-0 z-20 flex flex-col items-center px-3 pb-4 sm:px-5 sm:pb-5"
    >
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
        className={`pointer-events-auto w-full max-w-[1060px] rounded-xl border border-white/12 bg-[#0f1116]/95 shadow-2xl transition-transform duration-200 ease-out ${
          collapsed ? 'translate-y-[120%] hidden' : 'translate-y-0 block'
        }`}
      >
        <div className="flex flex-col gap-3 p-3.5 sm:p-4">
          {/* Row 1: Primary Transport, Scrubber with Mini Piano Roll, and Upload MP3 CTA */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Play / Pause & Restart */}
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

            {/* Interactive Piano-Roll Timeline Scrubber */}
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

            {/* Upload MP3 Primary Action */}
            <button
              type="button"
              onClick={onUploadClick}
              disabled={isAnalyzing}
              aria-label="Subir archivo MP3 para calcular notas"
              className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#d4a353]/50 bg-[#1e1a14] px-4 text-sm font-medium text-[#f5dfb8] transition-colors hover:border-[#d4a353] hover:bg-[#2a2318] disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d4a353] whitespace-nowrap shrink-0"
            >
              <Upload className="h-4 w-4 text-[#d4a353]" />
              <span>Subir MP3</span>
            </button>
          </div>

          {/* Hairline divider */}
          <div className="h-px w-full bg-white/8" />

          {/* Row 2: Piece Selector, Track Layers, Camera Presets, Lid & Tempo */}
          <div className="flex flex-wrap items-center justify-between gap-x-2.5 gap-y-2 text-xs">
            {/* Repertoire / Uploaded Song selector */}
            <div className="flex items-center gap-1.5">
              <label htmlFor="piece-select" className="text-[#9c968b] whitespace-nowrap">
                Obra
              </label>
              <select
                id="piece-select"
                value={currentPiece.id}
                onChange={(e) => onSelectPiece(e.target.value)}
                className="h-8 max-w-[170px] truncate rounded-md border border-white/10 bg-[#171a22] px-2 text-xs font-medium text-[#f3efe7] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d4a353]"
              >
                {pieces.map((p) => (
                  <option key={p.id} value={p.id} className="bg-[#14171f] text-[#f3efe7]">
                    {p.source === 'upload' ? `${p.title} (MP3)` : p.title}
                  </option>
                ))}
              </select>
            </div>

            {/* Track Layer Segmented Filter: Piano + Voz / Solo Voz / Solo Piano */}
            <div
              role="group"
              aria-label="Capas de reproducción"
              className="flex items-center rounded-lg border border-white/10 bg-[#14171f] p-0.5"
            >
              {(
                [
                  { id: 'both', label: 'Piano + Voz' },
                  { id: 'vocal', label: 'Solo Voz' },
                  { id: 'piano', label: 'Solo Piano' },
                ] as const
              ).map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => onTrackFilterChange(opt.id)}
                  aria-pressed={trackFilter === opt.id}
                  className={`rounded-md px-2 py-1 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    trackFilter === opt.id
                      ? 'bg-[#262a36] text-[#f8f3e8]'
                      : 'text-[#9c968b] hover:text-[#e4ded3]'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            {/* Camera Presets */}
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
                  className={`rounded-md px-2 py-1 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    cameraPreset === cam.id
                      ? 'bg-[#262a36] text-[#f8f3e8]'
                      : 'text-[#9c968b] hover:text-[#e4ded3]'
                  }`}
                >
                  {cam.label}
                </button>
              ))}
            </div>

            {/* Grand Piano Lid Angle */}
            <div className="flex items-center gap-1.5">
              <span className="text-[#9c968b] whitespace-nowrap">Tapa</span>
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
                    className={`rounded-md px-2 py-1 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                      lidState === lid.id
                        ? 'bg-[#262a36] text-[#f8f3e8]'
                        : 'text-[#9c968b] hover:text-[#e4ded3]'
                    }`}
                  >
                    {lid.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Speed & Visual/Audio Toggles */}
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  const next = playbackRate === 1 ? 0.75 : playbackRate === 0.75 ? 1.25 : 1;
                  onPlaybackRateChange(next);
                }}
                aria-label={`Velocidad actual ${playbackRate}x, cambiar velocidad`}
                className="font-mono-tabular h-7 rounded-md border border-white/10 bg-[#171a22] px-2 text-xs text-[#d9d2c5] transition-colors hover:bg-[#222632] whitespace-nowrap shrink-0"
              >
                {playbackRate}×
              </button>

              <button
                type="button"
                onClick={onToggleRibbons}
                aria-pressed={showRibbons}
                aria-label="Alternar cascada de notas 3D"
                className={`inline-flex h-7 items-center gap-1 rounded-md border px-2 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
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
                  className={`h-7 rounded-md border px-2 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    mixOriginalAudio
                      ? 'border-[#f26b7a]/55 bg-[#2a171b] text-[#f7b1ba]'
                      : 'border-white/10 bg-[#171a22] text-[#9c968b] hover:text-[#e4ded3]'
                  }`}
                >
                  MP3 original
                </button>
              )}
            </div>
          </div>

          {/* Row 3: Quiet unboxed live notes & color legend (zero pills, zero pulsing dots) */}
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/6 pt-2 text-[11px] text-[#9c968b]">
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex items-center gap-1.5">
                <span
                  aria-hidden="true"
                  className="inline-block h-2 w-2 rounded-xs bg-[#f26b7a]"
                />
                <span>Melodía vocal:</span>
                <span
                  data-testid="active-vocal-notes"
                  className="font-mono-tabular text-[#f4d8dc]"
                >
                  {vocalLabel}
                </span>
              </span>
              <span aria-hidden="true">·</span>
              <span className="inline-flex items-center gap-1.5">
                <span
                  aria-hidden="true"
                  className="inline-block h-2 w-2 rounded-xs bg-[#f2b84b]"
                />
                <span>Armonía piano:</span>
                <span
                  data-testid="active-piano-notes"
                  className="font-mono-tabular text-[#f2dfbc]"
                >
                  {pianoLabel}
                </span>
              </span>
            </div>

            <span className="hidden text-[#7e786e] sm:inline">
              Haz clic en las teclas o usa tu teclado (Z–M, Q–P) · Arrastra un MP3 aquí
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

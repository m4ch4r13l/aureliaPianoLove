import { useEffect, useImperativeHandle, useRef, forwardRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { TrackRole } from '../audio/pitchDsp';
import { createConcertTextures } from '../three/proceduralTextures';
import { buildGrandPiano, GrandPianoRig, LidState } from '../three/grandPianoBuilder';
import { buildConcertStage } from '../three/concertStage';

export type CameraPresetId = 'concert' | 'keyboard' | 'harp' | 'overhead';

export interface PianoViewportHandle {
  setKeyActive: (midi: number, active: boolean, role: TrackRole | null, velocity?: number) => void;
  clearAllKeys: () => void;
  setSustainActive: (active: boolean) => void;
  requestFrame: () => void;
}

interface GrandPianoViewportProps {
  cameraPreset: CameraPresetId;
  lidState: LidState;
  showRibbons: boolean;
  isPlaying: boolean;
  onUserKeyPlay: (midi: number) => void;
}

const CAMERA_PRESETS: Record<
  CameraPresetId,
  { pos: [number, number, number]; target: [number, number, number] }
> = {
  concert: {
    pos: [2.35, 1.82, 2.65],
    target: [-0.08, 0.56, -0.22],
  },
  keyboard: {
    pos: [0.02, 1.38, 1.56],
    target: [0.0, 0.74, 0.56],
  },
  harp: {
    pos: [1.55, 1.92, 0.35],
    target: [-0.14, 0.88, -0.32],
  },
  overhead: {
    pos: [0.0, 3.45, 0.72],
    target: [0.0, 0.72, -0.22],
  },
};

function detectSoftwareWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param) ?? '');
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

export const GrandPianoViewport = forwardRef<PianoViewportHandle, GrandPianoViewportProps>(
  function GrandPianoViewport(
    { cameraPreset, lidState, showRibbons, isPlaying, onUserKeyPlay },
    ref,
  ) {
    const mountRef = useRef<HTMLDivElement | null>(null);
    const [webglUnavailable, setWebglUnavailable] = useState(false);

    // Store mutable React props in refs so WebGLRenderer mounts strictly once
    const lidStateRef = useRef<LidState>(lidState);
    const showRibbonsRef = useRef<boolean>(showRibbons);
    const isPlayingRef = useRef<boolean>(isPlaying);
    const onUserKeyPlayRef = useRef(onUserKeyPlay);

    const rigRef = useRef<GrandPianoRig | null>(null);
    const sustainRef = useRef<boolean>(false);
    const triggerRenderRef = useRef<(() => void) | null>(null);

    // Camera transition state
    const camTransitionRef = useRef<{
      active: boolean;
      goalPos: THREE.Vector3;
      goalTarget: THREE.Vector3;
    }>({
      active: false,
      goalPos: new THREE.Vector3(...CAMERA_PRESETS.concert.pos),
      goalTarget: new THREE.Vector3(...CAMERA_PRESETS.concert.target),
    });

    lidStateRef.current = lidState;
    showRibbonsRef.current = showRibbons;
    isPlayingRef.current = isPlaying;
    onUserKeyPlayRef.current = onUserKeyPlay;

    // Trigger smooth camera glide when cameraPreset changes
    useEffect(() => {
      const preset = CAMERA_PRESETS[cameraPreset];
      camTransitionRef.current.goalPos.set(...preset.pos);
      camTransitionRef.current.goalTarget.set(...preset.target);
      camTransitionRef.current.active = true;
      triggerRenderRef.current?.();
    }, [cameraPreset]);

    // Wake renderer when lid state, ribbons toggle, or playback starts
    useEffect(() => {
      triggerRenderRef.current?.();
    }, [lidState, showRibbons, isPlaying]);

    useImperativeHandle(
      ref,
      () => ({
        setKeyActive: (midi, active, role, velocity = 0.8) => {
          rigRef.current?.setKeyActive(midi, active, role, velocity);
          triggerRenderRef.current?.();
        },
        clearAllKeys: () => {
          const rig = rigRef.current;
          if (!rig) return;
          for (const midi of rig.keys.keys()) {
            rig.setKeyActive(midi, false, null, 0);
          }
          triggerRenderRef.current?.();
        },
        setSustainActive: (active) => {
          sustainRef.current = active;
          triggerRenderRef.current?.();
        },
        requestFrame: () => {
          triggerRenderRef.current?.();
        },
      }),
      [],
    );

    useEffect(() => {
      const container = mountRef.current;
      if (!container) return;

      const isSoftwareGl = detectSoftwareWebGL();
      let renderer: THREE.WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({
          antialias: !isSoftwareGl,
          powerPreference: 'high-performance',
        });
      } catch {
        setWebglUnavailable(true);
        return;
      }

      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.12;
      renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = !isSoftwareGl;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;

      const width = container.clientWidth || window.innerWidth;
      const height = container.clientHeight || window.innerHeight;
      renderer.setSize(width, height);
      renderer.domElement.style.display = 'block';
      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';
      container.innerHTML = '';
      container.appendChild(renderer.domElement);

      const scene = new THREE.Scene();

      // Mandatory procedural studio environment reflections on every device
      const pmremGenerator = new THREE.PMREMGenerator(renderer);
      const envRT = pmremGenerator.fromScene(new RoomEnvironment(), 0.04);
      scene.environment = envRT.texture;
      scene.environmentIntensity = 1.15;
      pmremGenerator.dispose();

      const camera = new THREE.PerspectiveCamera(42, width / height, 0.08, 45);
      camera.position.set(...CAMERA_PRESETS.concert.pos);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.target.set(...CAMERA_PRESETS.concert.target);
      controls.enableDamping = !isSoftwareGl;
      controls.dampingFactor = 0.065;
      controls.minDistance = 0.95;
      controls.maxDistance = 7.2;
      controls.minPolarAngle = 0.12;
      controls.maxPolarAngle = Math.PI * 0.485; // keep camera cleanly above concert floor
      controls.enablePan = true;
      controls.panSpeed = 0.6;
      controls.update();

      // Build procedural textures, recital hall stage, and hero concert grand piano
      const maxAniso = renderer.capabilities.getMaxAnisotropy();
      const textures = createConcertTextures(maxAniso);
      const stage = buildConcertStage(scene, textures, isSoftwareGl);
      const pianoRig = buildGrandPiano(textures, isSoftwareGl);
      scene.add(pianoRig.root);
      rigRef.current = pianoRig;

      if (!isSoftwareGl) {
        renderer.compile(scene, camera);
      }

      // Cancel scripted camera transitions immediately if user drags camera
      const onControlStart = () => {
        camTransitionRef.current.active = false;
      };
      controls.addEventListener('start', onControlStart);

      // Interactive raycasting on the 88 piano keys
      const raycaster = new THREE.Raycaster();
      const pointerNdc = new THREE.Vector2();
      let pointerDownPos = { x: 0, y: 0 };

      const onPointerDown = (e: PointerEvent) => {
        pointerDownPos = { x: e.clientX, y: e.clientY };
      };

      const onPointerUp = (e: PointerEvent) => {
        // Distinguish crisp click from camera orbit drag
        const dragDist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
        if (dragDist > 6) return;

        const rect = renderer.domElement.getBoundingClientRect();
        pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(pointerNdc, camera);
        const hits = raycaster.intersectObjects(pianoRig.clickableKeyMeshes, false);
        if (hits.length > 0) {
          const midi = hits[0].object.userData.midi as number | undefined;
          if (typeof midi === 'number') {
            onUserKeyPlayRef.current(midi);
          }
        }
      };

      renderer.domElement.addEventListener('pointerdown', onPointerDown);
      renderer.domElement.addEventListener('pointerup', onPointerUp);

      // Context loss resilience
      const onContextLost = (e: Event) => {
        e.preventDefault();
      };
      const onContextRestored = () => {
        requestRender();
      };
      renderer.domElement.addEventListener('webglcontextlost', onContextLost);
      renderer.domElement.addEventListener('webglcontextrestored', onContextRestored);

      // --- Coalesced On-Demand + Software-WebGL Paced Render Loop ---
      let disposed = false;
      let frameId = 0;
      let timerId = 0;
      let renderBusy = false;
      let renderDirty = false;
      let lastTimeMs = performance.now();
      let pendingResize: { w: number; h: number } | null = null;

      function afterDraw(cb: () => void) {
        if (disposed) return;
        const gl = renderer.getContext() as WebGL2RenderingContext;
        if (typeof gl.fenceSync === 'function') {
          const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
          gl.flush();
          const poll = () => {
            if (disposed) {
              if (sync) gl.deleteSync(sync);
              return;
            }
            const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
            if (state === gl.UNSIGNALED) {
              timerId = window.setTimeout(poll, 5);
            } else {
              if (sync) gl.deleteSync(sync);
              timerId = window.setTimeout(cb, 80);
            }
          };
          timerId = window.setTimeout(poll, 5);
        } else {
          timerId = window.setTimeout(cb, 80);
        }
      }

      function stepAndRender() {
        if (disposed) return;
        renderDirty = false;

        // Apply any deferred resize right before rendering so the drawing buffer is never left blank
        if (pendingResize) {
          const { w, h } = pendingResize;
          pendingResize = null;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }

        const nowMs = performance.now();
        const dt = Math.min(0.08, Math.max(0.001, (nowMs - lastTimeMs) / 1000));
        lastTimeMs = nowMs;

        let keepAnimating = false;

        // Smooth camera preset glide
        const camTr = camTransitionRef.current;
        if (camTr.active) {
          const factor = 1 - Math.exp(-5.5 * dt);
          camera.position.lerp(camTr.goalPos, factor);
          controls.target.lerp(camTr.goalTarget, factor);
          if (
            camera.position.distanceToSquared(camTr.goalPos) < 0.0004 &&
            controls.target.distanceToSquared(camTr.goalTarget) < 0.0004
          ) {
            camera.position.copy(camTr.goalPos);
            controls.target.copy(camTr.goalTarget);
            camTr.active = false;
          } else {
            keepAnimating = true;
          }
        }

        controls.update();

        // Advance piano keys, hammers, lid, sustain pedal, and rising note ribbons
        const pianoNeedsMore = pianoRig.update(
          dt,
          lidStateRef.current,
          sustainRef.current,
          showRibbonsRef.current,
        );
        if (pianoNeedsMore || isPlayingRef.current) {
          keepAnimating = true;
        }

        renderer.render(scene, camera);

        const done = () => {
          if (disposed) return;
          renderBusy = false;
          if (renderDirty || keepAnimating || pendingResize) {
            requestRender();
          }
        };

        if (isSoftwareGl) {
          afterDraw(done);
        } else {
          done();
        }
      }

      function requestRender() {
        if (disposed) return;
        if (renderBusy) {
          renderDirty = true;
          return;
        }
        renderBusy = true;
        frameId = requestAnimationFrame(stepAndRender);
      }

      triggerRenderRef.current = requestRender;

      const onControlsChange = () => {
        requestRender();
      };
      controls.addEventListener('change', onControlsChange);

      const onResize = () => {
        if (disposed || !container) return;
        const w = container.clientWidth || window.innerWidth;
        const h = container.clientHeight || window.innerHeight;
        pendingResize = { w, h };
        requestRender();
      };
      window.addEventListener('resize', onResize);

      // Paint initial frame immediately
      requestRender();

      return () => {
        disposed = true;
        triggerRenderRef.current = null;
        rigRef.current = null;
        cancelAnimationFrame(frameId);
        window.clearTimeout(timerId);
        window.removeEventListener('resize', onResize);
        controls.removeEventListener('start', onControlStart);
        controls.removeEventListener('change', onControlsChange);
        controls.dispose();
        renderer.domElement.removeEventListener('pointerdown', onPointerDown);
        renderer.domElement.removeEventListener('pointerup', onPointerUp);
        renderer.domElement.removeEventListener('webglcontextlost', onContextLost);
        renderer.domElement.removeEventListener('webglcontextrestored', onContextRestored);
        pianoRig.dispose();
        stage.dispose();
        textures.dispose();
        envRT.dispose();
        renderer.dispose();
      };
    }, []);

    if (webglUnavailable) {
      return (
        <div
          className="flex h-full w-full items-center justify-center bg-[#0c0e13] p-8 text-center"
          role="alert"
        >
          <div className="max-w-md space-y-2">
            <h2 className="font-display text-2xl text-[#f3efe7]">Escena 3D no disponible</h2>
            <p className="text-sm text-[#a6a095]">
              Tu navegador no pudo inicializar WebGL para mostrar el piano de cola 3D.
            </p>
          </div>
        </div>
      );
    }

    return (
      <div
        ref={mountRef}
        data-testid="piano-webgl-canvas"
        aria-label="Escena 3D interactiva del piano de cola"
        className="h-full w-full cursor-grab active:cursor-grabbing"
      />
    );
  },
);

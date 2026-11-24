import * as THREE from 'three';
import { build88KeyLayout, MAX_MIDI, MIN_MIDI } from '../audio/pianoTheory';
import { TrackRole } from '../audio/pitchDsp';
import { ProceduralTextureSet } from './proceduralTextures';

export type LidState = 'open' | 'half' | 'closed';

export interface AnimatedKeyHandle {
  midi: number;
  isBlack: boolean;
  pivot: THREE.Group;
  keyMesh: THREE.Mesh;
  material: THREE.MeshStandardMaterial;
  hammerPivot: THREE.Group;
  baseColor: THREE.Color;
  keyX: number;
  keyTopY: number;
  keyFrontZ: number;
  /** Current normalized press depth (0..1) */
  press: number;
  targetPress: number;
  /** Hammer kick impulse (0..1) */
  hammerKick: number;
  activeRole: TrackRole | null;
}

export interface GrandPianoRig {
  root: THREE.Group;
  keys: Map<number, AnimatedKeyHandle>;
  clickableKeyMeshes: THREE.Mesh[];
  lidPivot: THREE.Group;
  propStick: THREE.Mesh;
  sustainPedalPivot: THREE.Group;
  ribbonGroup: THREE.Group;
  /** Updates key press kinematics, hammers, lid angle, pedal, and rising 3D note ribbons */
  update: (
    dt: number,
    lidState: LidState,
    sustainActive: boolean,
    showRibbons: boolean,
  ) => boolean;
  /** Triggers a visual key depression and hammer strike */
  setKeyActive: (midi: number, active: boolean, role: TrackRole | null, velocity?: number) => void;
  dispose: () => void;
}

/**
 * Creates the iconic S-curve silhouette of a concert grand piano body in XZ space.
 * Keyboard spans along X in [-0.74, +0.74] at Z = +0.58 (front).
 * The long bass spine runs along the left side (X = -0.74) back to Z = -1.68 (tail).
 * The treble bentside curves inward gracefully on the right (+X) toward the rounded tail.
 */
function createGrandBodyShape(inset = 0): THREE.Shape {
  const s = new THREE.Shape();
  const leftX = -0.755 + inset;
  const rightX = 0.755 - inset;
  const frontZ = 0.50 - inset;
  const trebleCheekZ = -0.04;
  const tailZ = -1.68 + inset;

  // Note: ExtrudeGeometry extrudes along +Z from XY shape.
  // We map shape (x, y) -> world (x, -z) after rotateX(-Math.PI / 2), so y = -z!
  const yFront = -frontZ;
  const yCheek = -trebleCheekZ;
  const yTail = -tailZ;

  s.moveTo(leftX, yFront);
  s.lineTo(rightX, yFront);
  // Short right treble cheek going straight back
  s.lineTo(rightX, yCheek);
  // Iconic grand piano S-curve bentside curving inward to the tail
  s.bezierCurveTo(
    rightX - 0.01,
    yCheek + 0.38,
    0.26 - inset * 0.5,
    yCheek + 0.52,
    0.16 - inset * 0.4,
    yTail - 0.48,
  );
  // Rounded concert tail curve sweeping smoothly to the bass spine
  s.bezierCurveTo(
    0.10 - inset * 0.3,
    yTail + 0.04,
    -0.58 + inset * 0.5,
    yTail + 0.06,
    leftX,
    yTail - 0.28,
  );
  // Long straight bass spine back to front-left cheek
  s.lineTo(leftX, yFront);

  return s;
}

/**
 * Creates an authentic fluted tapered concert grand leg with brass collar via LatheGeometry.
 */
function createTurnedLegGeometry(height: number): THREE.LatheGeometry {
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(0.018, 0.0),
    new THREE.Vector2(0.024, 0.04),
    new THREE.Vector2(0.022, 0.08),
    new THREE.Vector2(0.032, 0.16),
    new THREE.Vector2(0.046, height * 0.65),
    new THREE.Vector2(0.056, height * 0.86),
    new THREE.Vector2(0.068, height * 0.93),
    new THREE.Vector2(0.068, height),
  ];
  return new THREE.LatheGeometry(pts, 20);
}

interface RibbonParticle {
  mesh: THREE.Mesh;
  material: THREE.MeshStandardMaterial;
  age: number;
  maxAge: number;
  speed: number;
  baseScaleY: number;
}

export function buildGrandPiano(
  textures: ProceduralTextureSet,
  isSoftwareGl: boolean,
): GrandPianoRig {
  const root = new THREE.Group();
  root.name = 'ConcertGrandPiano';

  const disposables: Array<{ dispose: () => void }> = [];
  const trackDisposable = <T extends { dispose: () => void }>(item: T): T => {
    disposables.push(item);
    return item;
  };

  // --- Shared Luxury Concert Materials ---
  // High-gloss concert ebony lacquer (clearcoat reflections from RoomEnvironment)
  const ebonyLacquerMat = trackDisposable(
    new THREE.MeshPhysicalMaterial({
      color: 0x0e0f13,
      roughness: 0.13,
      metalness: 0.08,
      clearcoat: 1.0,
      clearcoatRoughness: 0.06,
      reflectivity: 0.95,
    }),
  );

  // Satin matte ebony (inner rim trim, keyblocks, fallboard recess, bench wood)
  const ebonySatinMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0x16171d,
      roughness: 0.48,
      metalness: 0.06,
    }),
  );

  // Cast-iron gilded concert harp bronze-gold (high metalness + anisotropy)
  const harpGoldMat = trackDisposable(
    new THREE.MeshPhysicalMaterial({
      color: 0xc89a45,
      roughness: 0.28,
      metalness: 0.86,
      anisotropy: isSoftwareGl ? 0 : 0.65,
      clearcoat: 0.35,
      clearcoatRoughness: 0.25,
    }),
  );

  // Polished solid brass (pedals, hinges, casters, prop cup, tuning pin trim)
  const polishedBrassMat = trackDisposable(
    new THREE.MeshPhysicalMaterial({
      color: 0xe3b963,
      roughness: 0.16,
      metalness: 0.94,
      anisotropy: isSoftwareGl ? 0 : 0.75,
    }),
  );

  // Crimson concert felt (under keys, damper rail, hitch-pin braid)
  const crimsonFeltMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0x6f191e,
      roughness: 0.92,
      metalness: 0.0,
    }),
  );

  // Spruce soundboard inside the rim
  const soundboardMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: textures.soundboardColor,
      roughness: 0.42,
      metalness: 0.02,
    }),
  );

  // Piano string steel & copper-wound bass strings
  const steelStringMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0xd4d9e2,
      roughness: 0.24,
      metalness: 0.92,
    }),
  );
  const copperStringMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0xb9723c,
      roughness: 0.3,
      metalness: 0.88,
    }),
  );

  // White hammer felt & mahogany shank
  const hammerFeltMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0xf1ebe0,
      roughness: 0.88,
      metalness: 0.0,
    }),
  );
  const hammerWoodMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0x8e5d38,
      roughness: 0.56,
      metalness: 0.04,
    }),
  );

  // Leather concert bench cushion
  const benchLeatherMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0x1a191d,
      roughness: 0.58,
      metalness: 0.06,
    }),
  );

  // Sheet music page material
  const sheetMusicMat = trackDisposable(
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: textures.sheetMusic,
      roughness: 0.78,
      metalness: 0.0,
    }),
  );

  // --- Key Elevation Coordinates (grounded cleanly at y = 0) ---
  const bodyBottomY = 0.68;
  const rimHeight = 0.30;
  const rimTopY = bodyBottomY + rimHeight; // 0.98 m
  const keybedY = 0.725;

  // =========================================================================
  // 1. CURVED GRAND PIANO BODY (HOLLOW RIM + LOWER FLOOR + SOUNDBOARD)
  // =========================================================================
  const outerShape = createGrandBodyShape(0);
  const innerHole = createGrandBodyShape(0.055);
  outerShape.holes.push(innerHole);

  const rimGeo = trackDisposable(
    new THREE.ExtrudeGeometry(outerShape, {
      depth: rimHeight,
      bevelEnabled: true,
      bevelThickness: 0.012,
      bevelSize: 0.008,
      bevelSegments: 3,
      curveSegments: 36,
    }),
  );
  rimGeo.rotateX(-Math.PI / 2);
  const rimMesh = new THREE.Mesh(rimGeo, ebonyLacquerMat);
  rimMesh.position.y = bodyBottomY;
  rimMesh.castShadow = true;
  rimMesh.receiveShadow = true;
  root.add(rimMesh);

  // Solid underside floor pan beneath soundboard
  const basePlateShape = createGrandBodyShape(0.012);
  const basePlateGeo = trackDisposable(
    new THREE.ExtrudeGeometry(basePlateShape, {
      depth: 0.065,
      bevelEnabled: false,
      curveSegments: 32,
    }),
  );
  basePlateGeo.rotateX(-Math.PI / 2);
  const basePlateMesh = new THREE.Mesh(basePlateGeo, ebonySatinMat);
  basePlateMesh.position.y = bodyBottomY;
  basePlateMesh.castShadow = true;
  basePlateMesh.receiveShadow = true;
  root.add(basePlateMesh);

  // Warm Sitka Spruce Soundboard sitting inside the rim cavity
  const sbShape = createGrandBodyShape(0.058);
  const sbGeo = trackDisposable(
    new THREE.ExtrudeGeometry(sbShape, {
      depth: 0.025,
      bevelEnabled: false,
      curveSegments: 28,
    }),
  );
  sbGeo.rotateX(-Math.PI / 2);
  const soundboardMesh = new THREE.Mesh(sbGeo, soundboardMat);
  soundboardMesh.position.y = bodyBottomY + 0.095;
  soundboardMesh.receiveShadow = true;
  root.add(soundboardMesh);

  // =========================================================================
  // 2. CAST-IRON GOLDEN HARP, STRUCTURAL RIBS, TUNING PINS & STRINGS
  // =========================================================================
  const harpGroup = new THREE.Group();
  harpGroup.position.y = bodyBottomY + 0.14;
  root.add(harpGroup);

  // Curved gilded perimeter cast-iron plate inside the rear & sides
  const harpOuter = createGrandBodyShape(0.072);
  const harpHole = createGrandBodyShape(0.165);
  harpOuter.holes.push(harpHole);
  const harpGeo = trackDisposable(
    new THREE.ExtrudeGeometry(harpOuter, {
      depth: 0.024,
      bevelEnabled: true,
      bevelThickness: 0.005,
      bevelSize: 0.005,
      bevelSegments: 2,
      curveSegments: 28,
    }),
  );
  harpGeo.rotateX(-Math.PI / 2);
  const harpMesh = new THREE.Mesh(harpGeo, harpGoldMat);
  harpMesh.receiveShadow = true;
  harpGroup.add(harpMesh);

  // Structural cast-iron fan struts (braces across the soundboard)
  const strutSpecs = [
    { start: [-0.56, 0.25], end: [-0.46, -1.42] },
    { start: [-0.20, 0.25], end: [-0.26, -1.48] },
    { start: [0.18, 0.25], end: [-0.04, -1.36] },
    { start: [0.52, 0.25], end: [0.16, -0.78] },
  ];
  const strutGeo = trackDisposable(new THREE.BoxGeometry(0.036, 0.034, 1.0));
  for (const st of strutSpecs) {
    const dx = st.end[0] - st.start[0];
    const dz = st.end[1] - st.start[1];
    const len = Math.hypot(dx, dz);
    const strut = new THREE.Mesh(strutGeo, harpGoldMat);
    strut.scale.set(1, 1, len);
    strut.position.set((st.start[0] + st.end[0]) * 0.5, 0.026, (st.start[1] + st.end[1]) * 0.5);
    strut.rotation.y = Math.atan2(dx, dz);
    harpGroup.add(strut);
  }

  // Front pinblock plate (clavijero)
  const pinBlockGeo = trackDisposable(new THREE.BoxGeometry(1.36, 0.028, 0.22));
  const pinBlock = new THREE.Mesh(pinBlockGeo, harpGoldMat);
  pinBlock.position.set(0, 0.02, 0.31);
  harpGroup.add(pinBlock);

  // Crimson hitch-pin felt strip
  const hitchFeltGeo = trackDisposable(new THREE.BoxGeometry(1.32, 0.012, 0.026));
  const hitchFelt = new THREE.Mesh(hitchFeltGeo, crimsonFeltMat);
  hitchFelt.position.set(0, 0.036, 0.19);
  harpGroup.add(hitchFelt);

  // Instanced tuning pins (3 rows x 38 pins = 114 metallic pins)
  const pinCount = 114;
  const pinGeo = trackDisposable(new THREE.CylinderGeometry(0.0042, 0.0042, 0.032, 8));
  const pinInst = new THREE.InstancedMesh(pinGeo, steelStringMat, pinCount);
  const dummy = new THREE.Object3D();
  let pIdx = 0;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 38; col++) {
      const px = -0.63 + (col / 37) * 1.26;
      const pz = 0.25 + row * 0.045;
      dummy.position.set(px, 0.044, pz);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      pinInst.setMatrixAt(pIdx++, dummy.matrix);
    }
  }
  pinInst.instanceMatrix.needsUpdate = true;
  pinInst.computeBoundingSphere();
  harpGroup.add(pinInst);

  // Instanced steel treble/mid strings (54 strings fanning toward the tail)
  const trebleStringCount = 54;
  const stringUnitGeo = trackDisposable(new THREE.CylinderGeometry(0.0014, 0.0014, 1.0, 5));
  stringUnitGeo.rotateX(Math.PI / 2); // align unit length along Z
  const trebleStrings = new THREE.InstancedMesh(stringUnitGeo, steelStringMat, trebleStringCount);
  for (let i = 0; i < trebleStringCount; i++) {
    const t = i / (trebleStringCount - 1);
    const sx = -0.34 + t * 0.96;
    const sz = 0.22;
    const ex = -0.58 + t * 0.84;
    // Match inner bentside curve length naturally from long mid to short treble
    const ez = -1.48 + Math.pow(t, 0.72) * 1.18;
    const dx = ex - sx;
    const dz = ez - sz;
    const len = Math.hypot(dx, dz);

    dummy.position.set((sx + ex) * 0.5, 0.046, (sz + ez) * 0.5);
    dummy.rotation.set(0, Math.atan2(dx, dz), 0);
    dummy.scale.set(1, 1, len);
    dummy.updateMatrix();
    trebleStrings.setMatrixAt(i, dummy.matrix);
  }
  trebleStrings.instanceMatrix.needsUpdate = true;
  trebleStrings.computeBoundingSphere();
  harpGroup.add(trebleStrings);

  // Overstrung copper-wound bass strings (22 thicker diagonal strings crossing over mid strings)
  const bassStringCount = 22;
  const bassStringGeo = trackDisposable(new THREE.CylinderGeometry(0.0024, 0.0024, 1.0, 6));
  bassStringGeo.rotateX(Math.PI / 2);
  const bassStrings = new THREE.InstancedMesh(bassStringGeo, copperStringMat, bassStringCount);
  for (let i = 0; i < bassStringCount; i++) {
    const t = i / (bassStringCount - 1);
    const sx = -0.63 + t * 0.28;
    const sz = 0.22;
    const ex = -0.22 + t * 0.22;
    const ez = -1.52 + t * 0.26;
    const dx = ex - sx;
    const dz = ez - sz;
    const len = Math.hypot(dx, dz);

    dummy.position.set((sx + ex) * 0.5, 0.062, (sz + ez) * 0.5);
    dummy.rotation.set(0, Math.atan2(dx, dz), 0);
    dummy.scale.set(1, 1, len);
    dummy.updateMatrix();
    bassStrings.setMatrixAt(i, dummy.matrix);
  }
  bassStrings.instanceMatrix.needsUpdate = true;
  bassStrings.computeBoundingSphere();
  harpGroup.add(bassStrings);

  // =========================================================================
  // 3. KEYBED, CHEEK BLOCKS, FALLBOARD & MUSIC DESK WITH SHEET MUSIC
  // =========================================================================
  const keyLayout = build88KeyLayout();
  const keybedWidth = keyLayout.totalWidth + 0.145; // ~1.37 m

  // Keybed shelf extending forward to Z = +0.76
  const keybedGeo = trackDisposable(new THREE.BoxGeometry(keybedWidth, 0.056, 0.34));
  const keybedMesh = new THREE.Mesh(keybedGeo, ebonyLacquerMat);
  keybedMesh.position.set(0, keybedY - 0.028, 0.61);
  keybedMesh.castShadow = true;
  keybedMesh.receiveShadow = true;
  root.add(keybedMesh);

  // Left and right sculpted cheek blocks flanking the 88 keys
  const cheekGeo = trackDisposable(new THREE.BoxGeometry(0.066, 0.115, 0.32));
  const leftCheek = new THREE.Mesh(cheekGeo, ebonyLacquerMat);
  leftCheek.position.set(-keybedWidth / 2 + 0.033, keybedY + 0.052, 0.60);
  leftCheek.castShadow = true;
  root.add(leftCheek);

  const rightCheek = new THREE.Mesh(cheekGeo, ebonyLacquerMat);
  rightCheek.position.set(keybedWidth / 2 - 0.033, keybedY + 0.052, 0.60);
  rightCheek.castShadow = true;
  root.add(rightCheek);

  // Front keyslip rail right in front of the white keys
  const keyslipGeo = trackDisposable(new THREE.BoxGeometry(keybedWidth - 0.02, 0.038, 0.018));
  const keyslip = new THREE.Mesh(keyslipGeo, ebonyLacquerMat);
  keyslip.position.set(0, keybedY + 0.014, 0.772);
  root.add(keyslip);

  // Open fallboard (nameboard) behind the keys with crimson felt strip & gold brand plaque
  const fallboardGeo = trackDisposable(new THREE.BoxGeometry(keybedWidth - 0.12, 0.155, 0.032));
  const fallboard = new THREE.Mesh(fallboardGeo, ebonyLacquerMat);
  fallboard.position.set(0, keybedY + 0.088, 0.505);
  fallboard.castShadow = true;
  root.add(fallboard);

  // Subtle brass inlay strip along the nameboard
  const inlayGeo = trackDisposable(new THREE.BoxGeometry(0.24, 0.008, 0.004));
  const inlayMesh = new THREE.Mesh(inlayGeo, polishedBrassMat);
  inlayMesh.position.set(0, keybedY + 0.105, 0.522);
  root.add(inlayMesh);

  // Crimson key-back felt strip directly behind the keys
  const backFeltGeo = trackDisposable(new THREE.BoxGeometry(keyLayout.totalWidth + 0.01, 0.014, 0.012));
  const backFelt = new THREE.Mesh(backFeltGeo, crimsonFeltMat);
  backFelt.position.set(0, keybedY + 0.032, 0.538);
  root.add(backFelt);

  // Music Desk (Atril) tilted cleanly behind the fallboard
  const deskGroup = new THREE.Group();
  deskGroup.position.set(0, rimTopY + 0.005, 0.42);
  root.add(deskGroup);

  const deskLedgeGeo = trackDisposable(new THREE.BoxGeometry(0.84, 0.016, 0.068));
  const deskLedge = new THREE.Mesh(deskLedgeGeo, ebonyLacquerMat);
  deskLedge.position.set(0, 0.012, 0.028);
  deskGroup.add(deskLedge);

  const deskBackGeo = trackDisposable(new THREE.BoxGeometry(0.82, 0.24, 0.012));
  const deskBack = new THREE.Mesh(deskBackGeo, ebonySatinMat);
  deskBack.position.set(0, 0.128, -0.025);
  deskBack.rotation.x = -0.24;
  deskBack.castShadow = true;
  deskGroup.add(deskBack);

  // Brass music desk trim bar
  const deskTrimGeo = trackDisposable(new THREE.BoxGeometry(0.82, 0.009, 0.016));
  const deskTrim = new THREE.Mesh(deskTrimGeo, polishedBrassMat);
  deskTrim.position.set(0, 0.244, -0.052);
  deskGroup.add(deskTrim);

  // Open sheet music resting on the desk
  const sheetGeo = trackDisposable(new THREE.PlaneGeometry(0.46, 0.28));
  const sheetMesh = new THREE.Mesh(sheetGeo, sheetMusicMat);
  sheetMesh.position.set(0, 0.145, -0.014);
  sheetMesh.rotation.x = -0.24;
  deskGroup.add(sheetMesh);

  // =========================================================================
  // 4. 88 INTERACTIVE KEYS + 88 INTERNAL FELT HAMMERS
  // =========================================================================
  const keys = new Map<number, AnimatedKeyHandle>();
  const clickableKeyMeshes: THREE.Mesh[] = [];

  const whiteKeyGeo = trackDisposable(
    new THREE.BoxGeometry(keyLayout.whiteKeyWidth, 0.022, 0.152),
  );
  const blackKeyGeo = trackDisposable(
    new THREE.BoxGeometry(0.0132, 0.016, 0.094),
  );
  const hammerShankGeo = trackDisposable(new THREE.BoxGeometry(0.0045, 0.005, 0.11));
  const hammerHeadGeo = trackDisposable(new THREE.BoxGeometry(0.009, 0.016, 0.022));

  const whiteBaseColor = new THREE.Color(0xf4efe4);
  const blackBaseColor = new THREE.Color(0x16171d);

  const keyPivotZ = 0.548; // Rear fulcrum line inside the fallboard
  const whiteTopY = keybedY + 0.036;
  const blackTopY = whiteTopY + 0.014;

  for (const k of keyLayout.keys) {
    const pivot = new THREE.Group();
    pivot.position.set(k.x, k.isBlack ? blackTopY - 0.008 : whiteTopY - 0.011, keyPivotZ);
    root.add(pivot);

    const keyMat = trackDisposable(
      new THREE.MeshStandardMaterial({
        color: k.isBlack ? blackBaseColor.clone() : whiteBaseColor.clone(),
        roughness: k.isBlack ? 0.34 : 0.18,
        metalness: 0.04,
        emissive: new THREE.Color(0x000000),
        emissiveIntensity: 0,
      }),
    );

    const mesh = new THREE.Mesh(k.isBlack ? blackKeyGeo : whiteKeyGeo, keyMat);
    const zLength = k.isBlack ? 0.094 : 0.152;
    // Offset forward from rear fulcrum so rotating +X tips the front keytop downward
    mesh.position.set(0, 0, zLength * 0.5 + (k.isBlack ? 0.002 : 0.062));
    mesh.castShadow = !isSoftwareGl;
    mesh.receiveShadow = true;
    mesh.userData.midi = k.midi;
    pivot.add(mesh);
    clickableKeyMeshes.push(mesh);

    // Internal felt hammer visible inside the open piano behind the pinblock
    const hammerPivot = new THREE.Group();
    hammerPivot.position.set(k.x * 0.96, bodyBottomY + 0.145, 0.14);
    root.add(hammerPivot);

    const shank = new THREE.Mesh(hammerShankGeo, hammerWoodMat);
    shank.position.set(0, 0, -0.055);
    hammerPivot.add(shank);

    const head = new THREE.Mesh(hammerHeadGeo, hammerFeltMat);
    head.position.set(0, 0.009, -0.105);
    hammerPivot.add(head);

    keys.set(k.midi, {
      midi: k.midi,
      isBlack: k.isBlack,
      pivot,
      keyMesh: mesh,
      material: keyMat,
      hammerPivot,
      baseColor: k.isBlack ? blackBaseColor : whiteBaseColor,
      keyX: k.x,
      keyTopY: k.isBlack ? blackTopY : whiteTopY,
      keyFrontZ: keyPivotZ + zLength + (k.isBlack ? 0.002 : 0.062),
      press: 0,
      targetPress: 0,
      hammerKick: 0,
      activeRole: null,
    });
  }

  // =========================================================================
  // 5. HINGED CONCERT LID, BRASS HINGES & ARTICULATED PROP STICK
  // =========================================================================
  const lidPivot = new THREE.Group();
  // Hinge sits along the straight left bass spine at X = -0.755, top of rim
  lidPivot.position.set(-0.755, rimTopY + 0.012, 0);
  root.add(lidPivot);

  const lidShape = createGrandBodyShape(-0.006);
  const lidGeo = trackDisposable(
    new THREE.ExtrudeGeometry(lidShape, {
      depth: 0.024,
      bevelEnabled: true,
      bevelThickness: 0.006,
      bevelSize: 0.005,
      bevelSegments: 2,
      curveSegments: 36,
    }),
  );
  lidGeo.rotateX(-Math.PI / 2);
  // Shift lid geometry so its left spine aligns with lidPivot origin (x = 0)
  lidGeo.translate(0.755, 0, 0);

  const lidMesh = new THREE.Mesh(lidGeo, ebonyLacquerMat);
  lidMesh.castShadow = true;
  lidMesh.receiveShadow = true;
  lidPivot.add(lidMesh);

  // 3 Brass spine hinges along left rim
  const hingeGeo = trackDisposable(new THREE.CylinderGeometry(0.011, 0.011, 0.075, 10));
  hingeGeo.rotateX(Math.PI / 2);
  for (const hz of [0.22, -0.45, -1.16]) {
    const hinge = new THREE.Mesh(hingeGeo, polishedBrassMat);
    hinge.position.set(-0.758, rimTopY + 0.012, hz);
    root.add(hinge);
  }

  // Concert lid prop stick (pivots from right treble rim up into the underside of the lid)
  const propStickLen = 0.84;
  const propStickGeo = trackDisposable(new THREE.CylinderGeometry(0.009, 0.012, propStickLen, 12));
  propStickGeo.translate(0, propStickLen * 0.5, 0);
  const propStick = new THREE.Mesh(propStickGeo, ebonySatinMat);
  propStick.position.set(0.68, rimTopY - 0.02, 0.06);
  propStick.castShadow = true;
  root.add(propStick);

  const propCupGeo = trackDisposable(new THREE.CylinderGeometry(0.017, 0.014, 0.022, 12));
  const propCup = new THREE.Mesh(propCupGeo, polishedBrassMat);
  propCup.position.set(0.68, rimTopY - 0.01, 0.06);
  root.add(propCup);

  // =========================================================================
  // 6. THREE TURNED CONCERT LEGS WITH BRASS CASTERS (GROUNDED AT Y = 0)
  // =========================================================================
  const legHeight = bodyBottomY - 0.055; // 0.625 m above 0.055 m brass caster assembly
  const legGeo = trackDisposable(createTurnedLegGeometry(legHeight));
  const casterCollarGeo = trackDisposable(new THREE.CylinderGeometry(0.024, 0.021, 0.03, 14));
  const casterWheelGeo = trackDisposable(new THREE.CylinderGeometry(0.021, 0.021, 0.014, 14));
  casterWheelGeo.rotateZ(Math.PI / 2);

  const legPositions: Array<[number, number]> = [
    [-0.60, 0.44],  // Front-left bass leg
    [0.60, 0.44],   // Front-right treble leg
    [-0.30, -1.36], // Rear concert tail leg
  ];

  for (const [lx, lz] of legPositions) {
    const legGroup = new THREE.Group();
    legGroup.position.set(lx, 0, lz);
    root.add(legGroup);

    const legMesh = new THREE.Mesh(legGeo, ebonyLacquerMat);
    legMesh.position.y = 0.055;
    legMesh.castShadow = true;
    legMesh.receiveShadow = true;
    legGroup.add(legMesh);

    const collar = new THREE.Mesh(casterCollarGeo, polishedBrassMat);
    collar.position.y = 0.05;
    legGroup.add(collar);

    const wheel = new THREE.Mesh(casterWheelGeo, polishedBrassMat);
    wheel.position.set(0, 0.021, 0.008);
    legGroup.add(wheel);
  }

  // =========================================================================
  // 7. PEDAL LYRE & 3 POLISHED BRASS PEDALS
  // =========================================================================
  const lyreGroup = new THREE.Group();
  lyreGroup.position.set(0, 0, 0.20);
  root.add(lyreGroup);

  // Pedal box sitting low near the floor
  const pedalBoxGeo = trackDisposable(new THREE.BoxGeometry(0.30, 0.075, 0.11));
  const pedalBox = new THREE.Mesh(pedalBoxGeo, ebonyLacquerMat);
  pedalBox.position.set(0, 0.085, 0);
  pedalBox.castShadow = true;
  lyreGroup.add(pedalBox);

  // Two vertical ebony lyre columns connecting pedal box up to underside of piano
  const lyrePostGeo = trackDisposable(new THREE.CylinderGeometry(0.016, 0.02, bodyBottomY - 0.12, 12));
  for (const px of [-0.11, 0.11]) {
    const post = new THREE.Mesh(lyrePostGeo, ebonyLacquerMat);
    post.position.set(px, 0.12 + (bodyBottomY - 0.12) * 0.5, 0);
    post.castShadow = true;
    lyreGroup.add(post);
  }

  // Diagonal rear brace rods supporting the lyre against foot pressure
  const braceGeo = trackDisposable(new THREE.CylinderGeometry(0.008, 0.008, 0.62, 8));
  for (const px of [-0.08, 0.08]) {
    const brace = new THREE.Mesh(braceGeo, ebonySatinMat);
    brace.position.set(px, 0.36, -0.22);
    brace.rotation.x = 0.66;
    lyreGroup.add(brace);
  }

  // Three thin vertical brass actuator rods inside the lyre
  const rodGeo = trackDisposable(new THREE.CylinderGeometry(0.0035, 0.0035, bodyBottomY - 0.12, 8));
  for (const px of [-0.055, 0, 0.055]) {
    const rod = new THREE.Mesh(rodGeo, polishedBrassMat);
    rod.position.set(px, 0.12 + (bodyBottomY - 0.12) * 0.5, 0.01);
    lyreGroup.add(rod);
  }

  // Three brass pedals (Una Corda, Sostenuto, Damper/Sustain)
  const pedalGeo = trackDisposable(new THREE.BoxGeometry(0.032, 0.014, 0.13));
  const sustainPedalPivot = new THREE.Group();
  const pedalXs = [-0.085, 0, 0.085];
  pedalXs.forEach((px, idx) => {
    const pPivot = idx === 2 ? sustainPedalPivot : new THREE.Group();
    pPivot.position.set(px, 0.072, 0.04);
    lyreGroup.add(pPivot);

    const pedal = new THREE.Mesh(pedalGeo, polishedBrassMat);
    pedal.position.set(0, 0, 0.065);
    pPivot.add(pedal);
  });

  // =========================================================================
  // 8. CONCERT ARTIST BENCH (GROUNDED AT Y = 0)
  // =========================================================================
  const benchGroup = new THREE.Group();
  benchGroup.position.set(0, 0, 1.18);
  root.add(benchGroup);

  const benchW = 0.78;
  const benchD = 0.36;
  const benchH = 0.48;

  const benchApronGeo = trackDisposable(new THREE.BoxGeometry(benchW, 0.068, benchD));
  const benchApron = new THREE.Mesh(benchApronGeo, ebonyLacquerMat);
  benchApron.position.y = benchH - 0.055;
  benchApron.castShadow = true;
  benchApron.receiveShadow = true;
  benchGroup.add(benchApron);

  const benchCushionGeo = trackDisposable(new THREE.BoxGeometry(benchW + 0.018, 0.056, benchD + 0.018));
  const benchCushion = new THREE.Mesh(benchCushionGeo, benchLeatherMat);
  benchCushion.position.y = benchH;
  benchCushion.castShadow = true;
  benchCushion.receiveShadow = true;
  benchGroup.add(benchCushion);

  const benchLegGeo = trackDisposable(new THREE.CylinderGeometry(0.021, 0.014, benchH - 0.08, 12));
  const bLegOffsets: Array<[number, number]> = [
    [-benchW / 2 + 0.045, -benchD / 2 + 0.045],
    [benchW / 2 - 0.045, -benchD / 2 + 0.045],
    [-benchW / 2 + 0.045, benchD / 2 - 0.045],
    [benchW / 2 - 0.045, benchD / 2 - 0.045],
  ];
  for (const [bx, bz] of bLegOffsets) {
    const bLeg = new THREE.Mesh(benchLegGeo, ebonyLacquerMat);
    bLeg.position.set(bx, (benchH - 0.08) * 0.5, bz);
    bLeg.castShadow = true;
    benchGroup.add(bLeg);
  }

  // Brass height-adjustment knobs on bench sides
  const knobGeo = trackDisposable(new THREE.CylinderGeometry(0.022, 0.022, 0.024, 12));
  knobGeo.rotateZ(Math.PI / 2);
  for (const sx of [-benchW / 2 - 0.012, benchW / 2 + 0.012]) {
    const knob = new THREE.Mesh(knobGeo, polishedBrassMat);
    knob.position.set(sx, benchH - 0.055, 0);
    benchGroup.add(knob);
  }

  // =========================================================================
  // 9. RISING 3D NOTE RIBBONS (VISUALIZING PIANO VS VOCAL MELODY ABOVE KEYS)
  // =========================================================================
  const ribbonGroup = new THREE.Group();
  root.add(ribbonGroup);

  const ribbonGeo = trackDisposable(new THREE.BoxGeometry(0.015, 0.09, 0.015));
  const ribbons: RibbonParticle[] = [];
  const maxRibbons = 56;

  // Refined concert color palette distinguishing the 4 separated instrument tracks
  const rolePalette: Record<
    TrackRole,
    { glow: THREE.Color; whiteTint: THREE.Color; blackTint: THREE.Color; emissiveStrength: number }
  > = {
    vocal: {
      glow: new THREE.Color(0xf26b7a),
      whiteTint: new THREE.Color(0xffb8c0),
      blackTint: new THREE.Color(0xa63646),
      emissiveStrength: 0.95,
    },
    chords: {
      glow: new THREE.Color(0xf2b84b),
      whiteTint: new THREE.Color(0xffe19c),
      blackTint: new THREE.Color(0x946922),
      emissiveStrength: 0.72,
    },
    bass: {
      glow: new THREE.Color(0xd9823b),
      whiteTint: new THREE.Color(0xf7c59b),
      blackTint: new THREE.Color(0x8a4d1e),
      emissiveStrength: 0.75,
    },
    lead: {
      glow: new THREE.Color(0x7db4e6),
      whiteTint: new THREE.Color(0xc5e1fa),
      blackTint: new THREE.Color(0x365f85),
      emissiveStrength: 0.85,
    },
  };
  const blackColorZero = new THREE.Color(0x000000);

  const spawnRibbon = (handle: AnimatedKeyHandle, role: TrackRole) => {
    if (ribbons.length >= maxRibbons) {
      const oldest = ribbons.shift()!;
      ribbonGroup.remove(oldest.mesh);
      oldest.material.dispose();
    }
    const col = rolePalette[role].glow;
    const mat = new THREE.MeshStandardMaterial({
      color: col,
      emissive: col,
      emissiveIntensity: 1.3,
      roughness: 0.25,
      metalness: 0.1,
      transparent: true,
      opacity: 0.84,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(ribbonGeo, mat);
    const scaleY = role === 'vocal' ? 1.45 : role === 'bass' ? 1.15 : 0.95;
    mesh.scale.set(handle.isBlack ? 0.72 : 1.0, scaleY, 0.75);
    mesh.position.set(handle.keyX, handle.keyTopY + 0.055, handle.keyFrontZ - 0.04);
    ribbonGroup.add(mesh);
    ribbons.push({
      mesh,
      material: mat,
      age: 0,
      maxAge: role === 'vocal' ? 1.35 : 1.05,
      speed: role === 'vocal' ? 0.56 : 0.44,
      baseScaleY: scaleY,
    });
  };

  // =========================================================================
  // 10. KINEMATIC STATE & UPDATE LOOP
  // =========================================================================
  let currentLidAngle = 0.62; // Start open (~35.5 deg) so the golden harp & strings shine
  let currentPedalAngle = 0;

  const setKeyActive = (
    midi: number,
    active: boolean,
    role: TrackRole | null,
    velocity = 0.8,
  ) => {
    const handle = keys.get(midi);
    if (!handle) return;

    const wasActive = handle.targetPress > 0.1;
    handle.targetPress = active ? Math.max(0.55, Math.min(1.0, velocity)) : 0;
    handle.activeRole = active ? role : null;

    if (active && !wasActive) {
      handle.hammerKick = 1.0;
      if (role && ribbonGroup.visible) {
        spawnRibbon(handle, role);
      }
    }
  };

  const update = (
    dt: number,
    lidState: LidState,
    sustainActive: boolean,
    showRibbons: boolean,
  ): boolean => {
    ribbonGroup.visible = showRibbons;
    let needsMoreFrames = false;

    // 1. Lid & prop stick smooth exponential damping
    const targetLidAngle =
      lidState === 'open' ? 0.62 : lidState === 'half' ? 0.30 : 0.0;
    const prevLid = currentLidAngle;
    currentLidAngle = THREE.MathUtils.damp(currentLidAngle, targetLidAngle, 6.5, dt);
    if (Math.abs(currentLidAngle - targetLidAngle) > 0.001) {
      needsMoreFrames = true;
    } else if (prevLid !== targetLidAngle) {
      currentLidAngle = targetLidAngle;
    }
    lidPivot.rotation.z = currentLidAngle;

    // Position prop stick naturally against the lid angle
    if (currentLidAngle < 0.05) {
      propStick.visible = false;
    } else {
      propStick.visible = true;
      const lidRadius = 1.36;
      const contactX = -0.755 + Math.cos(currentLidAngle) * lidRadius;
      const contactY = rimTopY + 0.012 + Math.sin(currentLidAngle) * lidRadius;
      const baseX = 0.68;
      const baseY = rimTopY - 0.02;
      const dx = contactX - baseX;
      const dy = contactY - baseY;
      const len = Math.hypot(dx, dy);
      propStick.scale.set(1, len / propStickLen, 1);
      propStick.rotation.z = Math.atan2(-dx, dy);
    }

    // 2. Sustain pedal animation
    const targetPedal = sustainActive ? 0.18 : 0.0;
    currentPedalAngle = THREE.MathUtils.damp(currentPedalAngle, targetPedal, 12, dt);
    sustainPedalPivot.rotation.x = currentPedalAngle;
    if (Math.abs(currentPedalAngle - targetPedal) > 0.002) needsMoreFrames = true;

    // 3. 88 keys & felt hammers kinematics
    for (let midi = MIN_MIDI; midi <= MAX_MIDI; midi++) {
      const h = keys.get(midi)!;
      const prevPress = h.press;
      const speed = h.targetPress > h.press ? 28 : 14;
      h.press = THREE.MathUtils.damp(h.press, h.targetPress, speed, dt);
      if (Math.abs(h.press - h.targetPress) < 0.004) h.press = h.targetPress;

      if (h.press > 0.002 || prevPress > 0.002) {
        needsMoreFrames = true;
        h.pivot.rotation.x = h.press * 0.072;

        const pal = rolePalette[h.activeRole ?? 'chords'];
        const tintCol = h.isBlack ? pal.blackTint : pal.whiteTint;

        h.material.color.copy(h.baseColor).lerp(tintCol, h.press * 0.85);
        h.material.emissive.copy(pal.glow);
        h.material.emissiveIntensity = h.press * pal.emissiveStrength;
      } else if (h.material.emissiveIntensity !== 0) {
        h.pivot.rotation.x = 0;
        h.material.color.copy(h.baseColor);
        h.material.emissive.copy(blackColorZero);
        h.material.emissiveIntensity = 0;
      }

      // Internal hammer strike recoil
      if (h.hammerKick > 0.005) {
        needsMoreFrames = true;
        h.hammerKick = THREE.MathUtils.damp(h.hammerKick, 0, 14, dt);
        h.hammerPivot.rotation.x = -h.hammerKick * 0.42;
      } else if (h.hammerPivot.rotation.x !== 0) {
        h.hammerKick = 0;
        h.hammerPivot.rotation.x = 0;
      }
    }

    // 4. Rising 3D note ribbons
    for (let i = ribbons.length - 1; i >= 0; i--) {
      const r = ribbons[i];
      r.age += dt;
      if (r.age >= r.maxAge) {
        ribbonGroup.remove(r.mesh);
        r.material.dispose();
        ribbons.splice(i, 1);
      } else {
        needsMoreFrames = true;
        const progress = r.age / r.maxAge;
        r.mesh.position.y += r.speed * dt;
        r.material.opacity = (1 - progress) * (1 - progress) * 0.82;
        r.mesh.scale.y = r.baseScaleY * (1 + progress * 0.9);
      }
    }

    return needsMoreFrames;
  };

  const dispose = () => {
    for (const r of ribbons) {
      ribbonGroup.remove(r.mesh);
      r.material.dispose();
    }
    ribbons.length = 0;
    for (const d of disposables) {
      d.dispose();
    }
  };

  return {
    root,
    keys,
    clickableKeyMeshes,
    lidPivot,
    propStick,
    sustainPedalPivot,
    ribbonGroup,
    update,
    setKeyActive,
    dispose,
  };
}

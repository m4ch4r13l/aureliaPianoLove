import * as THREE from 'three';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { ProceduralTextureSet } from './proceduralTextures';

export interface ConcertStageHandle {
  group: THREE.Group;
  dispose: () => void;
}

/**
 * Creates the recital hall architectural stage, parquet floor, contact shadow grounding,
 * curved acoustic fluted backdrop, and warm/cool 3-point concert lighting rig.
 */
export function buildConcertStage(
  scene: THREE.Scene,
  textures: ProceduralTextureSet,
  isSoftwareGl: boolean,
): ConcertStageHandle {
  const group = new THREE.Group();
  group.name = 'ConcertHallStage';
  const disposables: Array<{ dispose: () => void }> = [];
  const track = <T extends { dispose: () => void }>(d: T): T => {
    disposables.push(d);
    return d;
  };

  // Match background & exponential fog for seamless architectural depth
  const bgColor = new THREE.Color(0x161923);
  scene.background = bgColor;
  scene.fog = new THREE.FogExp2(0x161923, 0.052);

  // 1. Expansive Walnut Parquet Concert Stage Floor (y = 0)
  const floorGeo = track(new THREE.CircleGeometry(14, 48));
  floorGeo.rotateX(-Math.PI / 2);
  const floorMat = track(
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: textures.floorColor,
      bumpMap: textures.floorBump,
      bumpScale: 0.014,
      roughnessMap: textures.floorRoughness,
      roughness: 0.52,
      metalness: 0.05,
    }),
  );
  const floorMesh = new THREE.Mesh(floorGeo, floorMat);
  floorMesh.position.y = 0;
  floorMesh.receiveShadow = true;
  group.add(floorMesh);

  // Subtle brass architectural floor inlay ring framing the central grand piano
  const ringGeo = track(new THREE.RingGeometry(2.38, 2.42, 64));
  ringGeo.rotateX(-Math.PI / 2);
  const ringMat = track(
    new THREE.MeshStandardMaterial({
      color: 0xd4a554,
      roughness: 0.28,
      metalness: 0.86,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    }),
  );
  const ringMesh = new THREE.Mesh(ringGeo, ringMat);
  ringMesh.position.set(0, 0.004, -0.25);
  group.add(ringMesh);

  // 2. Soft Radial Contact Shadow Planes directly beneath the Grand Piano & Bench
  const shadowMat = track(
    new THREE.MeshBasicMaterial({
      map: textures.contactShadow,
      transparent: true,
      opacity: 0.56,
      depthWrite: false,
    }),
  );
  const pianoShadowGeo = track(new THREE.PlaneGeometry(2.65, 3.35));
  pianoShadowGeo.rotateX(-Math.PI / 2);
  const pianoShadow = new THREE.Mesh(pianoShadowGeo, shadowMat);
  pianoShadow.position.set(-0.08, 0.015, -0.45);
  group.add(pianoShadow);

  const benchShadowGeo = track(new THREE.PlaneGeometry(1.18, 0.72));
  benchShadowGeo.rotateX(-Math.PI / 2);
  const benchShadow = new THREE.Mesh(benchShadowGeo, shadowMat);
  benchShadow.position.set(0, 0.015, 1.18);
  group.add(benchShadow);

  // 3. Curved Architectural Acoustic Fluted Wall Backdrop (InstancedMesh vertical walnut slats)
  const slatCount = 46;
  const slatGeo = track(new THREE.BoxGeometry(0.11, 5.2, 0.08));
  const slatMat = track(
    new THREE.MeshStandardMaterial({
      color: 0x4b382a,
      roughness: 0.62,
      metalness: 0.05,
    }),
  );
  const slats = new THREE.InstancedMesh(slatGeo, slatMat, slatCount);
  const dummy = new THREE.Object3D();
  const wallRadius = 5.8;
  const arcStart = -Math.PI * 0.68;
  const arcEnd = Math.PI * 0.68;

  for (let i = 0; i < slatCount; i++) {
    const t = i / (slatCount - 1);
    const angle = arcStart + t * (arcEnd - arcStart);
    const x = Math.sin(angle) * wallRadius;
    const z = -Math.cos(angle) * wallRadius - 0.25;
    dummy.position.set(x, 2.6, z);
    dummy.rotation.set(0, -angle, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    slats.setMatrixAt(i, dummy.matrix);
  }
  slats.instanceMatrix.needsUpdate = true;
  slats.computeBoundingSphere();
  group.add(slats);

  // Subtle dark acoustic backing behind the wooden slats
  const backingGeo = track(
    new THREE.CylinderGeometry(
      wallRadius + 0.14,
      wallRadius + 0.14,
      5.4,
      40,
      1,
      true,
      Math.PI - arcEnd,
      arcEnd - arcStart,
    ),
  );
  const backingMat = track(
    new THREE.MeshStandardMaterial({
      color: 0x1b1e28,
      roughness: 0.9,
      metalness: 0.02,
      side: THREE.BackSide,
    }),
  );
  const backingMesh = new THREE.Mesh(backingGeo, backingMat);
  backingMesh.position.set(0, 2.7, -0.25);
  group.add(backingMesh);

  // 4. Balanced Concert Hall 3-Point Warm/Cool Lighting Rig
  // Cool architectural hall skylight vs warm parquet floor bounce
  const hemiLight = new THREE.HemisphereLight(0xb8cceb, 0x4f3b28, 1.15);
  hemiLight.position.set(0, 8, 0);
  scene.add(hemiLight);

  // Primary Warm Concert Spotlight / Key Directional Light (single shadow caster)
  const keyLight = new THREE.DirectionalLight(0xfff1d6, 2.65);
  keyLight.position.set(3.4, 5.8, 3.3);
  keyLight.target.position.set(0, 0.75, -0.15);
  scene.add(keyLight.target);
  if (!isSoftwareGl) {
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.near = 1.0;
    keyLight.shadow.camera.far = 14.0;
    const d = 3.1;
    keyLight.shadow.camera.left = -d;
    keyLight.shadow.camera.right = d;
    keyLight.shadow.camera.top = d;
    keyLight.shadow.camera.bottom = -d;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.02;
    keyLight.shadow.radius = 3;
  }
  scene.add(keyLight);

  // Cool Architectural Rim Light separating the gloss ebony silhouette from backdrop
  const rimLight = new THREE.DirectionalLight(0xaacbfa, 1.45);
  rimLight.position.set(-4.2, 3.8, -3.6);
  rimLight.target.position.set(0, 0.85, -0.35);
  scene.add(rimLight.target);
  scene.add(rimLight);

  // Warm concert fill lamp illuminating the golden cast-iron harp, strings & 88 keys
  const harpFillLight = new THREE.PointLight(0xffe1b0, 3.1, 7.5, 1.8);
  harpFillLight.position.set(0.75, 2.35, 0.65);
  scene.add(harpFillLight);

  // Overhead warm studio softbox for elongated specular gloss highlights on the lid (GPU only)
  if (!isSoftwareGl) {
    RectAreaLightUniformsLib.init();
    const softbox = new THREE.RectAreaLight(0xfff5e4, 2.5, 3.6, 1.4);
    softbox.position.set(0.4, 3.8, 1.1);
    softbox.lookAt(0, 0.85, -0.1);
    scene.add(softbox);
  }

  scene.add(group);

  return {
    group,
    dispose: () => {
      for (const d of disposables) {
        d.dispose();
      }
    },
  };
}

import * as THREE from 'three';

export interface ProceduralTextureSet {
  floorColor: THREE.CanvasTexture;
  floorBump: THREE.CanvasTexture;
  floorRoughness: THREE.CanvasTexture;
  soundboardColor: THREE.CanvasTexture;
  contactShadow: THREE.CanvasTexture;
  sheetMusic: THREE.CanvasTexture;
  dispose: () => void;
}

/**
 * Generates all 100% procedural concert-hall textures via HTML5 2D Canvas
 * (zero external file URLs, zero 404s, calibrated sRGB / linear color spaces).
 */
export function createConcertTextures(maxAnisotropy: number): ProceduralTextureSet {
  // 1. Concert Hall Warm Walnut Parquet Floor (Color + Bump + Roughness)
  const floorSize = 512;
  const colorCanvas = document.createElement('canvas');
  colorCanvas.width = floorSize;
  colorCanvas.height = floorSize;
  const cCtx = colorCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = floorSize;
  bumpCanvas.height = floorSize;
  const bCtx = bumpCanvas.getContext('2d')!;

  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = floorSize;
  roughCanvas.height = floorSize;
  const rCtx = roughCanvas.getContext('2d')!;

  cCtx.fillStyle = '#4a3322';
  cCtx.fillRect(0, 0, floorSize, floorSize);
  bCtx.fillStyle = '#808080';
  bCtx.fillRect(0, 0, floorSize, floorSize);
  rCtx.fillStyle = '#707070';
  rCtx.fillRect(0, 0, floorSize, floorSize);

  const plankRows = 8;
  const plankCols = 4;
  const plankH = floorSize / plankRows;
  const plankW = floorSize / plankCols;

  // Deterministic pseudo-random for consistent wood tones
  let seed = 4219;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  for (let row = 0; row < plankRows; row++) {
    const xOffset = (row % 2) * (plankW * 0.5);
    for (let col = -1; col <= plankCols; col++) {
      const px = col * plankW + xOffset;
      const py = row * plankH;
      const toneShift = (rand() - 0.5) * 24;
      const r = Math.round(108 + toneShift);
      const g = Math.round(74 + toneShift * 0.7);
      const b = Math.round(48 + toneShift * 0.5);

      cCtx.fillStyle = `rgb(${r},${g},${b})`;
      cCtx.fillRect(px + 1, py + 1, plankW - 2, plankH - 2);

      const roughVal = Math.round(92 + (rand() - 0.5) * 26);
      rCtx.fillStyle = `rgb(${roughVal},${roughVal},${roughVal})`;
      rCtx.fillRect(px + 1, py + 1, plankW - 2, plankH - 2);

      // Subtle wood grain fibers
      for (let gLine = 0; gLine < 14; gLine++) {
        const gy = py + 3 + rand() * (plankH - 6);
        const alpha = 0.05 + rand() * 0.09;
        cCtx.strokeStyle = rand() > 0.5 ? `rgba(18,10,5,${alpha})` : `rgba(120,84,54,${alpha * 0.6})`;
        cCtx.lineWidth = 1 + rand() * 1.4;
        cCtx.beginPath();
        cCtx.moveTo(px + 2, gy);
        cCtx.bezierCurveTo(
          px + plankW * 0.33,
          gy + (rand() - 0.5) * 4,
          px + plankW * 0.66,
          gy + (rand() - 0.5) * 4,
          px + plankW - 2,
          gy,
        );
        cCtx.stroke();

        bCtx.strokeStyle = `rgba(195,195,195,${alpha * 0.8})`;
        bCtx.lineWidth = 1;
        bCtx.beginPath();
        bCtx.moveTo(px + 2, gy);
        bCtx.lineTo(px + plankW - 2, gy);
        bCtx.stroke();
      }

      // Plank bevel seam
      cCtx.strokeStyle = 'rgba(10,7,5,0.65)';
      cCtx.lineWidth = 1.5;
      cCtx.strokeRect(px + 0.5, py + 0.5, plankW - 1, plankH - 1);

      bCtx.strokeStyle = '#262626';
      bCtx.lineWidth = 2;
      bCtx.strokeRect(px + 0.5, py + 0.5, plankW - 1, plankH - 1);

      rCtx.strokeStyle = '#d4d4d4';
      rCtx.lineWidth = 2;
      rCtx.strokeRect(px + 0.5, py + 0.5, plankW - 1, plankH - 1);
    }
  }

  const floorColor = new THREE.CanvasTexture(colorCanvas);
  floorColor.colorSpace = THREE.SRGBColorSpace;
  floorColor.wrapS = THREE.RepeatWrapping;
  floorColor.wrapT = THREE.RepeatWrapping;
  floorColor.repeat.set(7, 7);
  floorColor.anisotropy = maxAnisotropy;

  const floorBump = new THREE.CanvasTexture(bumpCanvas);
  floorBump.wrapS = THREE.RepeatWrapping;
  floorBump.wrapT = THREE.RepeatWrapping;
  floorBump.repeat.set(7, 7);
  floorBump.anisotropy = maxAnisotropy;

  const floorRoughness = new THREE.CanvasTexture(roughCanvas);
  floorRoughness.wrapS = THREE.RepeatWrapping;
  floorRoughness.wrapT = THREE.RepeatWrapping;
  floorRoughness.repeat.set(7, 7);
  floorRoughness.anisotropy = maxAnisotropy;

  // 2. Sitka Spruce Soundboard Texture (warm honey-amber fine grain)
  const sbCanvas = document.createElement('canvas');
  sbCanvas.width = 512;
  sbCanvas.height = 512;
  const sbCtx = sbCanvas.getContext('2d')!;
  const sbGrad = sbCtx.createLinearGradient(0, 0, 512, 512);
  sbGrad.addColorStop(0, '#c28b48');
  sbGrad.addColorStop(0.5, '#b17938');
  sbGrad.addColorStop(1, '#9b672d');
  sbCtx.fillStyle = sbGrad;
  sbCtx.fillRect(0, 0, 512, 512);

  for (let i = 0; i < 240; i++) {
    const x = (i / 240) * 512;
    sbCtx.strokeStyle = i % 3 === 0 ? 'rgba(85,48,16,0.16)' : 'rgba(235,195,125,0.09)';
    sbCtx.lineWidth = 1.1;
    sbCtx.beginPath();
    sbCtx.moveTo(x, 0);
    sbCtx.lineTo(x + (rand() - 0.5) * 6, 512);
    sbCtx.stroke();
  }
  const soundboardColor = new THREE.CanvasTexture(sbCanvas);
  soundboardColor.colorSpace = THREE.SRGBColorSpace;
  soundboardColor.wrapS = THREE.RepeatWrapping;
  soundboardColor.wrapT = THREE.RepeatWrapping;
  soundboardColor.anisotropy = maxAnisotropy;

  // 3. Soft Radial Contact Shadow Disc (for figure-ground separation under piano & bench)
  const csCanvas = document.createElement('canvas');
  csCanvas.width = 256;
  csCanvas.height = 256;
  const csCtx = csCanvas.getContext('2d')!;
  const csGrad = csCtx.createRadialGradient(128, 128, 12, 128, 128, 124);
  csGrad.addColorStop(0, 'rgba(0, 0, 0, 0.72)');
  csGrad.addColorStop(0.52, 'rgba(0, 0, 0, 0.36)');
  csGrad.addColorStop(0.82, 'rgba(0, 0, 0, 0.1)');
  csGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  csCtx.fillStyle = csGrad;
  csCtx.fillRect(0, 0, 256, 256);

  const contactShadow = new THREE.CanvasTexture(csCanvas);
  contactShadow.colorSpace = THREE.SRGBColorSpace;

  // 4. Concert Sheet Music Page Texture
  const smCanvas = document.createElement('canvas');
  smCanvas.width = 512;
  smCanvas.height = 384;
  const smCtx = smCanvas.getContext('2d')!;
  smCtx.fillStyle = '#efe7d6';
  smCtx.fillRect(0, 0, 512, 384);

  // Center fold crease
  const crease = smCtx.createLinearGradient(246, 0, 266, 0);
  crease.addColorStop(0, 'rgba(0,0,0,0)');
  crease.addColorStop(0.5, 'rgba(35,24,12,0.22)');
  crease.addColorStop(1, 'rgba(0,0,0,0)');
  smCtx.fillStyle = crease;
  smCtx.fillRect(246, 0, 20, 384);

  // Draw grand staves & notes on both pages
  const drawPage = (offsetX: number) => {
    smCtx.fillStyle = '#2d261e';
    smCtx.font = 'italic 600 13px Georgia, serif';
    smCtx.textAlign = 'center';
    smCtx.fillText(offsetX < 256 ? 'Nocturno · Op. 27' : 'Voz & Piano', offsetX + 112, 34);

    for (let sys = 0; sys < 5; sys++) {
      const yTop = 58 + sys * 62;
      smCtx.strokeStyle = 'rgba(42,35,28,0.55)';
      smCtx.lineWidth = 1;
      for (let line = 0; line < 5; line++) {
        smCtx.beginPath();
        smCtx.moveTo(offsetX + 20, yTop + line * 5.5);
        smCtx.lineTo(offsetX + 208, yTop + line * 5.5);
        smCtx.stroke();
      }
      // Subtle musical noteheads and stems
      smCtx.fillStyle = 'rgba(30,24,18,0.82)';
      for (let n = 0; n < 10; n++) {
        const nx = offsetX + 34 + n * 17;
        const ny = yTop + ((sys * 7 + n * 5) % 5) * 5.5;
        smCtx.beginPath();
        smCtx.ellipse(nx, ny, 3.2, 2.2, -0.35, 0, Math.PI * 2);
        smCtx.fill();
        smCtx.fillRect(nx + 2.4, ny - 15, 1.1, 15);
      }
    }
  };
  drawPage(14);
  drawPage(270);

  const sheetMusic = new THREE.CanvasTexture(smCanvas);
  sheetMusic.colorSpace = THREE.SRGBColorSpace;
  sheetMusic.anisotropy = maxAnisotropy;

  return {
    floorColor,
    floorBump,
    floorRoughness,
    soundboardColor,
    contactShadow,
    sheetMusic,
    dispose: () => {
      floorColor.dispose();
      floorBump.dispose();
      floorRoughness.dispose();
      soundboardColor.dispose();
      contactShadow.dispose();
      sheetMusic.dispose();
    },
  };
}

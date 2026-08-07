/**
 * eiffelGeometry.ts — the Eiffel Tower, generated rather than downloaded.
 *
 * Why procedural: the reference aesthetic is a line drawing, not a photoreal
 * render. A lattice tower is repeated truss geometry, which means it can be
 * described by a profile curve and a bracing rule — so it costs zero asset
 * bytes, has no licence attached, needs no Draco decoder, and every
 * proportion stays tunable. A downloaded mesh would be heavier, less
 * faithful to the drawing look, and impossible to adjust.
 *
 * Everything is LineSegments. There are no materials to light, no textures,
 * and no shadows: on a blueprint there is only ink.
 *
 * Real proportions (Gustave Eiffel, 1889), normalised to height 1.0:
 *   base half-width  0.208   (125 m square base over 300 m to the tip)
 *   platform 1       0.190   (57 m)
 *   platform 2       0.383   (115 m)
 *   platform 3       0.920   (276 m)
 */

import * as THREE from "three";

export interface TowerOptions {
  height?: number;
  /** Height samples. More = finer lattice, linearly more segments. */
  rings?: number;
  /** Diagonal braces per face per bay. */
  braces?: number;
}

const P1 = 0.19;
const P2 = 0.383;
const P3 = 0.92;

/**
 * Half-width of the tower at normalised height t.
 *
 * Eiffel sized the tower so wind load is carried without bending moment,
 * which produces a near-exponential taper. A power curve fits the real
 * silhouette closely and, unlike a true exponential, lands exactly on the
 * intended top width.
 */
export function halfWidth(t: number): number {
  const base = 0.208;
  const top = 0.0125;
  const shaped = Math.pow(1 - Math.min(Math.max(t, 0), 1), 2.35);
  let w = top + (base - top) * shaped;
  // The real tower steps slightly wider just under each platform, where the
  // deck cantilevers past the legs. Without this the silhouette is too clean.
  for (const p of [P1, P2, P3]) {
    const d = Math.abs(t - p);
    if (d < 0.028) w *= 1 + 0.16 * (1 - d / 0.028);
  }
  return w;
}

function corners(t: number, h: number): THREE.Vector3[] {
  const w = halfWidth(t) * h;
  const y = t * h;
  return [
    new THREE.Vector3(-w, y, -w),
    new THREE.Vector3(w, y, -w),
    new THREE.Vector3(w, y, w),
    new THREE.Vector3(-w, y, w),
  ];
}

function push(arr: number[], a: THREE.Vector3, b: THREE.Vector3) {
  arr.push(a.x, a.y, a.z, b.x, b.y, b.z);
}

/**
 * Build the tower as two geometries so the drawing has line weight:
 *   primary   legs, platforms, arches — the structure you read first
 *   secondary lattice bracing — the texture
 * WebGL cannot vary line width, so hierarchy comes from opacity instead.
 */
export function buildEiffel(opts: TowerOptions = {}): {
  primary: THREE.BufferGeometry;
  secondary: THREE.BufferGeometry;
} {
  const h = opts.height ?? 10;
  const rings = opts.rings ?? 56;
  const braces = opts.braces ?? 2;

  const main: number[] = [];
  const lattice: number[] = [];

  // ── legs + ring beams + bracing ──────────────────────────────────────
  let prev = corners(0, h);
  for (let i = 1; i <= rings; i++) {
    const t = i / rings;
    const cur = corners(t, h);

    // four legs
    for (let c = 0; c < 4; c++) push(main, prev[c], cur[c]);

    // ring beam at every other sample keeps the lattice from turning to mud
    if (i % 2 === 0 || isPlatform(t, rings)) {
      for (let c = 0; c < 4; c++) push(main, cur[c], cur[(c + 1) % 4]);
    }

    // diagonal bracing on each of the four faces
    for (let c = 0; c < 4; c++) {
      const a0 = prev[c];
      const a1 = prev[(c + 1) % 4];
      const b0 = cur[c];
      const b1 = cur[(c + 1) % 4];
      for (let k = 0; k < braces; k++) {
        const f0 = k / braces;
        const f1 = (k + 1) / braces;
        const pA = a0.clone().lerp(a1, f0);
        const pB = b0.clone().lerp(b1, f1);
        const pC = a0.clone().lerp(a1, f1);
        const pD = b0.clone().lerp(b1, f0);
        push(lattice, pA, pB);
        push(lattice, pC, pD);
      }
    }
    prev = cur;
  }

  // ── platforms ────────────────────────────────────────────────────────
  for (const p of [P1, P2, P3]) {
    const deckW = halfWidth(p) * h * 1.22;
    const y = p * h;
    const depth = h * 0.016;
    for (const dy of [0, -depth]) {
      const ring = [
        new THREE.Vector3(-deckW, y + dy, -deckW),
        new THREE.Vector3(deckW, y + dy, -deckW),
        new THREE.Vector3(deckW, y + dy, deckW),
        new THREE.Vector3(-deckW, y + dy, deckW),
      ];
      for (let c = 0; c < 4; c++) push(main, ring[c], ring[(c + 1) % 4]);
    }
    // fascia posts
    for (let c = 0; c < 4; c++) {
      const x = c === 0 || c === 3 ? -deckW : deckW;
      const z = c < 2 ? -deckW : deckW;
      push(
        main,
        new THREE.Vector3(x, y, z),
        new THREE.Vector3(x, y - depth, z),
      );
    }
  }

  // ── the base arches ──────────────────────────────────────────────────
  // The four decorative arches under platform 1 are the tower's most
  // recognisable feature at eye level. Without them the base reads as a
  // generic pylon.
  const archTop = P1 * 0.82;
  const wBase = halfWidth(0) * h;
  const wArch = halfWidth(archTop) * h;
  const segments = 22;
  for (let face = 0; face < 4; face++) {
    const pts: THREE.Vector3[] = [];
    for (let s = 0; s <= segments; s++) {
      const u = s / segments; // 0..1 across the face
      const t = Math.sin(u * Math.PI) * archTop; // rises to the crown
      const w = THREE.MathUtils.lerp(wBase, wArch, Math.sin(u * Math.PI));
      const lateral = THREE.MathUtils.lerp(-1, 1, u) * w;
      const y = t * h * 0.9;
      pts.push(faceVec(face, lateral, y, w));
    }
    for (let s = 0; s < pts.length - 1; s++) push(main, pts[s], pts[s + 1]);
  }

  // ── antenna ──────────────────────────────────────────────────────────
  push(
    main,
    new THREE.Vector3(0, h, 0),
    new THREE.Vector3(0, h * 1.075, 0),
  );

  return {
    primary: toGeometry(main, h),
    secondary: toGeometry(lattice, h),
  };
}

function faceVec(face: number, lateral: number, y: number, w: number) {
  switch (face) {
    case 0:
      return new THREE.Vector3(lateral, y, -w);
    case 1:
      return new THREE.Vector3(w, y, lateral);
    case 2:
      return new THREE.Vector3(lateral, y, w);
    default:
      return new THREE.Vector3(-w, y, lateral);
  }
}

function isPlatform(t: number, rings: number) {
  const tol = 1 / rings;
  return [P1, P2, P3].some((p) => Math.abs(t - p) < tol);
}

/** Centre the model on its own mid-height so rotation looks right. */
function toGeometry(positions: number[], h: number): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const arr = new Float32Array(positions);
  for (let i = 1; i < arr.length; i += 3) arr[i] -= h / 2;
  g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
  return g;
}

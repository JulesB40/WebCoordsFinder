/**
 * Automatic field of view from the texel lattice (port of TexRotation's texrot/fov.py).
 *
 * A plane's perspective lattice fixes the focal length as well as the pose; Minecraft's FOV is
 * the vertical FOV with square pixels and a centred principal point, so it is the only intrinsic.
 * 1. a full pose fit at a few starting FOVs;
 * 2. around each, a FOV scan with a local coherence grid over yaw, pitch and eye height;
 * 3. a joint polish of (yaw, pitch, h, FOV);
 * 4. the candidate explaining the most floor and ceiling pixels; then a fine scan scoring how
 *    well every plane level lines up (only the right FOV puts levels exactly one block apart).
 */
import type { Kernels } from './kernels'
import {
  type Masks,
  type PoseFit,
  type Report,
  coherenceGrid,
  fitPose,
  planeCoherence,
  pooledFitScore,
  refinePose,
  screenFor,
} from './geometry'
import { DEG, arange, linspace, nelderMead, rng, sampleIndices } from './numeric'
import type { Pose } from './types'

export const START_FOVS = [35, 55, 75, 95, 110]
export const FOV_RANGE: [number, number] = [25, 115]

interface Samples {
  /** Screen coordinates (a, b): ray = normalize(a t, b t, 1) with t = tan(fov / 2). */
  ab: Float32Array
  phx: Float32Array
  phz: Float32Array
}

function raysFrom(ab: Float32Array, t: number): Float32Array {
  const n = ab.length / 2
  const out = new Float32Array(n * 3)
  for (let k = 0; k < n; k += 1) {
    const a = ab[2 * k] * t
    const b = ab[2 * k + 1] * t
    const norm = Math.sqrt(a * a + b * b + 1)
    out[3 * k] = a / norm
    out[3 * k + 1] = b / norm
    out[3 * k + 2] = 1 / norm
  }
  return out
}

function tanHalf(fov: number): number {
  return Math.tan((fov * DEG) / 2)
}

/** Joint polish of (yaw, pitch, h, fov) on the lattice coherence of one plane's samples. */
function refineWithFov(pose: { yaw: number; pitch: number; h: number }, fov: number, s: Samples) {
  const objective = ([y, p, h, logt]: number[]) => {
    const c = planeCoherence(y, p, h, raysFrom(s.ab, Math.exp(logt)), s.phx, s.phz)
    return -(c.kx + c.kz)
  }
  const start = [pose.yaw, pose.pitch, pose.h, Math.log(tanHalf(fov))]
  const { x, value } = nelderMead(objective, start, [0.05, 0.05, Math.abs(pose.h) * 0.002, 0.003], {
    maxEvals: 400,
    tol: 1e-9,
  })
  if (!x.every(Number.isFinite)) return { ...pose, fov, coherence: 0 }
  return { yaw: x[0], pitch: x[1], h: x[2], fov: (2 * Math.atan(Math.exp(x[3]))) / DEG, coherence: -value }
}

async function scanFov(kernels: Kernels, pose: Pose, fov0: number, s: Samples, sign: number, half = 12, step = 2) {
  let best: { fov: number; yaw: number; pitch: number; h: number; value: number } | null = null
  for (const fov of arange(Math.max(FOV_RANGE[0], fov0 - half), Math.min(FOV_RANGE[1], fov0 + half) + 1e-3, step)) {
    const g = await coherenceGrid(
      kernels, raysFrom(s.ab, tanHalf(fov)), s.phx, s.phz,
      linspace(pose.yaw - 2, pose.yaw + 2, 9),
      linspace(pose.pitch - 10, pose.pitch + 10, 21).map((p) => Math.min(90, Math.max(-90, p))),
      linspace(-0.35, 0.35, 21).map((e) => pose.h * Math.exp(e)),
      sign,
    )
    if (!best || g.value > best.value) best = { fov, ...g }
  }
  return best!
}

interface FovCandidate extends Pose {
  fov: number
  explained: number
}

async function fineFov(
  kernels: Kernels,
  best: FovCandidate,
  plane: Samples,
  sign: number,
  width: number,
  height: number,
  tph: Float32Array,
  floor: Uint8Array,
  ceiling: Uint8Array,
  report: Report,
  half = 5,
  step = 0.5,
): Promise<{ fov: number; score: number; pose: PoseFit | null }> {
  const fovs: number[] = []
  const scores: number[] = []
  const poses: PoseFit[] = []
  for (const fov of arange(best.fov - half, best.fov + half + 1e-3, step)) {
    if (fov < FOV_RANGE[0] || fov > FOV_RANGE[1]) continue
    report(`Refining the field of view (${fov.toFixed(1)}°)`)
    const rays = raysFrom(plane.ab, tanHalf(fov))
    let { yaw: y, pitch: p, h } = best
    for (const [hw, n, hs] of [[3, 13, 0.12], [0.7, 11, 0.03], [0.15, 9, 0.006]]) {
      ;({ yaw: y, pitch: p, h } = await coherenceGrid(
        kernels, rays, plane.phx, plane.phz,
        linspace(y - hw, y + hw, n),
        linspace(p - hw, p + hw, n).map((v) => Math.min(90, Math.max(-90, v))),
        linspace(-hs, hs, 21).map((e) => h * Math.exp(e)),
        sign,
      ))
    }
    const pose = refinePose(y, p, h, rays, plane.phx, plane.phz)
    if (Math.abs(pose.pitch - p) > 1 || Math.abs(pose.yaw - y) > 1) continue
    fovs.push(fov)
    scores.push(pooledFitScore(pose, screenFor(width, height, fov), tph, floor, ceiling))
    poses.push(pose)
  }
  if (!fovs.length) return { fov: best.fov, score: best.explained, pose: null }
  const padded = [scores[0], ...scores, scores[scores.length - 1]]
  let i = 0
  let top = -Infinity
  scores.forEach((_, k) => {
    const v = 0.25 * padded[k] + 0.5 * padded[k + 1] + 0.25 * padded[k + 2]
    if (v > top) {
      top = v
      i = k
    }
  })
  return { fov: fovs[i], score: scores[i], pose: poses[i] }
}

export interface FovEstimate {
  fov: number
  /** False looking (almost) straight down at a single plane, where FOV and height trade off. */
  wellPosed: boolean
}

/** Vertical FOV (degrees) from network outputs, or null when there is no usable plane. */
export async function estimateFov(
  kernels: Kernels,
  width: number,
  height: number,
  tph: Float32Array,
  masks: Masks,
  report: Report = () => {},
): Promise<FovEstimate | null> {
  const n = width * height
  const floor = new Uint8Array(n)
  const ceiling = new Uint8Array(n)
  const usable: number[] = []
  for (let p = 0; p < n; p += 1) {
    floor[p] = masks.floor[p] & masks.smooth[p]
    ceiling[p] = masks.ceiling[p] & masks.smooth[p]
    if (floor[p] || ceiling[p]) usable.push(p)
  }
  if (usable.length < 500) return null

  // scoring pixels: all usable floor + ceiling pixels (subsampled)
  const pick = sampleIndices(usable.length, 40000, rng(3))
  const bySign = new Map<number, Samples>()
  for (const sign of [1, -1]) {
    const chosen = [...pick].map((k) => usable[k]).filter((p) => (sign === 1 ? floor[p] : ceiling[p]))
    const ab = new Float32Array(chosen.length * 2)
    const phx = new Float32Array(chosen.length)
    const phz = new Float32Array(chosen.length)
    chosen.forEach((p, k) => {
      const x = p % width
      const y = Math.floor(p / width)
      ab[2 * k] = ((2 * (x + 0.5)) / width - 1) * (width / height)
      ab[2 * k + 1] = 1 - (2 * (y + 0.5)) / height
      phx[k] = tph[3 * p]
      phz[k] = tph[3 * p + 2]
    })
    bySign.set(sign, { ab, phx, phz })
  }

  const cands: FovCandidate[] = []
  for (const fov0 of START_FOVS) {
    report(`Estimating the field of view (trying ${fov0}°)`)
    const { pose } = await fitPose(kernels, screenFor(width, height, fov0), tph, floor, ceiling)
    if (!pose) continue
    const sign = pose.h > 0 ? 1 : -1
    const plane = bySign.get(sign)!
    const scan = await scanFov(kernels, pose, fov0, plane, sign)
    let ref: { yaw: number; pitch: number; h: number; fov: number } = refineWithFov(scan, scan.fov, plane)
    if (!(ref.fov >= FOV_RANGE[0] && ref.fov <= FOV_RANGE[1]) || Math.abs(ref.fov - scan.fov) > 6) {
      ref = { yaw: scan.yaw, pitch: scan.pitch, h: scan.h, fov: scan.fov } // polish ran away
    }
    // sub-texel offsets for the explained-pixel score
    const c = planeCoherence(ref.yaw, ref.pitch, ref.h, raysFrom(plane.ab, tanHalf(ref.fov)), plane.phx, plane.phz)
    const cand = { ...ref, c16x: c.cx, c16z: c.cz, explained: 0 }
    cand.explained = pooledFitScore(cand, screenFor(width, height, ref.fov), tph, floor, ceiling)
    cands.push(cand)
  }
  if (!cands.length) return null
  const best = cands.reduce((a, b) => (b.explained > a.explained ? b : a))
  const sign = best.h > 0 ? 1 : -1
  const plane = bySign.get(sign)!
  let fine = await fineFov(kernels, best, plane, sign, width, height, tph, floor, ceiling, report)
  if (fine.pose && Math.abs(fine.fov - best.fov) >= 4.5) {
    // the best FOV sat on the window's edge: look again around it
    const again = await fineFov(kernels, { ...fine.pose, fov: fine.fov, explained: fine.score }, plane, sign,
      width, height, tph, floor, ceiling, report)
    if (again.score > fine.score) fine = again
  }
  return { fov: fine.fov, wellPosed: Math.abs(best.pitch) < 80 }
}

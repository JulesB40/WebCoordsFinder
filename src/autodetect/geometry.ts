/**
 * Network outputs + texture template matching -> matched block faces. A port of TexRotation's
 * texrot/geometry.py; see that file for the reasoning behind each stage.
 *
 * 1. Pose: grid search over (yaw, pitch, eye height) on lattice phase steps, staged coherence
 *    grids and a polish, floors and ceilings both tried; the pose explaining the most pixels wins.
 * 2. Camera sub-texel offset from the coherence phase.
 * 3. Horizontal planes: per-pixel level, resampling at world texel centres, masked NCC with the
 *    template bank; the block-grid alignment (ox, oz) pooled over every plane.
 * 4. Walls on block boundaries, matched block by block.
 *
 * Frames: the "grid frame" is camera-relative with X_g = x + c16x/16, Y_g = y + h, Z_g = z + c16z/16.
 */
import { cosTurns, sinTurns } from './cpuKernels'
import type { Kernels } from './kernels'
import type { FaceName, Pose } from './types'
import {
  DEG,
  TAU,
  arange,
  boxSum,
  cauchy,
  linspace,
  mod,
  nelderMead,
  rng,
  sampleIndices,
  signedInv,
  wrap,
} from './numeric'

export const PARAMS = {
  minFaceConf: 0.6,
  minTemplated: 0.5,
  minTexelPx: 1.2,
  minValidTexels: 150,
  maxExtentBlocks: 24,
  maxLevels: 6,
  maxWalls: 12,
  minPlanePixels: 300,
}

/** Face classes of the geometry network. */
export const FACE_CLASS = { other: 0, up: 1, down: 2, east: 3, west: 4, south: 5, north: 6 } as const

export interface DensePrediction {
  width: number
  height: number
  /** Most likely face class per pixel and its probability. */
  face: Uint8Array
  faceConf: Float32Array
  templated: Float32Array
  /** Camera-relative position [N * 3], blocks. */
  pos: Float32Array
  /** Texel lattice phase frac(16 * world xyz) [N * 3]. */
  tphase: Float32Array
}

export interface PoseFit extends Pose {
  coherence: [number, number]
  explained?: number
}

export type TemplateKind = 'four' | 'side' | 'netherrack' | 'static'

export interface TemplateMeta {
  block: string
  kind: TemplateKind
  value: number
  parity: number
}

export interface FaceTemplates {
  /** [count, 16, 16] luminance templates. */
  data: Float32Array
  count: number
  meta: TemplateMeta[]
}

export type TemplateBank = Record<FaceName, FaceTemplates>

export interface MatchedBlock {
  cell: [number, number, number]
  face: FaceName
  kind: TemplateKind
  value: number
  parity: number
  texture: string
  ncc: number
  margin: number
  valid: number
  centroid: [number, number]
  /** Best score per row value of the matched block (for review), best first. */
  scores: { variant: number; score: number }[]
}

export type Report = (message: string) => void

// ------------------------------------------------------------------- camera ---

/** Screen of a camera with vertical FOV `fov` (degrees). */
export interface Screen {
  width: number
  height: number
  /** tan(fov / 2) */
  t: number
}

export function screenFor(width: number, height: number, fov: number): Screen {
  return { width, height, t: Math.tan((fov * DEG) / 2) }
}

/** Camera-frame ray (x right, y up, z forward) through the centre of pixel (x, y). */
export function rayAt(s: Screen, x: number, y: number, out: Float32Array, o: number): void {
  const a = ((2 * (x + 0.5)) / s.width - 1) * s.t * (s.width / s.height)
  const b = (1 - (2 * (y + 0.5)) / s.height) * s.t
  const n = Math.sqrt(a * a + b * b + 1)
  out[o] = a / n
  out[o + 1] = b / n
  out[o + 2] = 1 / n
}

export function gatherRays(s: Screen, pixels: Int32Array): Float32Array {
  const out = new Float32Array(pixels.length * 3)
  for (let k = 0; k < pixels.length; k += 1) {
    const p = pixels[k]
    rayAt(s, p % s.width, Math.floor(p / s.width), out, 3 * k)
  }
  return out
}

export interface Basis {
  r: [number, number, number]
  u: [number, number, number]
  f: [number, number, number]
}

export function basis(yawDeg: number, pitchDeg: number): Basis {
  const sy = Math.sin(yawDeg * DEG)
  const cy = Math.cos(yawDeg * DEG)
  const sp = Math.sin(pitchDeg * DEG)
  const cp = Math.cos(pitchDeg * DEG)
  return { f: [-sy * cp, -sp, cy * cp], r: [-cy, 0, -sy], u: [-sy * sp, cp, cy * sp] }
}

function worldDir(b: Basis, rays: Float32Array, o: number, out: number[]): void {
  const x = rays[o]
  const y = rays[o + 1]
  const z = rays[o + 2]
  out[0] = x * b.r[0] + y * b.u[0] + z * b.f[0]
  out[1] = x * b.r[1] + y * b.u[1] + z * b.f[1]
  out[2] = x * b.r[2] + y * b.u[2] + z * b.f[2]
}

function gather(src: Float32Array, idx: Int32Array, channels: number, channel = -1): Float32Array {
  if (channel >= 0) {
    const out = new Float32Array(idx.length)
    for (let k = 0; k < idx.length; k += 1) out[k] = src[idx[k] * channels + channel]
    return out
  }
  const out = new Float32Array(idx.length * channels)
  for (let k = 0; k < idx.length; k += 1) {
    for (let c = 0; c < channels; c += 1) out[k * channels + c] = src[idx[k] * channels + c]
  }
  return out
}

// ------------------------------------------------------------- lattice cues ---

/** Pixels where the predicted lattice is resolvable (small, consistent phase steps on X and Z). */
export function latticeSmooth(tph: Float32Array, w: number, h: number, maxStep = 0.35, maxJitter = 0.06): Uint8Array {
  const ok = new Uint8Array(w * h)
  for (let y = 0; y < h - 2; y += 1) {
    for (let x = 0; x < w - 2; x += 1) {
      let good = 1
      for (const a of [0, 2]) {
        const p = (yy: number, xx: number) => tph[(yy * w + xx) * 3 + a]
        const sx0 = wrap(p(y, x + 1) - p(y, x))
        const sx1 = wrap(p(y, x + 2) - p(y, x + 1))
        const sy0 = wrap(p(y + 1, x) - p(y, x))
        const sy1 = wrap(p(y + 2, x) - p(y + 1, x))
        if (
          !(Math.abs(sx0) < maxStep && Math.abs(wrap(sx1 - sx0)) < maxJitter &&
            Math.abs(sy0) < maxStep && Math.abs(wrap(sy1 - sy0)) < maxJitter)
        ) {
          good = 0
          break
        }
      }
      ok[y * w + x] = good
    }
  }
  return ok
}

function neighbourPairs(mask: Uint8Array, w: number, h: number, maxPairs: number, seed: number): Int32Array {
  const found: number[] = []
  for (let y = 0; y < h - 1; y += 1) {
    for (let x = 0; x < w - 1; x += 1) {
      const p = y * w + x
      if (mask[p] && mask[p + 1] && mask[p + w]) found.push(p)
    }
  }
  const pick = sampleIndices(found.length, maxPairs, rng(seed))
  const out = new Int32Array(pick.length)
  for (let k = 0; k < pick.length; k += 1) out[k] = found[pick[k]]
  return out.sort()
}

// ---------------------------------------------------------------- pose search ---

function chunkSize(kernels: Kernels, perPose: number): number {
  return Math.max(1, Math.floor(kernels.budget / Math.max(1, perPose)))
}

interface Candidate {
  yaw: number
  pitch: number
  h: number
  loss: number
}

/** Brute force (yaw, pitch, h) against wrapped lattice phase steps; best distinct candidates. */
export async function gridSearchPose(
  kernels: Kernels,
  rays: Float32Array,
  phx: Float32Array,
  phz: Float32Array,
  pairs: Int32Array,
  sign: number,
  topK = 4,
  minSep = 6,
): Promise<Candidate[]> {
  let pick = Int32Array.from({ length: pairs.length / 2 }, (_, k) => k)
  if (pick.length > 20000) pick = sampleIndices(pick.length, 20000, rng(0))
  const m = pick.length
  const raysI = new Float32Array(m * 3)
  const raysJ = new Float32Array(m * 3)
  const obsX = new Float32Array(m)
  const obsZ = new Float32Array(m)
  for (let k = 0; k < m; k += 1) {
    const i = pairs[2 * pick[k]]
    const j = pairs[2 * pick[k] + 1]
    raysI.set(rays.subarray(3 * i, 3 * i + 3), 3 * k)
    raysJ.set(rays.subarray(3 * j, 3 * j + 3), 3 * k)
    obsX[k] = wrap(phx[j] - phx[i])
    obsZ[k] = wrap(phz[j] - phz[i])
  }
  const yaws = arange(-50, 50.01, 2.5)
  const pitches = arange(4, 90.01, 2).map((p) => p * sign)
  const hs = Float32Array.from(linspace(Math.log(0.6), Math.log(40), 48), (v) => sign * Math.exp(v))
  const combos = pitches.length * yaws.length
  const loss = new Float32Array(combos * hs.length)
  const step = chunkSize(kernels, hs.length * m)
  for (let c0 = 0; c0 < combos; c0 += step) {
    const k = Math.min(step, combos - c0)
    const yaw = new Float32Array(k)
    const pitch = new Float32Array(k)
    for (let c = 0; c < k; c += 1) {
      yaw[c] = yaws[(c0 + c) % yaws.length] * DEG
      pitch[c] = pitches[Math.floor((c0 + c) / yaws.length)] * DEG
    }
    loss.set(await kernels.pairScores({ raysI, raysJ, obsX, obsZ, yaw, pitch, hs, sign }), c0 * hs.length)
  }
  const order = Int32Array.from({ length: loss.length }, (_, i) => i).sort((a, b) => loss[a] - loss[b])
  const picked: Candidate[] = []
  for (const flat of order) {
    const hi = flat % hs.length
    const c = Math.floor(flat / hs.length)
    const cand = { yaw: yaws[c % yaws.length], pitch: pitches[Math.floor(c / yaws.length)], h: hs[hi], loss: loss[flat] }
    if (
      picked.every((q) =>
        Math.abs(cand.yaw - q.yaw) > minSep || Math.abs(cand.pitch - q.pitch) > minSep ||
        Math.abs(Math.log(Math.abs(cand.h / q.h))) > 0.3)
    ) {
      picked.push(cand)
    }
    if (picked.length === topK) break
  }
  return picked
}

/** Grid search maximising lattice phase coherence. */
export async function coherenceGrid(
  kernels: Kernels,
  rays: Float32Array,
  phx: Float32Array,
  phz: Float32Array,
  yawsDeg: number[],
  pitchesDeg: number[],
  hs: number[],
  sign: number,
  maxPoints = 15000,
): Promise<{ yaw: number; pitch: number; h: number; value: number }> {
  let n = phx.length
  if (n > maxPoints) {
    const sel = sampleIndices(n, maxPoints, rng(0))
    rays = gather(rays, sel, 3)
    phx = gather(phx, sel, 1)
    phz = gather(phz, sel, 1)
    n = maxPoints
  }
  const hsT = Float32Array.from(hs)
  const combos = pitchesDeg.length * yawsDeg.length
  const step = chunkSize(kernels, hs.length * n)
  let best = { yaw: 0, pitch: 0, h: 0, value: -Infinity }
  for (let c0 = 0; c0 < combos; c0 += step) {
    const k = Math.min(step, combos - c0)
    const yaw = new Float32Array(k)
    const pitch = new Float32Array(k)
    for (let c = 0; c < k; c += 1) {
      yaw[c] = yawsDeg[(c0 + c) % yawsDeg.length] * DEG
      pitch[c] = pitchesDeg[Math.floor((c0 + c) / yawsDeg.length)] * DEG
    }
    const coh = await kernels.coherence({ rays, phx, phz, yaw, pitch, hs: hsT, sign })
    for (let q = 0; q < coh.length; q += 1) {
      if (coh[q] > best.value) {
        const c = c0 + Math.floor(q / hs.length)
        best = {
          yaw: yawsDeg[c % yawsDeg.length],
          pitch: pitchesDeg[Math.floor(c / yawsDeg.length)],
          h: hs[q % hs.length],
          value: coh[q],
        }
      }
    }
  }
  return best
}

/** Lattice coherence of points at a plane `h` below (above, h < 0) the eye, and its phases. */
export function planeCoherence(
  yaw: number,
  pitch: number,
  h: number,
  rays: Float32Array,
  phx: Float32Array,
  phz: Float32Array,
): { kx: number; kz: number; cx: number; cz: number } {
  const b = basis(yaw, pitch)
  const d = [0, 0, 0]
  let rex = 0
  let imx = 0
  let rez = 0
  let imz = 0
  const n = phx.length
  for (let k = 0; k < n; k += 1) {
    worldDir(b, rays, 3 * k, d)
    const s = h * signedInv(-d[1], 1e-3)
    const ax = phx[k] - 16 * s * d[0]
    const az = phz[k] - 16 * s * d[2]
    rex += cosTurns(ax)
    imx += sinTurns(ax)
    rez += cosTurns(az)
    imz += sinTurns(az)
  }
  return {
    kx: Math.hypot(rex, imx) / n,
    kz: Math.hypot(rez, imz) / n,
    cx: mod(Math.atan2(imx, rex) / TAU, 1),
    cz: mod(Math.atan2(imz, rez) / TAU, 1),
  }
}

/** Polish (yaw, pitch, h) on lattice coherence; adds the camera's sub-texel offsets. */
export function refinePose(
  y0: number,
  p0: number,
  h0: number,
  rays: Float32Array,
  phx: Float32Array,
  phz: Float32Array,
): PoseFit {
  const objective = ([y, p, h]: number[]) => {
    const c = planeCoherence(y, p, h, rays, phx, phz)
    return -(c.kx + c.kz)
  }
  const { x } = nelderMead(objective, [y0, p0, h0], [0.04, 0.04, Math.abs(h0) * 0.001], { maxEvals: 300, tol: 1e-9 })
  const [yaw, pitch, h] = x
  const c = planeCoherence(yaw, pitch, h, rays, phx, phz)
  return { yaw, pitch, h, c16x: c.cx, c16z: c.cz, coherence: [c.kx, c.kz] }
}

/**
 * Pose candidates from floors (+1) and ceilings (-1), each refined on its own plane; the one
 * explaining the most floor + ceiling pixels wins. `floor`/`ceiling` are already smooth-masked.
 */
export async function fitPose(
  kernels: Kernels,
  screen: Screen,
  tph: Float32Array,
  floor: Uint8Array,
  ceiling: Uint8Array,
  perSign = 3,
): Promise<{ pose: PoseFit | null; tried: (Candidate & { sign: number })[] }> {
  const { width: w, height: h } = screen
  const refined: PoseFit[] = []
  const tried: (Candidate & { sign: number })[] = []
  for (const [sign, mask] of [[1, floor], [-1, ceiling]] as const) {
    const base = neighbourPairs(mask, w, h, 30000, 1)
    const n = base.length
    if (n < 200) continue
    const pixels = new Int32Array(3 * n)
    for (let k = 0; k < n; k += 1) {
      pixels[k] = base[k]
      pixels[n + k] = base[k] + 1
      pixels[2 * n + k] = base[k] + w
    }
    const rays = gatherRays(screen, pixels)
    const phx = gather(tph, pixels, 3, 0)
    const phz = gather(tph, pixels, 3, 2)
    const pairs = new Int32Array(4 * n)
    for (let k = 0; k < n; k += 1) {
      pairs[2 * k] = k
      pairs[2 * k + 1] = k + n
      pairs[2 * (n + k)] = k
      pairs[2 * (n + k) + 1] = k + 2 * n
    }
    const cands = await gridSearchPose(kernels, rays, phx, phz, pairs, sign, perSign)
    tried.push(...cands.map((c) => ({ ...c, sign })))
    const sub = sampleIndices(3 * n, 30000, rng(0))
    const subRays = gather(rays, sub, 3)
    const subX = gather(phx, sub, 1)
    const subZ = gather(phz, sub, 1)
    for (const cand of cands) {
      let { yaw: y0, pitch: p0, h: h0 } = cand
      // close-up views give flat phase-step scores, so the first window is wide
      for (const [half, steps, hSpan] of [[15, 21, 0.6], [3, 13, 0.12], [0.7, 11, 0.03], [0.15, 9, 0.006]]) {
        const best = await coherenceGrid(
          kernels, subRays, subX, subZ,
          linspace(y0 - half, y0 + half, steps),
          linspace(p0 - half, p0 + half, steps).map((p) => Math.min(90, Math.max(-90, p))),
          linspace(-hSpan, hSpan, 21).map((e) => h0 * Math.exp(e)),
          sign,
        )
        ;({ yaw: y0, pitch: p0, h: h0 } = best)
      }
      const p = refinePose(y0, p0, h0, subRays, subX, subZ)
      if (Math.abs(p.pitch - p0) > 2 || Math.abs(p.yaw - y0) > 2 || Math.abs(Math.log(Math.abs(p.h / h0))) > 0.1) continue
      refined.push(p)
    }
  }
  if (!refined.length) return { pose: null, tried }
  for (const p of refined) p.explained = pooledFitScore(p, screen, tph, floor, ceiling)
  return { pose: refined.reduce((a, b) => ((b.explained ?? 0) > (a.explained ?? 0) ? b : a)), tried }
}

// ------------------------------------------------------------------- levels ---

interface PlaneUnit {
  /** Pixel indices (within the sampled grid) that are masked and reach the plane. */
  pixels: Int32Array
  u: Float32Array
  v: Float32Array
  /** Valid mask over the whole grid. */
  m: Float32Array
}

/** Unit-height plane coordinates of masked pixels on a (possibly strided) pixel grid. */
function planeUnit(
  pose: Pose,
  screen: Screen,
  mask: Uint8Array,
  sign: number,
  stride: number,
): { unit: PlaneUnit; gw: number; gh: number } {
  const gw = Math.ceil(screen.width / stride)
  const gh = Math.ceil(screen.height / stride)
  const b = basis(pose.yaw, pose.pitch)
  const ray = new Float32Array(3)
  const d = [0, 0, 0]
  const m = new Float32Array(gw * gh)
  const pixels: number[] = []
  const u: number[] = []
  const v: number[] = []
  for (let gy = 0; gy < gh; gy += 1) {
    for (let gx = 0; gx < gw; gx += 1) {
      const x = gx * stride
      const y = gy * stride
      if (!mask[y * screen.width + x]) continue
      rayAt(screen, x, y, ray, 0)
      worldDir(b, ray, 0, d)
      const down = -d[1]
      if (!(down * sign > 0.02)) continue
      const inv = signedInv(down, 0.02)
      const g = gy * gw + gx
      m[g] = 1
      pixels.push(g)
      u.push(d[0] * inv)
      v.push(d[2] * inv)
    }
  }
  return { unit: { pixels: Int32Array.from(pixels), u: Float32Array.from(u), v: Float32Array.from(v), m }, gw, gh }
}

/**
 * Runs `visit(n, pooled)` for every plane level n (plane at eye height h + n) with the pooled
 * absolute-phase cost of each grid pixel.
 */
function levelCosts(
  pose: Pose,
  unit: PlaneUnit,
  tph: Float32Array,
  tphIndex: (g: number) => number,
  gw: number,
  gh: number,
  sign: number,
  span: number,
  win: number,
  visit: (n: number, pooled: Float32Array) => void,
): void {
  const r = Math.floor(win / 2)
  const den = boxSum(unit.m, gw, gh, r)
  const floorDen = 1e-3 * win * win
  for (let g = 0; g < den.length; g += 1) den[g] = Math.max(den[g], floorDen)
  const cost = new Float32Array(gw * gh)
  const pooled = new Float32Array(gw * gh)
  const t0 = new Float32Array(unit.pixels.length)
  const t2 = new Float32Array(unit.pixels.length)
  for (let k = 0; k < unit.pixels.length; k += 1) {
    const i = tphIndex(unit.pixels[k])
    t0[k] = tph[3 * i]
    t2[k] = tph[3 * i + 2]
  }
  for (let n = -span; n <= span; n += 1) {
    const hn = pose.h + n
    if (sign * hn <= 0.05) continue
    for (let k = 0; k < unit.pixels.length; k += 1) {
      cost[unit.pixels[k]] =
        cauchy(wrap(16 * hn * unit.u[k] + pose.c16x - t0[k]), 0.08) +
        cauchy(wrap(16 * hn * unit.v[k] + pose.c16z - t2[k]), 0.08)
    }
    boxSum(cost, gw, gh, r, pooled)
    for (let g = 0; g < pooled.length; g += 1) pooled[g] /= den[g]
    visit(n, pooled)
  }
}

/**
 * How many floor/ceiling pixels a pose explains, judged per neighbourhood: a pixel counts if one
 * integer plane level matches the absolute lattice phase consistently across its window.
 */
export function pooledFitScore(
  pose: Pose,
  screen: Screen,
  tph: Float32Array,
  floor: Uint8Array,
  ceiling: Uint8Array,
  span = 24,
  win = 15,
  thr = 1,
  stride = 2,
): number {
  const winS = Math.max(3, Math.floor(win / stride) | 1)
  let total = 0
  for (const [sign, mask] of [[1, floor], [-1, ceiling]] as const) {
    let count = 0
    for (let y = 0; y < screen.height; y += stride) {
      for (let x = 0; x < screen.width; x += stride) count += mask[y * screen.width + x]
    }
    if (count < 50) continue
    const { unit, gw, gh } = planeUnit(pose, screen, mask, sign, stride)
    const best = new Float32Array(gw * gh).fill(Infinity)
    const index = (g: number) => Math.floor(g / gw) * stride * screen.width + (g % gw) * stride
    levelCosts(pose, unit, tph, index, gw, gh, sign, span, winS, (_, pooled) => {
      for (let g = 0; g < best.length; g += 1) if (pooled[g] < best[g]) best[g] = pooled[g]
    })
    for (const g of unit.pixels) if (best[g] < thr) total += 1
  }
  return total
}

/** Windowed majority vote of integer labels (only where valid); -99999 elsewhere. */
export function majority(
  labels: Int32Array,
  valid: Uint8Array,
  w: number,
  h: number,
  smooth = 9,
  maxLabels = 32,
  minShare = 0.5,
): Int32Array {
  const out = new Int32Array(w * h).fill(-99999)
  const counts = new Map<number, number>()
  for (let p = 0; p < labels.length; p += 1) if (valid[p]) counts.set(labels[p], (counts.get(labels[p]) ?? 0) + 1)
  if (!counts.size) return out
  const vals = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, maxLabels).map(([v]) => v)
  const share = new Float32Array(w * h)
  const winner = new Int32Array(w * h).fill(-1)
  const total = new Float32Array(w * h)
  const onehot = new Float32Array(w * h)
  const votes = new Float32Array(w * h)
  const r = Math.floor(smooth / 2)
  vals.forEach((value, li) => {
    for (let p = 0; p < labels.length; p += 1) onehot[p] = valid[p] && labels[p] === value ? 1 : 0
    boxSum(onehot, w, h, r, votes)
    for (let p = 0; p < votes.length; p += 1) {
      total[p] += votes[p]
      if (votes[p] > share[p] || winner[p] < 0) {
        share[p] = votes[p]
        winner[p] = li
      }
    }
  })
  for (let p = 0; p < labels.length; p += 1) {
    if (valid[p] && share[p] / Math.max(total[p], 1e-6) > minShare && vals[winner[p]] === labels[p]) out[p] = labels[p]
  }
  return out
}

/** Integer level n (plane at eye height h + n) per horizontal pixel; -999 where unassigned. */
export function assignLevels(
  mask: Uint8Array,
  screen: Screen,
  tph: Float32Array,
  pose: Pose,
  sign: number,
  span = 24,
  smooth = 9,
  pool = 15,
): Int32Array {
  const { width: w, height: h } = screen
  const { unit } = planeUnit(pose, screen, mask, sign, 1)
  const bestCost = new Float32Array(w * h).fill(Infinity)
  const bestN = new Int32Array(w * h)
  let any = false
  levelCosts(pose, unit, tph, (g) => g, w, h, sign, span, pool, (n, pooled) => {
    any = true
    for (let g = 0; g < pooled.length; g += 1) {
      if (pooled[g] < bestCost[g]) {
        bestCost[g] = pooled[g]
        bestN[g] = n
      }
    }
  })
  if (!any) return new Int32Array(w * h).fill(-999)
  const m = new Uint8Array(w * h)
  for (const g of unit.pixels) m[g] = 1
  const voted = majority(bestN, m, w, h, smooth)
  for (let g = 0; g < voted.length; g += 1) if (voted[g] === -99999) voted[g] = -999
  return voted
}

/**
 * For pixels on walls facing +-axis (0 = x, 2 = z): the block boundary B (wall at grid
 * coordinate (16 B + oAxis) / 16) whose predicted absolute lattice phase they match.
 */
export function assignWalls(
  mask: Uint8Array,
  axis: 0 | 2,
  screen: Screen,
  tph: Float32Array,
  pos: Float32Array,
  pose: Pose,
  oAxis: number,
  smooth = 9,
  window = 5,
  pool = 15,
): Int32Array {
  const { width: w, height: h } = screen
  const other = axis === 0 ? 2 : 0
  const cAxis = axis === 0 ? pose.c16x : pose.c16z
  const cOther = axis === 0 ? pose.c16z : pose.c16x
  const b = basis(pose.yaw, pose.pitch)
  const ray = new Float32Array(3)
  const d = [0, 0, 0]
  const pixels: number[] = []
  for (let p = 0; p < w * h; p += 1) if (mask[p]) pixels.push(p)
  const n = pixels.length
  const coarse = new Float32Array(n)
  const dy = new Float32Array(n)
  const dq = new Float32Array(n)
  const inv = new Float32Array(n)
  for (let k = 0; k < n; k += 1) {
    const p = pixels[k]
    rayAt(screen, p % w, Math.floor(p / w), ray, 0)
    worldDir(b, ray, 0, d)
    coarse[k] = Math.round((16 * (pos[3 * p + axis] + cAxis / 16) - oAxis) / 16)
    dy[k] = d[1]
    dq[k] = d[other]
    inv[k] = signedInv(d[axis], 1e-3)
  }
  const m = new Float32Array(w * h)
  for (const p of pixels) m[p] = 1
  const r = Math.floor(pool / 2)
  const den = boxSum(m, w, h, r)
  const floorDen = 1e-3 * pool * pool
  const cost = new Float32Array(w * h)
  const pooled = new Float32Array(w * h)
  const bestCost = new Float32Array(w * h).fill(Infinity)
  const bestB = new Int32Array(w * h)
  for (let off = -window; off <= window; off += 1) {
    for (let k = 0; k < n; k += 1) {
      const p = pixels[k]
      const B = coarse[k] + off
      const t = ((16 * B + oAxis) / 16 - cAxis / 16) * inv[k]
      if (!(t > 0.05)) {
        cost[p] = 50
        continue
      }
      const phY = 16 * (t * dy[k] + pose.h)
      const phQ = 16 * (t * dq[k] + cOther / 16)
      cost[p] = cauchy(wrap(phY - tph[3 * p + 1]), 0.08) + cauchy(wrap(phQ - tph[3 * p + other]), 0.08)
    }
    boxSum(cost, w, h, r, pooled)
    for (let k = 0; k < n; k += 1) {
      const p = pixels[k]
      const c = pooled[p] / Math.max(den[p], floorDen)
      if (c < bestCost[p]) {
        bestCost[p] = c
        bestB[p] = coarse[k] + off
      }
    }
  }
  const valid = new Uint8Array(w * h)
  for (const p of pixels) if (bestCost[p] < 10) valid[p] = 1
  return majority(bestB, valid, w, h, smooth)
}

// ---------------------------------------------------------------- resampling ---

export interface Sampled {
  mi: number
  mj: number
  /** Luminance at texel centres, zero where not valid. */
  T: Float32Array
  M: Float32Array
  px: Float32Array
  py: Float32Array
}

/**
 * Luminance at the camera-relative texel-centre points origin + i * di + j * dj, masked by
 * membership, visibility and texel size.
 */
export function resample(
  lum: Float32Array,
  screen: Screen,
  member: Uint8Array,
  pose: Pose,
  origin: number[],
  di: number[],
  dj: number[],
  mi: number,
  mj: number,
  minTexelPx = PARAMS.minTexelPx,
): Sampled {
  const { width: W, height: H, t } = screen
  const b = basis(pose.yaw, pose.pitch)
  const size = mi * mj
  const px = new Float32Array(size)
  const py = new Float32Array(size)
  const inside = new Uint8Array(size)
  for (let i = 0; i < mi; i += 1) {
    for (let j = 0; j < mj; j += 1) {
      const p = [0, 1, 2].map((c) => origin[c] + i * di[c] + j * dj[c])
      const cx = p[0] * b.r[0] + p[1] * b.r[1] + p[2] * b.r[2]
      const cy = p[0] * b.u[0] + p[1] * b.u[1] + p[2] * b.u[2]
      const cz = p[0] * b.f[0] + p[1] * b.f[1] + p[2] * b.f[2]
      const z = Math.max(cz, 1e-4)
      const x = ((cx / z / ((t * W) / H) + 1) * W) / 2 - 0.5
      const y = ((1 - cy / z / t) * H) / 2 - 0.5
      const k = i * mj + j
      px[k] = x
      py[k] = y
      inside[k] = cz > 0.05 && x >= 0 && x <= W - 1 && y >= 0 && y <= H - 1 ? 1 : 0
    }
  }
  const T = new Float32Array(size)
  const M = new Float32Array(size)
  for (let i = 0; i < mi; i += 1) {
    for (let j = 0; j < mj; j += 1) {
      const k = i * mj + j
      if (!inside[k]) continue
      const a = Math.min(i, mi - 2) * mj + Math.min(j, mj - 2)
      const texPx = Math.min(
        Math.hypot(px[a + mj] - px[a], py[a + mj] - py[a]),
        Math.hypot(px[a + 1] - px[a], py[a + 1] - py[a]),
      )
      if (!(texPx >= minTexelPx)) continue
      const x = px[k]
      const y = py[k]
      if (!member[Math.round(y) * W + Math.round(x)]) continue
      const x0 = Math.floor(x)
      const y0 = Math.floor(y)
      const fx = x - x0
      const fy = y - y0
      const x1 = Math.min(x0 + 1, W - 1)
      const y1 = Math.min(y0 + 1, H - 1)
      T[k] =
        (1 - fy) * ((1 - fx) * lum[y0 * W + x0] + fx * lum[y0 * W + x1]) +
        fy * ((1 - fx) * lum[y1 * W + x0] + fx * lum[y1 * W + x1])
      M[k] = 1
    }
  }
  return { mi, mj, T, M, px, py }
}

/** Masked NCC of the 16x16 window at (a, b) with every template of a face; returns the count. */
export function windowNcc(s: Sampled, a: number, b: number, bank: FaceTemplates, out: Float64Array): number {
  let n = 0
  let s1 = 0
  let s2 = 0
  for (let i = 0; i < 16; i += 1) {
    for (let j = 0; j < 16; j += 1) {
      const k = (a + i) * s.mj + b + j
      n += s.M[k]
      s1 += s.M[k] * s.T[k]
      s2 += s.M[k] * s.T[k] * s.T[k]
    }
  }
  const nn = Math.max(n, 1)
  const floor = nn * 0.006 ** 2
  const varT = Math.max(s2 - (s1 * s1) / nn, 1e-8) + floor
  for (let q = 0; q < bank.count; q += 1) {
    let st = 0
    let stt = 0
    let sTt = 0
    const base = q * 256
    for (let i = 0; i < 16; i += 1) {
      for (let j = 0; j < 16; j += 1) {
        const k = (a + i) * s.mj + b + j
        const m = s.M[k]
        if (!m) continue
        const tv = bank.data[base + i * 16 + j]
        st += tv
        stt += tv * tv
        sTt += s.T[k] * tv
      }
    }
    const vart = Math.max(stt - (st * st) / nn, 1e-8) + floor
    out[q] = Math.min(1, Math.max(-1, (sTt - (s1 * st) / nn) / Math.sqrt(varT * vart)))
  }
  return n
}

/** Reads one plane at block windows (a0 + 16 i, b0 + 16 j) into matched blocks. */
function readBlocks(
  s: Sampled,
  a0: number,
  b0: number,
  bank: FaceTemplates,
  face: FaceName,
  cellOf: (i: number, j: number) => [number, number, number],
): MatchedBlock[] {
  const out: MatchedBlock[] = []
  const ncc = new Float64Array(bank.count)
  for (let i = 0; a0 + 16 * i + 16 <= s.mi; i += 1) {
    for (let j = 0; b0 + 16 * j + 16 <= s.mj; j += 1) {
      const n = windowNcc(s, a0 + 16 * i, b0 + 16 * j, bank, ncc)
      if (n < PARAMS.minValidTexels) continue
      let k = 0
      for (let q = 1; q < bank.count; q += 1) if (ncc[q] > ncc[k]) k = q
      const m = bank.meta[k]
      // runner-up: the same block's best template implying a different row (value or parity)
      let runner = -1
      const perValue = new Map<number, number>()
      for (let q = 0; q < bank.count; q += 1) {
        const o = bank.meta[q]
        if (o.block !== m.block) continue
        if (o.value !== m.value || o.parity !== m.parity) runner = Math.max(runner, ncc[q])
        if (o.parity === m.parity || o.parity < 0) perValue.set(o.value, Math.max(perValue.get(o.value) ?? -1, ncc[q]))
      }
      const a = Math.min(a0 + 16 * i + 8, s.mi - 1) * s.mj + Math.min(b0 + 16 * j + 8, s.mj - 1)
      out.push({
        cell: cellOf(i, j),
        face,
        kind: m.kind,
        value: m.value,
        parity: m.parity,
        texture: m.block,
        ncc: ncc[k],
        margin: m.kind !== 'static' ? ncc[k] - runner : 0,
        valid: n,
        centroid: [s.px[a], s.py[a]],
        scores: [...perValue.entries()]
          .map(([variant, score]) => ({ variant, score }))
          .sort((x, y) => y.score - x.score),
      })
    }
  }
  return out
}

// ----------------------------------------------------------------- pipeline ---

export interface MatchDebug {
  pose?: PoseFit | null
  grid?: [number, number]
  gridMargin?: number
  planes?: { face: FaceName; n: number; mi: number; mj: number }[]
  walls?: { face: FaceName; B: number; pixels: number }[]
  reason?: string
}

export function luminance(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Float32Array {
  const lum = new Float32Array(w * h)
  for (let p = 0; p < w * h; p += 1) {
    lum[p] = (0.2126 * rgba[4 * p] + 0.7152 * rgba[4 * p + 1] + 0.0722 * rgba[4 * p + 2]) / 255
  }
  return lum
}

export interface Masks {
  good: Uint8Array
  floor: Uint8Array
  ceiling: Uint8Array
  smooth: Uint8Array
}

export function predictionMasks(pred: DensePrediction): Masks {
  const n = pred.width * pred.height
  const good = new Uint8Array(n)
  const floor = new Uint8Array(n)
  const ceiling = new Uint8Array(n)
  for (let p = 0; p < n; p += 1) {
    good[p] = pred.faceConf[p] >= PARAMS.minFaceConf && pred.templated[p] >= PARAMS.minTemplated ? 1 : 0
    floor[p] = good[p] && pred.face[p] === FACE_CLASS.up ? 1 : 0
    ceiling[p] = good[p] && pred.face[p] === FACE_CLASS.down ? 1 : 0
  }
  return { good, floor, ceiling, smooth: latticeSmooth(pred.tphase, pred.width, pred.height) }
}

function and(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length)
  for (let p = 0; p < a.length; p += 1) out[p] = a[p] & b[p]
  return out
}

function countOf(a: Uint8Array): number {
  let n = 0
  for (let p = 0; p < a.length; p += 1) n += a[p]
  return n
}

function topLabels(labels: Int32Array, invalid: number, limit: number): { label: number; count: number }[] {
  const counts = new Map<number, number>()
  for (const l of labels) if (l !== invalid) counts.set(l, (counts.get(l) ?? 0) + 1)
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || a.label - b.label)
    .slice(0, limit)
}

/** Full block matching for a known FOV. */
export async function matchBlocks(
  lum: Float32Array,
  pred: DensePrediction,
  fov: number,
  bank: TemplateBank,
  kernels: Kernels,
  report: Report = () => {},
  masks: Masks = predictionMasks(pred),
): Promise<{ blocks: MatchedBlock[]; debug: MatchDebug }> {
  const { width: W, height: H } = pred
  const screen = screenFor(W, H, fov)
  const { good, floor, ceiling, smooth } = masks
  if (countOf(floor) + countOf(ceiling) < 500) {
    return { blocks: [], debug: { reason: 'too few floor/ceiling pixels of known blocks' } }
  }

  // 1-2. pose
  report('Fitting the camera pose')
  const { pose } = await fitPose(kernels, screen, pred.tphase, and(floor, smooth), and(ceiling, smooth))
  if (!pose || Math.max(...pose.coherence) < 0.3) {
    return { blocks: [], debug: { reason: 'lattice pose fit failed', pose } }
  }
  const h0 = pose.h
  const cx = pose.c16x / 16
  const cz = pose.c16z / 16
  const b = basis(pose.yaw, pose.pitch)
  const lim = PARAMS.maxExtentBlocks

  // 3. horizontal planes
  interface Plane {
    face: FaceName
    n: number
    m0x: number
    m0z: number
    s: Sampled
    best: Float32Array
    bi: number
    bj: number
  }
  const planes: Plane[] = []
  for (const [sign, mask, face] of [[1, floor, 'up'], [-1, ceiling, 'down']] as const) {
    if (countOf(mask) < PARAMS.minPlanePixels) continue
    report(`Assigning ${face === 'up' ? 'floor' : 'ceiling'} levels`)
    const level = assignLevels(mask, screen, pred.tphase, pose, sign)
    for (const { label: n, count } of topLabels(level, -999, PARAMS.maxLevels)) {
      if (count < PARAMS.minPlanePixels) continue
      const hn = h0 + n
      const member = new Uint8Array(W * H)
      let x0 = Infinity
      let x1 = -Infinity
      let z0 = Infinity
      let z1 = -Infinity
      const ray = new Float32Array(3)
      const d = [0, 0, 0]
      for (let p = 0; p < level.length; p += 1) {
        if (level[p] !== n) continue
        member[p] = 1
        rayAt(screen, p % W, Math.floor(p / W), ray, 0)
        worldDir(b, ray, 0, d)
        const s = hn * signedInv(-d[1], 1e-3)
        x0 = Math.min(x0, s * d[0])
        x1 = Math.max(x1, s * d[0])
        z0 = Math.min(z0, s * d[2])
        z1 = Math.max(z1, s * d[2])
      }
      const xl = [Math.max(x0 + cx, -lim), Math.min(x1 + cx, lim)]
      const zl = [Math.max(z0 + cz, -lim), Math.min(z1 + cz, lim)]
      if (xl[1] - xl[0] < 1 || zl[1] - zl[0] < 1) continue
      const m0x = Math.floor(xl[0] * 16)
      const m0z = Math.floor(zl[0] * 16)
      const mi = Math.ceil(xl[1] * 16) + 1 - m0x
      const mj = Math.ceil(zl[1] * 16) + 1 - m0z
      if (mi < 16 || mj < 16) continue
      report(`Matching textures on ${face === 'up' ? 'floor' : 'ceiling'} level ${n}`)
      const s = resample(lum, screen, member, pose,
        [(m0x + 0.5) / 16 - cx, -hn, (m0z + 0.5) / 16 - cz], [1 / 16, 0, 0], [0, 0, 1 / 16], mi, mj)
      const tm = bank[face]
      const { best, valid } = await kernels.nccBest(s.T, s.M, mi, mj, tm.data, tm.count)
      for (let k = 0; k < best.length; k += 1) if (valid[k] < PARAMS.minValidTexels) best[k] = 0
      planes.push({ face, n, m0x, m0z, s, best, bi: mi - 15, bj: mj - 15 })
    }
  }
  if (!planes.length) return { blocks: [], debug: { reason: 'no horizontal plane could be sampled', pose } }

  // block-grid alignment: texel offset (ox, oz) pooled over every horizontal plane
  const scores = new Float64Array(256)
  for (const pl of planes) {
    for (let ox = 0; ox < 16; ox += 1) {
      for (let oz = 0; oz < 16; oz += 1) {
        let sum = 0
        for (let i = mod(ox - pl.m0x, 16); i < pl.bi; i += 16) {
          for (let j = mod(oz - pl.m0z, 16); j < pl.bj; j += 16) sum += pl.best[i * pl.bj + j]
        }
        scores[ox * 16 + oz] += sum
      }
    }
  }
  let bestFlat = 0
  for (let k = 1; k < 256; k += 1) if (scores[k] > scores[bestFlat]) bestFlat = k
  const ox = Math.floor(bestFlat / 16)
  const oz = bestFlat % 16
  const sorted = [...scores].sort((a, c) => c - a)

  report('Reading block rotations')
  const blocks: MatchedBlock[] = []
  for (const pl of planes) {
    const a0 = mod(ox - pl.m0x, 16)
    const b0 = mod(oz - pl.m0z, 16)
    const cy = pl.face === 'up' ? -pl.n - 1 : -pl.n
    blocks.push(...readBlocks(pl.s, a0, b0, bank[pl.face], pl.face, (i, j) => [
      Math.floor((pl.m0x + a0 + 16 * i - ox) / 16), cy, Math.floor((pl.m0z + b0 + 16 * j - oz) / 16),
    ]))
  }

  // 4. walls
  const walls: { face: FaceName; B: number; pixels: number }[] = []
  const wallSpecs: [0 | 2, number, FaceName][] = [
    [0, FACE_CLASS.east, 'east'], [0, FACE_CLASS.west, 'west'], [2, FACE_CLASS.south, 'south'], [2, FACE_CLASS.north, 'north'],
  ]
  for (const [axis, cls, face] of wallSpecs) {
    const mask = new Uint8Array(W * H)
    for (let p = 0; p < mask.length; p += 1) mask[p] = good[p] && pred.face[p] === cls ? 1 : 0
    if (countOf(mask) < PARAMS.minPlanePixels) continue
    report(`Matching ${face} walls`)
    const oAxis = axis === 0 ? ox : oz
    const bmap = assignWalls(mask, axis, screen, pred.tphase, pred.pos, pose, oAxis)
    const other = axis === 0 ? 2 : 0
    const cAxis = axis === 0 ? cx : cz
    const cOther = axis === 0 ? cz : cx
    const oOther = axis === 0 ? oz : ox
    const tm = bank[face]
    for (const { label: B, count } of topLabels(bmap, -99999, PARAMS.maxWalls)) {
      if (count < PARAMS.minPlanePixels) continue
      const member = new Uint8Array(W * H)
      const xRel = (16 * B + oAxis) / 16 - cAxis
      let q0 = Infinity
      let q1 = -Infinity
      let y0 = Infinity
      let y1 = -Infinity
      const ray = new Float32Array(3)
      const d = [0, 0, 0]
      for (let p = 0; p < bmap.length; p += 1) {
        if (bmap[p] !== B) continue
        member[p] = 1
        rayAt(screen, p % W, Math.floor(p / W), ray, 0)
        worldDir(b, ray, 0, d)
        const t = xRel * signedInv(d[axis], 1e-3)
        const q = t * d[other] + cOther
        const yg = t * d[1] + h0
        q0 = Math.min(q0, q)
        q1 = Math.max(q1, q)
        y0 = Math.min(y0, yg)
        y1 = Math.max(y1, yg)
      }
      const ql = [Math.max(q0, -lim), Math.min(q1, lim)]
      const yl = [Math.max(y0, -lim), Math.min(y1, lim)]
      if (ql[1] - ql[0] < 1 || yl[1] - yl[0] < 1) continue
      const m0q = Math.floor(ql[0] * 16)
      const m0y = Math.floor(yl[0] * 16)
      const mi = Math.ceil(ql[1] * 16) + 1 - m0q
      const mj = Math.ceil(yl[1] * 16) + 1 - m0y
      if (mi < 16 || mj < 16) continue
      const origin = [0, (m0y + 0.5) / 16 - h0, 0]
      origin[axis] = xRel
      origin[other] = (m0q + 0.5) / 16 - cOther
      const di = [0, 0, 0]
      di[other] = 1 / 16
      const s = resample(lum, screen, member, pose, origin, di, [0, 1 / 16, 0], mi, mj)
      const a0 = mod(oOther - m0q, 16)
      const b0 = mod(-m0y, 16)
      const cellAxis = face === 'east' || face === 'south' ? B - 1 : B
      walls.push({ face, B, pixels: count })
      blocks.push(...readBlocks(s, a0, b0, tm, face, (i, j) => {
        const cq = Math.floor((m0q + a0 + 16 * i - oOther) / 16)
        const cyy = Math.floor((m0y + b0 + 16 * j) / 16)
        return axis === 0 ? [cellAxis, cyy, cq] : [cq, cyy, cellAxis]
      }))
    }
  }

  return {
    blocks,
    debug: {
      pose,
      grid: [ox, oz],
      gridMargin: sorted[0] - sorted[1],
      planes: planes.map((p) => ({ face: p.face, n: p.n, mi: p.s.mi, mj: p.s.mj })),
      walls,
    },
  }
}

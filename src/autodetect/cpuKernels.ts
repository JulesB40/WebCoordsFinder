/**
 * Plain TypeScript versions of the pose-search kernels: the same maths as
 * kernels/pair_scores.onnx and kernels/coherence.onnx (texrot/kernels.py), written as tight
 * float64 loops. Phases are in turns, so sin/cos reduce to a rounding plus short polynomials,
 * and Gaussian weights that would underflow are skipped. They run on the CPU on every machine:
 * GPU sin/cos lose accuracy at the thousands of radians these phases reach.
 */
import type { Kernels } from './kernels'

// Quarter-turn rotation tables for the octant-reduced sin/cos.
const QS = Float64Array.of(0, 1, 0, -1)
const QC = Float64Array.of(1, 0, -1, 0)

/** sin(2 pi x) for any x, |error| < 1e-7. */
export function sinTurns(x: number): number {
  const f4 = 4 * (x - Math.floor(x))
  const q = Math.floor(f4 + 0.5)
  const r = 1.5707963267948966 * (f4 - q)
  const r2 = r * r
  const s = r * (1 + r2 * (-1 / 6 + r2 * (1 / 120 + r2 * (-1 / 5040 + r2 / 362880))))
  const c = 1 + r2 * (-0.5 + r2 * (1 / 24 + r2 * (-1 / 720 + r2 / 40320)))
  const qi = q & 3
  return QS[qi] * c + QC[qi] * s
}

export function cosTurns(x: number): number {
  const f4 = 4 * (x - Math.floor(x))
  const q = Math.floor(f4 + 0.5)
  const r = 1.5707963267948966 * (f4 - q)
  const r2 = r * r
  const s = r * (1 + r2 * (-1 / 6 + r2 * (1 / 120 + r2 * (-1 / 5040 + r2 / 362880))))
  const c = 1 + r2 * (-0.5 + r2 * (1 / 24 + r2 * (-1 / 720 + r2 / 40320)))
  const qi = q & 3
  return QC[qi] * c - QS[qi] * s
}

/**
 * |mean exp(2 pi i a_k)| for a_k = ph[k] - h * g[k], k < n: the coherence of one lattice axis.
 * The hot loop of the pose search.
 */
export function phaseCoherence(ph: Float64Array, g: Float64Array, h: number, n: number): number {
  let re = 0
  let im = 0
  for (let k = 0; k < n; k += 1) {
    const a = ph[k] - h * g[k]
    const f4 = 4 * (a - Math.floor(a))
    const q = Math.floor(f4 + 0.5)
    const r = 1.5707963267948966 * (f4 - q)
    const r2 = r * r
    const s = r * (1 + r2 * (-1 / 6 + r2 * (1 / 120 + r2 * (-1 / 5040 + r2 / 362880))))
    const c = 1 + r2 * (-0.5 + r2 * (1 / 24 + r2 * (-1 / 720 + r2 / 40320)))
    const qi = q & 3
    re += QC[qi] * c - QS[qi] * s
    im += QS[qi] * c + QC[qi] * s
  }
  return Math.sqrt(re * re + im * im)
}

/** Unit-plane coordinates (x, z per unit eye height) and validity for one pose. */
function unitPlane(
  yaw: number,
  pitch: number,
  rays: Float32Array,
  sign: number,
  u: Float64Array,
  v: Float64Array,
  ok: Uint8Array,
): number {
  const sy = Math.sin(yaw)
  const cy = Math.cos(yaw)
  const sp = Math.sin(pitch)
  const cp = Math.cos(pitch)
  // f = (-sy cp, -sp, cy cp), r = (-cy, 0, -sy), u = r x f
  const fx = -sy * cp
  const fy = -sp
  const fz = cy * cp
  const rx = -cy
  const rz = -sy
  const ux = -sy * sp
  const uy = cp
  const uz = cy * sp
  let valid = 0
  const n = u.length
  for (let k = 0; k < n; k += 1) {
    const a = rays[3 * k]
    const b = rays[3 * k + 1]
    const c = rays[3 * k + 2]
    const dx = a * rx + b * ux + c * fx
    const dy = b * uy + c * fy
    const dz = a * rz + b * uz + c * fz
    const down = -dy
    const den = Math.abs(down) < 0.02 ? (down + 1e-12 >= 0 ? 0.02 : -0.02) : down
    u[k] = dx / den
    v[k] = dz / den
    const good = down * sign > 0.02 ? 1 : 0
    ok[k] = good
    valid += good
  }
  return valid
}

export type PairScoresInput = Parameters<Kernels['pairScores']>[0]
export type CoherenceInput = Parameters<Kernels['coherence']>[0]

export function cpuPairScores(input: PairScoresInput): Float32Array {
  const { raysI, raysJ, obsX, obsZ, yaw, pitch, hs, sign } = input
  const m = obsX.length
  const K = yaw.length
  const Hh = hs.length
  const out = new Float32Array(K * Hh)
  const ui = new Float64Array(m)
  const vi = new Float64Array(m)
  const oki = new Uint8Array(m)
  const uj = new Float64Array(m)
  const vj = new Float64Array(m)
  const okj = new Uint8Array(m)
  const gx = new Float64Array(m)
  const gz = new Float64Array(m)
  const ox = new Float64Array(m)
  const oz = new Float64Array(m)
  const scale = 1 / (2 * 0.03 * 0.03)
  // exp(-r2 * scale) < 1e-12 beyond this: nothing at float32 precision
  const cutoff = 27.7 / scale
  for (let q = 0; q < K; q += 1) {
    unitPlane(yaw[q], pitch[q], raysI, sign, ui, vi, oki)
    unitPlane(yaw[q], pitch[q], raysJ, sign, uj, vj, okj)
    let nValid = 0
    for (let k = 0; k < m; k += 1) {
      if (!(oki[k] && okj[k])) continue
      gx[nValid] = 16 * (uj[k] - ui[k])
      gz[nValid] = 16 * (vj[k] - vi[k])
      ox[nValid] = obsX[k]
      oz[nValid] = obsZ[k]
      nValid += 1
    }
    const penalty = nValid / m < 0.5 ? 100 : 0
    for (let h = 0; h < Hh; h += 1) {
      const hh = hs[h]
      let sum = 0
      for (let t = 0; t < nValid; t += 1) {
        let rx = ox[t] - hh * gx[t]
        rx -= Math.floor(rx + 0.5)
        const r2x = rx * rx
        if (r2x < cutoff) {
          let rz = oz[t] - hh * gz[t]
          rz -= Math.floor(rz + 0.5)
          const r2 = r2x + rz * rz
          if (r2 < cutoff) sum += Math.exp(-r2 * scale)
        }
      }
      out[q * Hh + h] = 1 - sum / Math.max(nValid, 1) + penalty
    }
  }
  return out
}

export function cpuCoherence(input: CoherenceInput): Float32Array {
  const { rays, phx, phz, yaw, pitch, hs, sign } = input
  const n = phx.length
  const K = yaw.length
  const Hh = hs.length
  const out = new Float32Array(K * Hh)
  const u = new Float64Array(n)
  const v = new Float64Array(n)
  const ok = new Uint8Array(n)
  const cu = new Float64Array(n)
  const cv = new Float64Array(n)
  const px = new Float64Array(n)
  const pz = new Float64Array(n)
  for (let q = 0; q < K; q += 1) {
    const valid = unitPlane(yaw[q], pitch[q], rays, sign, u, v, ok)
    let m = 0
    for (let k = 0; k < n; k += 1) {
      if (!ok[k]) continue
      cu[m] = 16 * u[k]
      cv[m] = 16 * v[k]
      px[m] = phx[k]
      pz[m] = phz[k]
      m += 1
    }
    const norm = 1 / Math.max(valid, 1)
    for (let h = 0; h < Hh; h += 1) {
      out[q * Hh + h] = (phaseCoherence(px, cu, hs[h], m) + phaseCoherence(pz, cv, hs[h], m)) * norm
    }
  }
  return out
}

/** Splits the poses of a kernel call into `parts` contiguous ranges. */
export function splitPoses<T extends { yaw: Float32Array; pitch: Float32Array }>(input: T, parts: number): T[] {
  const K = input.yaw.length
  const size = Math.ceil(K / Math.max(1, parts))
  const out: T[] = []
  for (let a = 0; a < K; a += size) {
    out.push({ ...input, yaw: input.yaw.slice(a, a + size), pitch: input.pitch.slice(a, a + size) })
  }
  return out
}

export interface PoseKernelRunner {
  pairScores(input: PairScoresInput): Promise<Float32Array>
  coherence(input: CoherenceInput): Promise<Float32Array>
}

export const inThreadPoseKernels: PoseKernelRunner = {
  pairScores: async (input) => cpuPairScores(input),
  coherence: async (input) => cpuCoherence(input),
}

/** Kernels with the pose search on the CPU (`runner`) and NCC from `base` (ONNX). */
export function withCpuPoseKernels(base: Kernels, runner: PoseKernelRunner = inThreadPoseKernels): Kernels {
  return {
    // nothing of size [K, Hh, N] is ever materialised: whole grids per call
    budget: Number.MAX_SAFE_INTEGER,
    pairScores: (input) => runner.pairScores(input),
    coherence: (input) => runner.coherence(input),
    nccBest: (...args) => base.nccBest(...args),
  }
}

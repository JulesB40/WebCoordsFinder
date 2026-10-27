/** Small numeric helpers shared by the detector stages (CPU side). */

export const TAU = 2 * Math.PI
export const DEG = Math.PI / 180

/** Signed distance to the nearest integer, as torch `x - round(x)`. */
export function wrap(x: number): number {
  return x - Math.round(x)
}

export function cauchy(r: number, c: number): number {
  const q = r / c
  return Math.log1p(q * q)
}

export function signedInv(den: number, eps: number): number {
  if (Math.abs(den) < eps) return 1 / (den + 1e-12 >= 0 ? eps : -eps)
  return 1 / den
}

/** Python-style floor modulo. */
export function mod(a: number, n: number): number {
  return ((a % n) + n) % n
}

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** `count` distinct indices of [0, n) (all of them, in order, when n <= count). */
export function sampleIndices(n: number, count: number, random: () => number): Int32Array<ArrayBuffer> {
  if (n <= count) return Int32Array.from({ length: n }, (_, i) => i)
  const pool = new Int32Array(n)
  for (let i = 0; i < n; i += 1) pool[i] = i
  for (let i = 0; i < count; i += 1) {
    const j = i + Math.floor(random() * (n - i))
    const t = pool[i]
    pool[i] = pool[j]
    pool[j] = t
  }
  return pool.slice(0, count)
}

export function linspace(a: number, b: number, n: number): number[] {
  if (n === 1) return [a]
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
}

export function arange(a: number, b: number, step: number): number[] {
  const out: number[] = []
  for (let i = 0; a + i * step < b; i += 1) out.push(a + i * step)
  return out
}

/**
 * Window sums of a [h, w] map over a (2r+1)^2 window clipped to the image: the numerator or
 * denominator of torch's avg_pool2d(stride 1, padding r) up to the constant window size.
 */
export function boxSum(src: Float32Array, w: number, h: number, r: number, out?: Float32Array): Float32Array {
  const W1 = w + 1
  const integral = new Float64Array(W1 * (h + 1))
  for (let y = 0; y < h; y += 1) {
    let row = 0
    const base = (y + 1) * W1
    const prev = y * W1
    for (let x = 0; x < w; x += 1) {
      row += src[y * w + x]
      integral[base + x + 1] = integral[prev + x + 1] + row
    }
  }
  const result = out ?? new Float32Array(w * h)
  for (let y = 0; y < h; y += 1) {
    const y0 = Math.max(0, y - r)
    const y1 = Math.min(h, y + r + 1)
    for (let x = 0; x < w; x += 1) {
      const x0 = Math.max(0, x - r)
      const x1 = Math.min(w, x + r + 1)
      result[y * w + x] = integral[y1 * W1 + x1] - integral[y0 * W1 + x1] - integral[y1 * W1 + x0] + integral[y0 * W1 + x0]
    }
  }
  return result
}

/** Nelder–Mead minimisation (small problems: pose polish). */
export function nelderMead(
  f: (x: number[]) => number,
  x0: number[],
  steps: number[],
  { maxEvals = 400, tol = 1e-7 }: { maxEvals?: number; tol?: number } = {},
): { x: number[]; value: number } {
  const n = x0.length
  let simplex = [x0, ...steps.map((s, i) => x0.map((v, j) => (j === i ? v + s : v)))]
  let values = simplex.map(f)
  let evals = n + 1
  while (evals < maxEvals) {
    const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b])
    simplex = order.map((i) => simplex[i])
    values = order.map((i) => values[i])
    if (Math.abs(values[n] - values[0]) < tol) break
    const centroid = x0.map((_, j) => simplex.slice(0, n).reduce((s, p) => s + p[j], 0) / n)
    const at = (t: number) => centroid.map((c, j) => c + t * (simplex[n][j] - c))
    const xr = at(-1)
    const fr = f(xr)
    evals += 1
    if (fr < values[0]) {
      const xe = at(-2)
      const fe = f(xe)
      evals += 1
      ;[simplex[n], values[n]] = fe < fr ? [xe, fe] : [xr, fr]
    } else if (fr < values[n - 1]) {
      simplex[n] = xr
      values[n] = fr
    } else {
      const xc = fr < values[n] ? at(-0.5) : at(0.5)
      const fc = f(xc)
      evals += 1
      if (fc < Math.min(fr, values[n])) {
        simplex[n] = xc
        values[n] = fc
      } else {
        for (let i = 1; i <= n; i += 1) {
          simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]))
          values[i] = f(simplex[i])
        }
        evals += n
      }
    }
  }
  let best = 0
  for (let i = 1; i <= n; i += 1) if (values[i] < values[best]) best = i
  return { x: simplex[best], value: values[best] }
}

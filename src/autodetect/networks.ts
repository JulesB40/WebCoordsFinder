/**
 * Runs TexRotation's lattice and geometry networks (ONNX, fp32 compute) on a screenshot.
 * Input: RGB - 0.5 and per-pixel camera rays for the given FOV, replicate-padded to multiples
 * of 32. Large screenshots run in overlapping tiles: the networks were trained on 320 px crops
 * (rays carry the position in the frame), tiles bound memory and keep each GPU dispatch short,
 * and only each tile's centre is kept.
 */
import type * as OrtModule from 'onnxruntime-web'
import type { DensePrediction } from './geometry'
import { screenFor, rayAt } from './geometry'
import type { Ort, SessionFactory } from './kernels'
import { TAU, mod } from './numeric'

export interface ImagePixels {
  width: number
  height: number
  /** RGBA, as ImageData. */
  data: Uint8ClampedArray | Uint8Array
}

export const MULTIPLE = 32
export const TILE = 1024
export const TILE_MARGIN = 64

export function networkInput(image: ImagePixels, fov: number): { data: Float32Array; hp: number; wp: number } {
  const { width: w, height: h } = image
  const wp = Math.ceil(w / MULTIPLE) * MULTIPLE
  const hp = Math.ceil(h / MULTIPLE) * MULTIPLE
  const plane = hp * wp
  const data = new Float32Array(6 * plane)
  const screen = screenFor(w, h, fov)
  const ray = new Float32Array(3)
  for (let y = 0; y < hp; y += 1) {
    const sy = Math.min(y, h - 1)
    for (let x = 0; x < wp; x += 1) {
      const sx = Math.min(x, w - 1)
      const src = 4 * (sy * w + sx)
      const dst = y * wp + x
      data[dst] = image.data[src] / 255 - 0.5
      data[plane + dst] = image.data[src + 1] / 255 - 0.5
      data[2 * plane + dst] = image.data[src + 2] / 255 - 0.5
      rayAt(screen, sx, sy, ray, 0)
      data[3 * plane + dst] = ray[0]
      data[4 * plane + dst] = ray[1]
      data[5 * plane + dst] = ray[2]
    }
  }
  return { data, hp, wp }
}

async function read(tensor: OrtModule.Tensor): Promise<Float32Array> {
  const data = tensor.location === 'cpu' ? tensor.data : await tensor.getData(true)
  return data as Float32Array
}

export interface TileWindow {
  /** Window fed to the network (size multiple of 32). */
  x0: number
  y0: number
  w: number
  h: number
  /** Part of the window whose outputs are kept. */
  keepX0: number
  keepY0: number
  keepX1: number
  keepY1: number
}

/** Overlapping windows covering a [hp, wp] input; kept parts tile it exactly. */
export function tileWindows(hp: number, wp: number, tile = TILE, margin = TILE_MARGIN): TileWindow[] {
  const axis = (size: number) => {
    if (size <= tile) return [{ start: 0, size, keep0: 0, keep1: size }]
    const core = tile - 2 * margin
    const out: { start: number; size: number; keep0: number; keep1: number }[] = []
    for (let k0 = 0; k0 < size; k0 += core) {
      const k1 = Math.min(size, k0 + core)
      const start = Math.min(Math.max(0, k0 - margin), size - tile)
      out.push({ start, size: tile, keep0: k0, keep1: k1 })
    }
    return out
  }
  const windows: TileWindow[] = []
  for (const ty of axis(hp)) {
    for (const tx of axis(wp)) {
      windows.push({
        x0: tx.start, y0: ty.start, w: tx.size, h: ty.size,
        keepX0: tx.keep0, keepY0: ty.keep0, keepX1: tx.keep1, keepY1: ty.keep1,
      })
    }
  }
  return windows
}

function crop(data: Float32Array, hp: number, wp: number, t: TileWindow): Float32Array {
  const out = new Float32Array(6 * t.h * t.w)
  for (let c = 0; c < 6; c += 1) {
    for (let y = 0; y < t.h; y += 1) {
      const src = c * hp * wp + (t.y0 + y) * wp + t.x0
      out.set(data.subarray(src, src + t.w), (c * t.h + y) * t.w)
    }
  }
  return out
}

export async function predictDense(
  ort: Ort,
  session: SessionFactory,
  image: ImagePixels,
  fov: number,
  progress: (done: number, total: number) => void = () => {},
  tile = TILE,
): Promise<DensePrediction> {
  const { width: w, height: h } = image
  const { data, hp, wp } = networkInput(image, fov)
  const n = w * h
  const face = new Uint8Array(n)
  const faceConf = new Float32Array(n)
  const templated = new Float32Array(n)
  const pos = new Float32Array(n * 3)
  const tphase = new Float32Array(n * 3)
  const geometry = await session('geometry')
  const lattice = await session('lattice')
  const windows = tileWindows(hp, wp, tile)
  for (const [i, t] of windows.entries()) {
    const x = new ort.Tensor('float32', windows.length === 1 ? data : crop(data, hp, wp, t), [1, 6, t.h, t.w])
    const g = await geometry.run({ x })
    const faceProb = await read(g.face_prob)
    const templatedP = await read(g.templated)
    const posP = await read(g.pos)
    const l = await lattice.run({ x })
    const tp = await read(l.tphase)
    const plane = t.h * t.w
    for (let y = t.keepY0; y < Math.min(t.keepY1, h); y += 1) {
      for (let xx = t.keepX0; xx < Math.min(t.keepX1, w); xx += 1) {
        const q = (y - t.y0) * t.w + (xx - t.x0)
        const p = y * w + xx
        let best = 0
        for (let c = 1; c < 7; c += 1) if (faceProb[c * plane + q] > faceProb[best * plane + q]) best = c
        face[p] = best
        faceConf[p] = faceProb[best * plane + q]
        templated[p] = templatedP[q]
        for (let c = 0; c < 3; c += 1) {
          pos[3 * p + c] = posP[c * plane + q]
          tphase[3 * p + c] = mod(Math.atan2(tp[(2 * c + 1) * plane + q], tp[2 * c * plane + q]) / TAU, 1)
        }
      }
    }
    progress(i + 1, windows.length)
  }
  return { width: w, height: h, face, faceConf, templated, pos, tphase }
}

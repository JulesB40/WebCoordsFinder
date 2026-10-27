/**
 * Screenshot -> Detection: networks, automatic FOV, pose, block matching and rows. Everything
 * runs locally (onnxruntime-web); the caller provides the ORT module and a session factory.
 */
import { estimateFov } from './fov'
import {
  type DensePrediction,
  type MatchDebug,
  type Report,
  type TemplateBank,
  type TemplateKind,
  luminance,
  matchBlocks,
  predictionMasks,
} from './geometry'
import type { Kernels, Ort, SessionFactory } from './kernels'
import { type ImagePixels, predictDense } from './networks'
import { chooseOrigin, resolveFrame, selectRows } from './rows'
import type { Detection, FaceName } from './types'

/** Minecraft's default; the networks see rays for it until the FOV is known. */
export const NOMINAL_FOV = 70

export const FACE_NAMES: FaceName[] = ['up', 'down', 'north', 'south', 'east', 'west']

export interface TemplateIndex {
  size: number
  faces: Record<FaceName, { offset: number; count: number; meta: [string, TemplateKind, number, number][] }>
}

export function parseTemplates(index: TemplateIndex, data: ArrayBuffer): TemplateBank {
  const bank = {} as TemplateBank
  for (const face of FACE_NAMES) {
    const f = index.faces[face]
    bank[face] = {
      data: new Float32Array(data, f.offset, f.count * 256),
      count: f.count,
      meta: f.meta.map(([block, kind, value, parity]) => ({ block, kind, value, parity })),
    }
  }
  return bank
}

export interface DetectOptions {
  /** Vertical FOV in degrees; estimated from the lattice when omitted. */
  fov?: number
  report?: Report
}

export type DetectResult =
  | { ok: true; detection: Detection; fovEstimated: boolean; wellPosed: boolean; debug: MatchDebug }
  | { ok: false; reason: string; debug?: MatchDebug }

export interface DetectorRuntime {
  ort: Ort
  session: SessionFactory
  kernels: Kernels
  templates: TemplateBank
}

/** Detection from dense network outputs at a known FOV (rows not yet frame-resolved). */
export async function detectFromPrediction(
  runtime: Pick<DetectorRuntime, 'kernels' | 'templates'>,
  image: ImagePixels,
  pred: DensePrediction,
  fov: number,
  report: Report = () => {},
): Promise<DetectResult> {
  const lum = luminance(image.data, image.width, image.height)
  const { blocks, debug } = await matchBlocks(lum, pred, fov, runtime.templates, runtime.kernels, report)
  if (!debug.pose || !debug.grid) return { ok: false, reason: debug.reason ?? 'no pose', debug }
  const selected = selectRows(blocks)
  if (!selected.length) return { ok: false, reason: 'no block face was read confidently', debug }
  const { rows, parity } = resolveFrame(selected)
  const origin = chooseOrigin(rows, image.width, image.height)
  const { yaw, pitch, h, c16x, c16z } = debug.pose
  return {
    ok: true,
    fovEstimated: false,
    wellPosed: true,
    debug,
    detection: {
      width: image.width,
      height: image.height,
      fov,
      pose: { yaw, pitch, h, c16x, c16z },
      grid: { ox: debug.grid[0], oz: debug.grid[1] },
      parity,
      rows,
      origin: origin.cell,
    },
  }
}

export async function detect(runtime: DetectorRuntime, image: ImagePixels, options: DetectOptions = {}): Promise<DetectResult> {
  const report = options.report ?? (() => {})
  let fov = options.fov
  let wellPosed = true
  const tiles = (done: number, total: number) => report(`Running the networks (${done}/${total})`)
  report('Running the networks')
  let pred = await predictDense(runtime.ort, runtime.session, image, fov ?? NOMINAL_FOV, tiles)
  if (fov === undefined) {
    const estimate = await estimateFov(runtime.kernels, image.width, image.height, pred.tphase, predictionMasks(pred), report)
    fov = estimate?.fov ?? NOMINAL_FOV
    wellPosed = estimate?.wellPosed ?? false
    if (Math.abs(fov - NOMINAL_FOV) > 2) {
      // the networks take per-pixel rays: give them the right ones
      report('Running the networks at the estimated field of view')
      pred = await predictDense(runtime.ort, runtime.session, image, fov, tiles)
    }
  }
  const result = await detectFromPrediction(runtime, image, pred, fov, report)
  if (!result.ok) return result
  return { ...result, fovEstimated: options.fov === undefined, wellPosed }
}

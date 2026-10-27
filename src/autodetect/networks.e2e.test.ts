// @vitest-environment node
/**
 * Screenshot -> rows entirely in TypeScript (networks in onnxruntime-web WASM, tiled as in the
 * browser), compared with TexRotation's Python pipeline on real screenshots. Needs the dumps of
 * export_pred.py: set TEXROT_PRED_DIR (e.g. ../out/pred); TEXROT_TILE overrides the tile size.
 * Writes each config to TEXROT_PRED_DIR for a native CoordsFinder search.
 */
import { existsSync, readFileSync, readFileSync as read, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import * as ort from 'onnxruntime-web'
import { describe, expect, it } from 'vitest'
import { generateCoordsFinderConfig } from '../domain/exportConfig'
import { createEmptyDocument } from '../store/editorStore'
import { withCpuPoseKernels } from './cpuKernels'
import { buildDetectedDocument } from './document'
import { createOrtKernels } from './kernels'
import { TILE, predictDense } from './networks'
import { detectFromPrediction, parseTemplates, type TemplateIndex } from './pipeline'
import type { Detection } from './types'

function configRows(detection: Detection): Set<string> {
  const parts = buildDetectedDocument(detection)
  const document = createEmptyDocument()
  document.scene = parts.scene
  document.evidence = parts.evidence.map((e) => ({ ...e, reviewStatus: 'confirmed' }))
  document.anchorFaceId = parts.anchorFaceId
  const text = generateCoordsFinderConfig(document)
  return new Set(text.split('[filter]')[1].split('\n').map((l) => l.split('#')[0].trim()).filter(Boolean))
}

const dir = process.env.TEXROT_PRED_DIR
const tile = Number(process.env.TEXROT_TILE ?? TILE)

describe.skipIf(!dir || !existsSync(join(dir ?? '', 'cave.json')))(`screenshot to rows (networks in ${tile} px tiles, WASM)`, () => {
  it.each(['cave', 'forest'])('%s', async (name) => {
    const meta = JSON.parse(readFileSync(join(dir!, `${name}.json`), 'utf8'))
    const buf = readFileSync(join(dir!, `${name}.bin`))
    const raw = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
    const arr = (key: string) => {
      const a = meta.arrays[key]
      return raw.subarray(a.offset / 4, a.offset / 4 + a.shape.reduce((x: number, y: number) => x * y, 1))
    }
    const { width, height } = meta
    const rgb = arr('image')
    const data = new Uint8ClampedArray(width * height * 4)
    for (let p = 0; p < width * height; p += 1) data.set([rgb[3 * p], rgb[3 * p + 1], rgb[3 * p + 2], 255], 4 * p)
    ort.env.wasm.numThreads = 1
    const t = performance.now()
    const pred = await predictDense(ort, (name) =>
      ort.InferenceSession.create(`public/texrot/${name}.onnx`, { executionProviders: ['wasm'] }), { width, height, data }, meta.fov,
      undefined, tile)
    console.log('networks', ((performance.now() - t) / 1000).toFixed(1), 's')
    const face = arr('face')
    const tph = arr('tphase')
    let agree = 0
    let phaseErr = 0
    let phaseBad = 0
    for (let p = 0; p < width * height; p += 1) {
      if (pred.face[p] === face[p]) agree += 1
      for (const a of [0, 1, 2]) {
        const d = Math.abs(pred.tphase[3 * p + a] - tph[3 * p + a])
        const e = Math.min(d, 1 - d)
        phaseErr += e
        if (e > 0.05) phaseBad += 1
      }
    }
    const n = width * height
    // diagnostics only: tiles see less context than Python's whole-image pass, and every pixel
    // counts here (sky, unmatched surfaces); what matters is the rows below
    console.log(name, tile, 'face agreement', agree / n, 'mean phase error', phaseErr / (3 * n), 'phase > 0.05', phaseBad / (3 * n))

    const kernels = withCpuPoseKernels(createOrtKernels(ort, (k) =>
      ort.InferenceSession.create(`public/texrot/kernels/${k}.onnx`, { executionProviders: ['wasm'] })))
    const bin = read('public/texrot/templates.bin')
    const templates = parseTemplates(JSON.parse(read('public/texrot/templates.json', 'utf8')) as TemplateIndex,
      bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength))
    const result = await detectFromPrediction({ kernels, templates }, { width, height, data }, pred, meta.fov)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const fixture = JSON.parse(read(`src/autodetect/fixtures/${name}.json`, 'utf8'))
    // same grid -> same cell numbering: compare relative to Python's origin block
    const hasOrigin = result.detection.rows.some((r) => r.cell.every((v, i) => v === fixture.origin[i]))
    if (hasOrigin) result.detection.origin = fixture.origin
    const ours = configRows(result.detection)
    writeFileSync(join(dir!, `${name}_tile${tile}.conf`), [
      `# ${name}: TypeScript pipeline, ${tile} px tiles`, 'algorithm = Vanilla-3', 'scanOrder = spiral',
      'directions = [0, 90, 180, 270]', 'xRange = (-5000, 5000)', 'yRange = (-64, 320)', 'zRange = (-5000, 5000)',
      'errorTolerance = 0', '[filter]', ...ours, '',
    ].join('\n'))
    const theirs = configRows(fixture)
    const common = [...ours].filter((r) => theirs.has(r)).length
    console.log(name, tile, 'origin shared', hasOrigin, 'grid', result.detection.grid, fixture.grid, 'rows common', common, 'ours', ours.size, 'python', theirs.size)
    expect(result.detection.grid).toEqual(fixture.grid)
    expect(common).toBeGreaterThan(0.8 * theirs.size)
    // rows that differ must not contradict: same block face, different row value
    const key = (r: string) => r.split('|')[0].trim()
    const theirKeys = new Map([...theirs].map((r) => [key(r), r]))
    const conflicts = [...ours].filter((r) => theirKeys.has(key(r)) && theirKeys.get(key(r)) !== r)
    console.log(name, 'conflicting rows', conflicts.length)
    expect(conflicts.length).toBe(0)
  }, 1_800_000)
})

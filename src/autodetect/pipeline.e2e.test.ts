// @vitest-environment node
/**
 * End-to-end check of the TypeScript geometry against TexRotation's Python pipeline on real
 * screenshots. Needs the dense predictions dumped by TexRotation's export_pred.py; set
 * TEXROT_PRED_DIR to their folder (e.g. ../out/pred). Skipped otherwise.
 */
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import * as ort from 'onnxruntime-web'
import { describe, expect, it } from 'vitest'
import { generateCoordsFinderConfig } from '../domain/exportConfig'
import { createEmptyDocument } from '../store/editorStore'
import { buildDetectedDocument } from './document'
import { fitPose, predictionMasks, screenFor, type DensePrediction } from './geometry'
import { withCpuPoseKernels } from './cpuKernels'
import { createOrtKernels } from './kernels'
import { detectFromPrediction, parseTemplates, type TemplateIndex } from './pipeline'
import type { Detection } from './types'

const dir = process.env.TEXROT_PRED_DIR
const assets = 'public/texrot'

function load(name: string) {
  const meta = JSON.parse(readFileSync(join(dir!, `${name}.json`), 'utf8'))
  const buf = readFileSync(join(dir!, `${name}.bin`))
  const raw = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
  const arr = (key: string) => {
    const a = meta.arrays[key]
    const len = a.shape.reduce((x: number, y: number) => x * y, 1)
    return raw.subarray(a.offset / 4, a.offset / 4 + len)
  }
  const { width, height } = meta
  const rgb = arr('image')
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let p = 0; p < width * height; p += 1) {
    rgba.set([rgb[3 * p], rgb[3 * p + 1], rgb[3 * p + 2], 255], 4 * p)
  }
  const pred: DensePrediction = {
    width,
    height,
    face: Uint8Array.from(arr('face')),
    faceConf: arr('face_conf'),
    templated: arr('templated'),
    pos: arr('pos'),
    tphase: arr('tphase'),
  }
  return { meta, pred, image: { width, height, data: rgba } }
}

function runtime() {
  ort.env.wasm.numThreads = 1
  // as in the browser: pose search on the CPU (in-thread here), NCC through ONNX
  const kernels = withCpuPoseKernels(createOrtKernels(ort, (name) =>
    ort.InferenceSession.create(`${assets}/kernels/${name}.onnx`, { executionProviders: ['wasm'] })))
  const index = JSON.parse(readFileSync(`${assets}/templates.json`, 'utf8')) as TemplateIndex
  const bin = readFileSync(`${assets}/templates.bin`)
  const templates = parseTemplates(index, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength))
  return { kernels, templates }
}

function configRows(detection: Detection): Set<string> {
  const parts = buildDetectedDocument(detection)
  const document = createEmptyDocument()
  document.image = { ...document.image, width: detection.width, height: detection.height }
  document.scene = parts.scene
  document.evidence = parts.evidence.map((e) => ({ ...e, reviewStatus: 'confirmed' }))
  document.anchorFaceId = parts.anchorFaceId
  const text = generateCoordsFinderConfig(document)
  return new Set(text.split('[filter]')[1].split('\n').map((l) => l.split('#')[0].trim()).filter(Boolean))
}

describe.skipIf(!dir || !existsSync(join(dir ?? '', 'cave.json')))('TypeScript port vs Python pipeline', () => {
  for (const name of ['cave', 'forest']) {
    it(`${name}: pose and rows`, async () => {
      const { meta, pred, image } = load(name)
      const rt = runtime()
      const masks = predictionMasks(pred)
      const smoothFloor = masks.floor.map((v, p) => v & masks.smooth[p])
      const smoothCeil = masks.ceiling.map((v, p) => v & masks.smooth[p])
      let t = performance.now()
      const { pose } = await fitPose(rt.kernels, screenFor(pred.width, pred.height, meta.fov), pred.tphase, smoothFloor, smoothCeil)
      console.log(name, 'fitPose', ((performance.now() - t) / 1000).toFixed(1), 's', pose, 'python', meta.fit_pose.pose)
      const want = meta.fit_pose.pose
      expect(pose).not.toBeNull()
      expect(Math.abs(pose!.yaw - want.yaw)).toBeLessThan(0.05)
      expect(Math.abs(pose!.pitch - want.pitch)).toBeLessThan(0.05)
      expect(Math.abs(pose!.h / want.h - 1)).toBeLessThan(0.002)

      t = performance.now()
      const result = await detectFromPrediction(rt, image, pred, meta.fov, (m) => console.log('  ', m))
      console.log(name, 'detect', ((performance.now() - t) / 1000).toFixed(1), 's')
      expect(result.ok).toBe(true)
      if (!result.ok) return
      const fixture = JSON.parse(readFileSync(`src/autodetect/fixtures/${name}.json`, 'utf8'))
      console.log(name, 'grid', result.detection.grid, 'python', fixture.grid, 'rows', result.detection.rows.length,
        'python', fixture.rows.length, 'parity', result.detection.parity, fixture.parity)
      expect(result.detection.grid).toEqual(fixture.grid)
      expect(result.detection.parity).toBe(fixture.parity)
      // same rows up to a few borderline ones (float32 vs float64 NCC, different subsamples)
      const ours = configRows(result.detection)
      const theirs = configRows(fixture)
      const common = [...ours].filter((r) => theirs.has(r)).length
      console.log(name, 'rows common', common, 'ours', ours.size, 'python', theirs.size)
      expect(common).toBeGreaterThan(0.9 * Math.max(ours.size, theirs.size))
    }, 1_800_000)
  }
})

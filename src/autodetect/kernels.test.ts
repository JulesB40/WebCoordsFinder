// @vitest-environment node
import * as ort from 'onnxruntime-web'
import { describe, expect, it } from 'vitest'
import { npz } from './testVectors'
import { createOrtKernels } from './kernels'

describe('ONNX kernels (WASM)', () => {
  it('ncc_best matches PyTorch', async () => {
    ort.env.wasm.numThreads = 1
    const kernels = createOrtKernels(ort, (name) =>
      ort.InferenceSession.create(`public/texrot/kernels/${name}.onnx`, { executionProviders: ['wasm'] }))
    const a = npz('ncc_best')
    const [mi, mj] = [40, 50]
    const { best, valid } = await kernels.nccBest(a.in_T, a.in_M, mi, mj, a.in_tmpl, a.in_tmpl.length / 256)
    let maxBest = 0
    let maxValid = 0
    for (let i = 0; i < best.length; i += 1) {
      maxBest = Math.max(maxBest, Math.abs(best[i] - a.out_best[i]))
      maxValid = Math.max(maxValid, Math.abs(valid[i] - a.out_count[i]))
    }
    expect(best.length).toBe((mi - 15) * (mj - 15))
    expect(maxBest).toBeLessThan(1e-4)
    expect(maxValid).toBe(0)
  })
})

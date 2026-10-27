// Compares onnxruntime-web (WASM backend, as the browser falls back to) with PyTorch reference outputs.
// Usage: node scripts/check-texrot-onnx.mjs ../out
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web'

const ref = process.argv[2] ?? '../out'
const load = (name) => new Float32Array(readFileSync(`${ref}/ref_${name}.bin`).buffer.slice(0))
ort.env.wasm.numThreads = 1
const x = new ort.Tensor('float32', load('x'), [1, 6, 320, 448])
for (const [model, outputs] of [['lattice', ['tphase']], ['geometry', ['face_prob', 'templated', 'pos']]]) {
  const t0 = Date.now()
  const session = await ort.InferenceSession.create(`public/texrot/${model}.onnx`, { executionProviders: ['wasm'] })
  const result = await session.run({ x })
  for (const name of outputs) {
    const got = result[name].data
    const want = load(name)
    let max = 0, sum = 0
    for (let i = 0; i < want.length; i += 1) {
      const d = Math.abs(got[i] - want[i])
      max = Math.max(max, d)
      sum += d
    }
    console.log(`${model}.${name}: max |diff| ${max.toFixed(4)}, mean ${(sum / want.length).toFixed(5)}`)
  }
  console.log(`  ${model}: ${((Date.now() - t0) / 1000).toFixed(1)} s incl. load`)
}

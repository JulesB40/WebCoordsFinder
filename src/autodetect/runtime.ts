/**
 * Browser runtime of the detector: onnxruntime-web with WebGPU when the browser has it, WASM
 * otherwise. Models, kernels, templates and the ORT WebAssembly binary are all served from
 * this application's own origin; nothing is fetched from elsewhere.
 */
import * as ort from 'onnxruntime-web'
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'
import { withCpuPoseKernels } from './cpuKernels'
import { createPoseKernelPool } from './kernelPool'
import { createOrtKernels } from './kernels'
import { parseTemplates, type DetectorRuntime, type TemplateIndex } from './pipeline'

export type Backend = 'webgpu' | 'wasm'

export const TEXROT_ASSETS = `${import.meta.env.BASE_URL}texrot/`

const MODEL_FILES: Record<string, string> = {
  lattice: 'lattice.onnx',
  geometry: 'geometry.onnx',
  ncc_best: 'kernels/ncc_best.onnx',
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Unable to load ${url} (${response.status}).`)
  return new Uint8Array(await response.arrayBuffer())
}

/**
 * A WebGPU device on the high-performance adapter (laptops otherwise get the integrated GPU),
 * with the adapter's compute limits; null without WebGPU.
 */
async function webgpuDevice(): Promise<GPUDevice | null> {
  if (!('gpu' in navigator)) return null
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) return null
    const { limits } = adapter
    return await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: limits.maxBufferSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
      },
      requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
    })
  } catch {
    return null
  }
}

export async function createBrowserRuntime(
  base = TEXROT_ASSETS,
  report: (message: string) => void = () => {},
): Promise<DetectorRuntime & { backend: Backend }> {
  ort.env.wasm.wasmPaths = { wasm: new URL(wasmUrl, globalThis.location.href).href }
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1
  const device = await webgpuDevice()
  if (device) ort.env.webgpu.device = device
  const backend: Backend = device ? 'webgpu' : 'wasm'
  const providers = backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']

  const sessions = new Map<string, Promise<ort.InferenceSession>>()
  const session = (name: string) => {
    let s = sessions.get(name)
    if (!s) {
      s = (async () => {
        if (name === 'lattice' || name === 'geometry') report(`Loading the ${name} network`)
        const bytes = await fetchBytes(`${base}${MODEL_FILES[name]}`)
        return ort.InferenceSession.create(bytes, { executionProviders: providers, graphOptimizationLevel: 'all' })
      })()
      sessions.set(name, s)
    }
    return s
  }

  const [index, bin] = await Promise.all([
    fetch(`${base}templates.json`).then((r) => r.json() as Promise<TemplateIndex>),
    fetchBytes(`${base}templates.bin`),
  ])
  return {
    backend,
    ort,
    session,
    // pose search on CPU workers everywhere (exact sin/cos), texture NCC through ONNX
    kernels: withCpuPoseKernels(
      createOrtKernels(ort, session),
      createPoseKernelPool(Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 2) - 1))),
    ),
    templates: parseTemplates(index, bin.buffer as ArrayBuffer),
  }
}

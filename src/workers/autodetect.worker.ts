import type { AutoDetectRequest, AutoDetectResponse } from '../autodetect/protocol'
import { detect } from '../autodetect/pipeline'
import { createBrowserRuntime } from '../autodetect/runtime'

async function decodedPixels(source: string): Promise<ImageData> {
  const response = await fetch(source)
  if (!response.ok) throw new Error(`Unable to load the screenshot (${response.status}).`)
  const bitmap = await createImageBitmap(await response.blob())
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) throw new Error('Off-thread image extraction is unavailable.')
    context.drawImage(bitmap, 0, 0)
    return context.getImageData(0, 0, bitmap.width, bitmap.height)
  } finally {
    bitmap.close()
  }
}

self.onmessage = async (event: MessageEvent<AutoDetectRequest>) => {
  const { requestId, sourceUrl, fov } = event.data
  const post = (message: AutoDetectResponse) => self.postMessage(message)
  const report = (message: string) => post({ type: 'progress', requestId, message })
  const start = performance.now()
  try {
    report('Loading the screenshot')
    const image = await decodedPixels(sourceUrl)
    const runtime = await createBrowserRuntime(undefined, report)
    report(runtime.backend === 'webgpu' ? 'Using the GPU (WebGPU)' : 'Using the CPU (WebAssembly); this can take a few minutes')
    const result = await detect(runtime, image, { fov, report })
    if (!result.ok) {
      post({ type: 'error', requestId, error: `No coordinates evidence found: ${result.reason}.` })
      return
    }
    post({
      type: 'result',
      requestId,
      detection: result.detection,
      backend: runtime.backend,
      fovEstimated: result.fovEstimated,
      wellPosed: result.wellPosed,
      seconds: (performance.now() - start) / 1000,
    })
  } catch (error) {
    post({ type: 'error', requestId, error: error instanceof Error ? error.message : String(error) })
  }
}

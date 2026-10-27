import { readFileSync } from 'node:fs'
import { unzipSync } from 'fflate'

/** float32 arrays of a kernel's test vectors (written by TexRotation's export_kernels.py). */
export function npz(name: string): Record<string, Float32Array> {
  const files = unzipSync(new Uint8Array(readFileSync(`src/autodetect/fixtures/kernels/${name}.test.npz`)))
  const out: Record<string, Float32Array> = {}
  for (const [file, bytes] of Object.entries(files)) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    const headerLength = view.getUint16(8, true)
    const header = new TextDecoder().decode(bytes.subarray(10, 10 + headerLength))
    if (!header.includes("'<f4'")) continue
    out[file.replace('.npy', '')] = new Float32Array(bytes.slice(10 + headerLength).buffer)
  }
  return out
}

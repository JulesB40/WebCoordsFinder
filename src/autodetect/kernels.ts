/**
 * The heavy, branch-free parts of the detector, run as small ONNX graphs (exported from
 * TexRotation's texrot/kernels.py) on WebGPU when available, WASM otherwise. Inputs are
 * flat float32 arrays; angles in radians; `sign` is +1 for floors, -1 for ceilings.
 */
import type * as OrtModule from 'onnxruntime-web'

export type Ort = typeof OrtModule

export interface Kernels {
  /** Largest number of elements of one [K, Hh, N] intermediate a call should create. */
  readonly budget: number
  /** Coarse pose search loss [K * Hh] for poses (yaw[K], pitch[K]) and eye heights hs[Hh]. */
  pairScores(input: {
    raysI: Float32Array
    raysJ: Float32Array
    obsX: Float32Array
    obsZ: Float32Array
    yaw: Float32Array
    pitch: Float32Array
    hs: Float32Array
    sign: number
  }): Promise<Float32Array>
  /** Lattice phase coherence [K * Hh]. */
  coherence(input: {
    rays: Float32Array
    phx: Float32Array
    phz: Float32Array
    yaw: Float32Array
    pitch: Float32Array
    hs: Float32Array
    sign: number
  }): Promise<Float32Array>
  /** Best masked NCC over `count` 16x16 templates for every window of a [mi, mj] map. */
  nccBest(
    T: Float32Array,
    M: Float32Array,
    mi: number,
    mj: number,
    templates: Float32Array,
    count: number,
  ): Promise<{ best: Float32Array; valid: Float32Array }>
}

export type SessionFactory = (name: string) => Promise<OrtModule.InferenceSession>

export function createOrtKernels(ort: Ort, session: SessionFactory, budget = 1 << 22): Kernels {
  const sessions = new Map<string, Promise<OrtModule.InferenceSession>>()
  const get = (name: string) => {
    let s = sessions.get(name)
    if (!s) {
      s = session(name)
      sessions.set(name, s)
    }
    return s
  }
  const t = (data: Float32Array, dims: number[]) => new ort.Tensor('float32', data, dims)
  const scalar = (v: number) => new ort.Tensor('float32', Float32Array.of(v), [])
  const read = async (tensor: OrtModule.Tensor) => {
    const data = tensor.location === 'cpu' ? tensor.data : await tensor.getData(true)
    return data instanceof Float32Array ? data : Float32Array.from(data as ArrayLike<number>)
  }

  return {
    budget,
    async pairScores({ raysI, raysJ, obsX, obsZ, yaw, pitch, hs, sign }) {
      const s = await get('pair_scores')
      const m = obsX.length
      const out = await s.run({
        rays_i: t(raysI, [m, 3]),
        rays_j: t(raysJ, [m, 3]),
        obs_x: t(obsX, [m]),
        obs_z: t(obsZ, [m]),
        yaw: t(yaw, [yaw.length]),
        pitch: t(pitch, [pitch.length]),
        hs: t(hs, [hs.length]),
        sign: scalar(sign),
      })
      return read(out.loss)
    },
    async coherence({ rays, phx, phz, yaw, pitch, hs, sign }) {
      const s = await get('coherence')
      const n = phx.length
      const out = await s.run({
        rays: t(rays, [n, 3]),
        phx: t(phx, [n]),
        phz: t(phz, [n]),
        yaw: t(yaw, [yaw.length]),
        pitch: t(pitch, [pitch.length]),
        hs: t(hs, [hs.length]),
        sign: scalar(sign),
      })
      return read(out.coh)
    },
    async nccBest(T, M, mi, mj, templates, count) {
      const s = await get('ncc_best')
      const out = await s.run({ T: t(T, [mi, mj]), M: t(M, [mi, mj]), tmpl: t(templates, [count, 16, 16]) })
      return { best: await read(out.best), valid: await read(out.count) }
    },
  }
}

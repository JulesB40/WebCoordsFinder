/** Runs the CPU pose kernels on a pool of workers, each taking a contiguous range of poses. */
import { splitPoses, type PoseKernelRunner } from './cpuKernels'
import type { PoseKernelRequest } from '../workers/poseKernels.worker'

type Pending = { resolve: (out: Float32Array) => void; reject: (error: Error) => void }

export function createPoseKernelPool(size: number): PoseKernelRunner & { size: number; terminate(): void } {
  const workers = Array.from({ length: size }, () =>
    new Worker(new URL('../workers/poseKernels.worker.ts', import.meta.url), { type: 'module' }))
  const pending = new Map<number, Pending>()
  let nextId = 0
  for (const worker of workers) {
    worker.onmessage = (event: MessageEvent<{ id: number; out?: Float32Array; error?: string }>) => {
      const job = pending.get(event.data.id)
      if (!job) return
      pending.delete(event.data.id)
      if (event.data.out) job.resolve(event.data.out)
      else job.reject(new Error(event.data.error ?? 'Pose kernel failed.'))
    }
    worker.onerror = () => {
      pending.forEach((job) => job.reject(new Error('A pose kernel worker failed.')))
      pending.clear()
    }
  }

  const run = async (op: PoseKernelRequest['op'], input: PoseKernelRequest['input']) => {
    const parts = splitPoses(input, workers.length)
    const outs = await Promise.all(parts.map((part, i) => new Promise<Float32Array>((resolve, reject) => {
      const id = nextId++
      pending.set(id, { resolve, reject })
      workers[i].postMessage({ id, op, input: part } as PoseKernelRequest)
    })))
    const total = outs.reduce((n, o) => n + o.length, 0)
    const joined = new Float32Array(total)
    let offset = 0
    for (const o of outs) {
      joined.set(o, offset)
      offset += o.length
    }
    return joined
  }

  return {
    size,
    pairScores: (input) => run('pairScores', input),
    coherence: (input) => run('coherence', input),
    terminate: () => workers.forEach((w) => w.terminate()),
  }
}

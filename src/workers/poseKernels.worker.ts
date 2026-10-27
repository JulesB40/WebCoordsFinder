import { cpuCoherence, cpuPairScores, type CoherenceInput, type PairScoresInput } from '../autodetect/cpuKernels'

export type PoseKernelRequest =
  | { id: number; op: 'pairScores'; input: PairScoresInput }
  | { id: number; op: 'coherence'; input: CoherenceInput }

self.onmessage = (event: MessageEvent<PoseKernelRequest>) => {
  const { id } = event.data
  try {
    const out = event.data.op === 'pairScores' ? cpuPairScores(event.data.input) : cpuCoherence(event.data.input)
    self.postMessage({ id, out }, { transfer: [out.buffer] })
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) })
  }
}

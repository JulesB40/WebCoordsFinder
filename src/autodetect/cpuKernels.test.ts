// @vitest-environment node
import { describe, expect, it } from 'vitest'
import { cosTurns, cpuCoherence, cpuPairScores, sinTurns } from './cpuKernels'
import { npz } from './testVectors'

// Kernel test vectors are random and unnormalised, so phases reach thousands of turns near
// grazing angles where the float32 PyTorch reference is only good to ~1e-4 turns; these
// loops run in float64, hence the 1e-3 tolerance.

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length)
  let m = 0
  for (let i = 0; i < a.length; i += 1) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

describe('CPU pose kernels', () => {
  it('computes sin and cos of turns accurately', () => {
    for (let x = -3; x <= 3; x += 0.001237) {
      expect(Math.abs(sinTurns(x) - Math.sin(2 * Math.PI * x))).toBeLessThan(2e-7)
      expect(Math.abs(cosTurns(x) - Math.cos(2 * Math.PI * x))).toBeLessThan(2e-7)
    }
  })

  it('matches the pair_scores kernel', () => {
    const a = npz('pair_scores')
    const got = cpuPairScores({
      raysI: a.in_rays_i, raysJ: a.in_rays_j, obsX: a.in_obs_x, obsZ: a.in_obs_z,
      yaw: a.in_yaw, pitch: a.in_pitch, hs: a.in_hs, sign: a.in_sign[0],
    })
    expect(maxDiff(got, a.out_loss)).toBeLessThan(1e-3)
  })

  it('matches the coherence kernel', () => {
    const a = npz('coherence')
    const got = cpuCoherence({
      rays: a.in_rays, phx: a.in_phx, phz: a.in_phz, yaw: a.in_yaw, pitch: a.in_pitch, hs: a.in_hs, sign: a.in_sign[0],
    })
    expect(maxDiff(got, a.out_coh)).toBeLessThan(1e-3)
  })
})

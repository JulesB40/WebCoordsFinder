import { describe, expect, it } from 'vitest'
import { TILE, TILE_MARGIN, tileWindows } from './networks'

describe('tileWindows', () => {
  it.each([[1504, 2560], [320, 448], [1024, 1024], [1056, 2080], [2176, 3840]])('keeps every pixel of %ix%i exactly once', (hp, wp) => {
    const seen = new Uint8Array(hp * wp)
    for (const t of tileWindows(hp, wp)) {
      expect(t.w % 32).toBe(0)
      expect(t.h % 32).toBe(0)
      expect(t.x0).toBeGreaterThanOrEqual(0)
      expect(t.y0).toBeGreaterThanOrEqual(0)
      expect(t.x0 + t.w).toBeLessThanOrEqual(wp)
      expect(t.y0 + t.h).toBeLessThanOrEqual(hp)
      // kept part inside the window, with context on every side not at the image border
      expect(t.keepX0).toBeGreaterThanOrEqual(t.x0)
      expect(t.keepX1).toBeLessThanOrEqual(t.x0 + t.w)
      if (t.keepX0 > 0 && t.w === TILE) expect(t.keepX0 - t.x0).toBeGreaterThanOrEqual(TILE_MARGIN)
      if (t.keepX1 < wp && t.w === TILE) expect(t.x0 + t.w - t.keepX1).toBeGreaterThanOrEqual(TILE_MARGIN)
      for (let y = t.keepY0; y < t.keepY1; y += 1) for (let x = t.keepX0; x < t.keepX1; x += 1) seen[y * wp + x] += 1
    }
    expect(seen.every((v) => v === 1)).toBe(true)
  })
})

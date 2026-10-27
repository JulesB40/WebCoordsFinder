/**
 * Matched block faces -> CoordsFinder rows (port of TexRotation's texrot/postprocess.py).
 *
 * Frame parity (mirror family): north is unknown, so rows live in a lattice turned k quarter
 * turns from the world and CoordsFinder tries every direction. That is exact for quarter-turn,
 * netherrack and side rows but not for stone-like tops, whose CoordsFinder variant changes
 * "plain" into "mirrored" per quarter-turn. Those blocks reveal k's parity themselves; the
 * majority parity wins, rows disagreeing with it are dropped and, when k is odd, rows are
 * re-expressed one quarter-turn further (the axis mapping then says so).
 */
import type { MatchedBlock } from './geometry'
import type { DetectedRow, FaceName } from './types'

export function selectRows(blocks: MatchedBlock[], minNcc = 0.5, minMargin = 0.15, maxRows = 250): MatchedBlock[] {
  const good = blocks
    .filter((b) => b.kind !== 'static' && b.ncc >= minNcc && b.margin >= minMargin)
    .sort((a, b) => b.margin * b.ncc - a.margin * a.ncc)
  const seen = new Set<string>()
  const rows: MatchedBlock[] = []
  for (const b of good) {
    const key = `${b.cell.join(',')}:${b.face}`
    if (seen.has(key)) continue
    seen.add(key)
    rows.push(b)
  }
  return rows.slice(0, maxRows)
}

const SIDE_TURN: Record<FaceName, FaceName> = {
  north: 'east', east: 'south', south: 'west', west: 'north', up: 'up', down: 'down',
}

/** A row's value re-expressed one quarter-turn further (CoordsFinder rotate_observation, 90). */
export function turnedValue(b: Pick<MatchedBlock, 'kind' | 'value' | 'parity' | 'face'>): { value: number; parity: number } {
  if (b.kind === 'four') {
    if (b.parity >= 0) return { value: (b.value + 2) % 4, parity: b.parity === 1 ? 0 : 1 }
    return { value: (b.value + 1) % 4, parity: b.parity }
  }
  if (b.kind === 'netherrack') {
    const value = b.face === 'up' ? (b.value + 1) % 4 : b.face === 'down' ? (b.value + 3) % 4 : b.value
    return { value, parity: b.parity }
  }
  return { value: b.value, parity: b.parity }
}

/** Cell and face of a block one quarter-turn further: (x, z) -> (-z, x). */
export function turnedCell(cell: [number, number, number], face: FaceName) {
  return { cell: [-cell[2] - 1, cell[1], cell[0]] as [number, number, number], face: SIDE_TURN[face] }
}

/**
 * Rows in the detector's lattice with their final CoordsFinder values, the frame parity
 * (null when no mirror-family block was seen) and the directions CoordsFinder must try.
 */
export function resolveFrame(rows: MatchedBlock[]): {
  rows: DetectedRow[]
  parity: number | null
  directions: number[]
} {
  const mirror = rows.filter((r) => r.kind === 'four' && r.parity >= 0)
  const toRow = (r: MatchedBlock, value: number): DetectedRow => ({
    cell: r.cell,
    face: r.face,
    kind: r.kind as DetectedRow['kind'],
    value,
    parity: r.parity,
    texture: r.texture,
    ncc: r.ncc,
    margin: r.margin,
    centroid: r.centroid,
    scores: r.scores,
  })
  if (!mirror.length) return { rows: rows.map((r) => toRow(r, r.value)), parity: null, directions: [0, 90, 180, 270] }
  const weight = [0, 1].map((p) => mirror.filter((r) => r.parity === p).reduce((s, r) => s + r.margin, 0))
  const parity = weight[1] > weight[0] ? 1 : 0
  const kept = rows.filter((r) => !(r.kind === 'four' && r.parity >= 0 && r.parity !== parity))
  const turn = (r: MatchedBlock) => {
    if (parity !== 1) return toRow(r, r.value)
    const row = toRow(r, turnedValue(r).value)
    // per-variant scores follow the same re-expression
    row.scores = r.scores.map((s) => ({ ...s, variant: turnedValue({ ...r, value: s.variant }).value }))
    return row
  }
  return { rows: kept.map(turn), parity, directions: [0, 180] }
}

export function chooseOrigin(rows: DetectedRow[], width: number, height: number): DetectedRow {
  return rows.reduce((best, r) => {
    const d = (r.centroid[0] - width / 2) ** 2 + (r.centroid[1] - height / 2) ** 2
    const e = (best.centroid[0] - width / 2) ** 2 + (best.centroid[1] - height / 2) ** 2
    return d < e ? r : best
  })
}

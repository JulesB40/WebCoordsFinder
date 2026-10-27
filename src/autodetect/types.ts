/**
 * Shared shapes of the automatic detector (a TypeScript port of TexRotation's Python pipeline).
 *
 * Coordinates: the detector works in a canonical lattice turned some unknown number of quarter
 * turns from the world (north can't be seen). Lattice integers are block boundaries; a block
 * cell (x, y, z) spans [x, x+1) etc. `FaceName` is the face as seen in that lattice.
 */

export type FaceName = 'up' | 'down' | 'north' | 'south' | 'east' | 'west'
export type RowKind = 'four' | 'side' | 'netherrack'

export interface Pose {
  /** Degrees, canonical frame; 0 looks along +Z. */
  yaw: number
  /** Degrees, positive looks down. */
  pitch: number
  /** Eye height above the main plane in blocks (negative: the plane is a ceiling). */
  h: number
  /** Camera position within a texel along X and Z, in texels (0..1). */
  c16x: number
  c16z: number
}

export interface DetectedRow {
  /** Block cell in the detector's lattice. */
  cell: [number, number, number]
  /** Visible face in the detector's lattice. */
  face: FaceName
  kind: RowKind
  /** CoordsFinder row value in the world frame implied by the axis mapping. */
  value: number
  /** Mirror family top/bottom faces: frame parity of the match, otherwise -1. */
  parity: number
  /** Block id (matches WebCoordsFinder profile ids). */
  texture: string
  ncc: number
  /** Best score minus the best score implying a different row. */
  margin: number
  /** Image pixel at the face centre. */
  centroid: [number, number]
  /** Per-variant scores, best first, if computed. */
  scores?: { variant: number; score: number }[]
}

export interface Detection {
  width: number
  height: number
  /** Vertical field of view in degrees. */
  fov: number
  pose: Pose
  /** Texel offset of the block grid (0..15) along X and Z. */
  grid: { ox: number; oz: number }
  /** 1: rows were re-expressed one quarter-turn further (stone frame parity); 0/null otherwise. */
  parity: number | null
  rows: DetectedRow[]
  /** Cell of the origin block (the row nearest the image centre). */
  origin: [number, number, number]
}

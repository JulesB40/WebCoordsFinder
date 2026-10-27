import type {
  AxisMapping,
  CalibrationObservation,
  EditorDocument,
  FaceEvidence,
  Matrix3x4,
  MeshFace,
  Point3,
} from '../domain/types'
import { projectCamera } from '../domain/geometry'
import type { Detection, FaceName } from './types'

const NORMALS: Record<FaceName, Point3> = {
  up: { x: 0, y: 1, z: 0 },
  down: { x: 0, y: -1, z: 0 },
  north: { x: 0, y: 0, z: -1 },
  south: { x: 0, y: 0, z: 1 },
  east: { x: 1, y: 0, z: 0 },
  west: { x: -1, y: 0, z: 0 },
}

/** Camera basis for yaw/pitch (degrees): columns right, up, forward in the lattice frame. */
export function cameraBasis(yaw: number, pitch: number): { r: Point3; u: Point3; f: Point3 } {
  const y = (yaw * Math.PI) / 180
  const p = (pitch * Math.PI) / 180
  const f = { x: -Math.sin(y) * Math.cos(p), y: -Math.sin(p), z: Math.cos(y) * Math.cos(p) }
  // right = f x up, normalised; straight down keeps yaw's right vector
  let r = { x: -f.z, y: 0, z: f.x }
  let n = Math.hypot(r.x, r.z)
  if (n < 1e-6) {
    r = { x: -Math.cos(y), y: 0, z: -Math.sin(y) }
    n = 1
  }
  r = { x: r.x / n, y: 0, z: r.z / n }
  const u = {
    x: r.y * f.z - r.z * f.y,
    y: r.z * f.x - r.x * f.z,
    z: r.x * f.y - r.y * f.x,
  }
  return { r, u, f }
}

/**
 * Projective camera taking lattice points (block boundaries at integers) to image pixels
 * (top-left image corner = 0, pixel X right, Y down), as WebCoordsFinder's CameraProjection.
 */
export function detectionCameraMatrix(detection: Detection): Matrix3x4 {
  const { width: W, height: H, fov, pose, grid } = detection
  const { r, u, f } = cameraBasis(pose.yaw, pose.pitch)
  const focal = H / 2 / Math.tan((fov * Math.PI) / 360)
  // Camera centre in lattice coordinates: the detector's grid frame is shifted by the
  // texel offsets of the block grid.
  const c = {
    x: pose.c16x / 16 - grid.ox / 16,
    y: pose.h,
    z: pose.c16z / 16 - grid.oz / 16,
  }
  // rows of R^T: camera x = right, camera y = up, camera z = forward
  const rows = [r, u, f]
  const t = rows.map((axis) => -(axis.x * c.x + axis.y * c.y + axis.z * c.z))
  // K = [[f, 0, W/2], [0, -f, H/2], [0, 0, 1]] (image Y points down)
  const k = [
    [focal, 0, W / 2],
    [0, -focal, H / 2],
    [0, 0, 1],
  ]
  const m: number[] = []
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 4; j += 1) {
      let sum = 0
      for (let q = 0; q < 3; q += 1) {
        const rt = j < 3 ? [rows[q].x, rows[q].y, rows[q].z][j] : t[q]
        sum += k[i][q] * rt
      }
      m.push(sum)
    }
  }
  return m as Matrix3x4
}

/**
 * World axis labels for the detector's lattice. With stone-like blocks the detector knows the
 * frame parity; when it re-expressed its rows one quarter-turn further, the world frame is the
 * lattice turned once, (x, z) -> (-z, x): lattice +X is world +Z and lattice +Z is world -X.
 */
export function detectionAxisMapping(detection: Detection): AxisMapping {
  return detection.parity === 1
    ? { a: 'z+', b: 'y+', c: 'x-' }
    : { a: 'x+', b: 'y+', c: 'z+' }
}

function faceSquareCorner(cell: [number, number, number], normal: Point3): Point3 {
  return {
    x: cell[0] + Math.max(0, normal.x),
    y: cell[1] + Math.max(0, normal.y),
    z: cell[2] + Math.max(0, normal.z),
  }
}

export interface AutoDetectedDocumentParts {
  scene: EditorDocument['scene']
  evidence: FaceEvidence[]
  anchorFaceId: string | null
}

/**
 * Turns a detection into WebCoordsFinder scene geometry and evidence. Every detected face
 * becomes a mesh face with scored evidence; whether it is shown as a proposal follows the
 * project's confidence threshold, and nothing is confirmed automatically.
 */
export function buildDetectedDocument(
  detection: Detection,
  newId: () => string = () => crypto.randomUUID(),
): AutoDetectedDocumentParts {
  const matrix = detectionCameraMatrix(detection)
  const faces: MeshFace[] = []
  const evidence: FaceEvidence[] = []
  let anchorFaceId: string | null = null
  const seen = new Set<string>()
  for (const row of detection.rows) {
    const key = `${row.cell.join(',')}:${row.face}`
    if (seen.has(key)) continue
    seen.add(key)
    const normal = NORMALS[row.face]
    const face: MeshFace = { id: newId(), blockCoordinate: faceSquareCorner(row.cell, normal), normal }
    faces.push(face)
    const scores = row.scores ?? [
      { variant: row.value, score: row.ncc },
      { variant: row.kind === 'side' ? 1 - row.value : (row.value + 2) % 4, score: row.ncc - row.margin },
    ]
    evidence.push({
      id: newId(),
      faceId: face.id,
      latticeCoordinate: { x: row.cell[0], y: row.cell[1], z: row.cell[2] },
      localNormal: normal,
      blockId: row.texture,
      stateCount: row.kind === 'side' ? 2 : 4,
      selectedVariant: row.value,
      reviewStatus: 'proposed',
      scores,
      confidence: row.margin,
      blockSettings: {},
    })
    if (row.cell.every((v, i) => v === detection.origin[i]) && anchorFaceId === null) anchorFaceId = face.id
  }

  // Calibration dots at every face corner keep any later refit on the detected camera.
  const corners = new Map<string, Point3>()
  for (const face of faces) {
    const { blockCoordinate: o, normal: n } = face
    const [ua, va]: [Point3, Point3] = n.x !== 0
      ? [{ x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 }]
      : n.y !== 0
        ? [{ x: 1, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }]
        : [{ x: 1, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }]
    for (const [a, b] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
      const p = { x: o.x + ua.x * a + va.x * b, y: o.y + ua.y * a + va.y * b, z: o.z + ua.z * a + va.z * b }
      corners.set(`${p.x},${p.y},${p.z}`, p)
    }
  }
  const observations: CalibrationObservation[] = [...corners.values()].map((lattice) => ({
    id: newId(),
    lattice,
    image: projectCamera(matrix, lattice),
    weight: 1,
  }))

  const upFace = faces.find((face) => face.normal.y === 1) ?? faces.find((face) => face.normal.y === -1)
  return {
    scene: {
      faces,
      observations,
      projection: { kind: 'camera', matrix, rmsError: 0, maxError: 0 },
      axisMapping: detectionAxisMapping(detection),
      worldUpIntent: upFace
        ? { faceId: upFace.id, surfaceKind: upFace.normal.y === 1 ? 'top' : 'bottom', edge: null }
        : null,
      horizontalOrientationIntent: null,
    },
    evidence,
    anchorFaceId: anchorFaceId ?? faces[0]?.id ?? null,
  }
}

import { describe, expect, it } from 'vitest'
import { generateCoordsFinderConfig } from '../domain/exportConfig'
import { isAxisMappingComplete, projectCamera, sceneLatticeParity } from '../domain/geometry'
import { createEmptyDocument } from '../store/editorStore'
import type { EditorDocument } from '../domain/types'
import { buildDetectedDocument, detectionCameraMatrix } from './document'
import type { Detection } from './types'
import cave from './fixtures/cave.json'
import forest from './fixtures/forest.json'

// Fixtures come from TexRotation's Python pipeline on real screenshots (export_fixture.py):
// its pose, grid alignment, rows in its own lattice and the CoordsFinder rows it writes.
type Fixture = Detection & { config: string[] }

function documentFor(fixture: Fixture): EditorDocument {
  let n = 0
  const parts = buildDetectedDocument(fixture, () => `id-${(n += 1)}`)
  const document = createEmptyDocument()
  document.image = { ...document.image, width: fixture.width, height: fixture.height }
  document.scene = parts.scene
  document.evidence = parts.evidence.map((entry) => ({ ...entry, reviewStatus: 'confirmed' }))
  document.anchorFaceId = parts.anchorFaceId
  document.scanner.directions = [0, 90, 180, 270]
  return document
}

function filterRows(config: string): string[] {
  return config
    .split('[filter]')[1]
    .split('\n')
    .map((line) => line.split('#')[0].trim())
    .filter(Boolean)
}

describe.each([
  ['cave (stone, odd frame parity)', cave as unknown as Fixture],
  ['forest (grass)', forest as unknown as Fixture],
])('auto-detected document: %s', (_, fixture) => {
  it('uses a camera whose parity agrees with the axis mapping', () => {
    const document = documentFor(fixture)
    expect(sceneLatticeParity(document.scene)).toBe(1)
    expect(isAxisMappingComplete(document.scene.axisMapping, sceneLatticeParity(document.scene))).toBe(true)
  })

  it('projects every matched texel onto the pixel where the detector read it', () => {
    const matrix = detectionCameraMatrix(fixture)
    const normals = { up: [0, 1, 0], down: [0, -1, 0], north: [0, 0, -1], south: [0, 0, 1], east: [1, 0, 0], west: [-1, 0, 0] }
    for (const row of fixture.rows) {
      const n = normals[row.face]
      // the detector reports the pixel of the face's texel (8, 8), whose centre is at 8.5/16;
      // its pixel indices are pixel centres, WebCoordsFinder's image coordinates pixel corners
      const along = (axis: number) => row.cell[axis] + (n[axis] === 0 ? 8.5 / 16 : n[axis] > 0 ? 1 : 0)
      const p = projectCamera(matrix, { x: along(0), y: along(1), z: along(2) })
      expect(Math.hypot(p.x - row.centroid[0] - 0.5, p.y - row.centroid[1] - 0.5)).toBeLessThan(0.05)
    }
  })

  it('exports exactly the CoordsFinder rows the Python pipeline wrote', () => {
    const rows = filterRows(generateCoordsFinderConfig(documentFor(fixture)))
    expect(new Set(rows)).toEqual(new Set(fixture.config))
  })
})

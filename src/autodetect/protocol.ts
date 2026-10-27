import type { Detection } from './types'

export interface AutoDetectRequest {
  type: 'detect'
  requestId: string
  sourceUrl: string
  /** Vertical FOV in degrees; estimated from the screenshot when omitted. */
  fov?: number
}

export type AutoDetectResponse =
  | { type: 'progress'; requestId: string; message: string }
  | {
      type: 'result'
      requestId: string
      detection: Detection
      backend: string
      fovEstimated: boolean
      wellPosed: boolean
      seconds: number
    }
  | { type: 'error'; requestId: string; error: string }

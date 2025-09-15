export type AskRequest = {
  query: string
  top_k?: number
  web_search?: boolean
}

export type Citation = {
  serial: number
  filename: string
  page: number
  chunk_id: string
  start: number
  end: number
}

export type AskResponse = {
  answer: string
  citations: Citation[]
  latency_ms: number
}


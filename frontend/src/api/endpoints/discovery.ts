import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'
import type { DiscoveryStats } from '@/types/api'

export function useDiscoveryStats() {
  return useQuery<DiscoveryStats>({
    queryKey: ['discovery-stats'],
    queryFn: () => api.get('/api/discovery/stats').then(r => r.data),
    staleTime: 60_000,
  })
}

// ── Discovery List Types ─────────────────────────────────────────────────────

export type DiscoveryStatus = 'new' | 'screened' | 'promoted' | 'dismissed'

export interface Discovery {
  id: number
  ticker: string
  name: string
  status: DiscoveryStatus
  discovery_score: number
  signal: string
  reason: string
  discovered_at: string
}

interface LastRun {
  run_type: string
  run_at: string
  tickers_scanned: number
  discoveries_found: number
  promoted_count: number
  duration_seconds: number
  errors: boolean
}

export interface DiscoveriesStats {
  week_total: number
  week_promoted: number
  total_new: number
  total_screened: number
  total_dismissed: number
  last_run: LastRun | null
}

export interface DiscoveriesResponse {
  discoveries: Discovery[]
  stats: DiscoveriesStats
  log: unknown[]
  status_filter: string
}

// ── Discovery Hooks ──────────────────────────────────────────────────────────

export function useDiscoveries(status: string) {
  return useQuery<DiscoveriesResponse>({
    queryKey: ['discoveries', status],
    queryFn: () => api.get(`/api/discoveries?status=${status}`).then(r => r.data),
    staleTime: 30_000,
  })
}

export function usePromoteDiscovery() {
  return useMutation({
    mutationFn: (id: number) =>
      api.post(`/api/discoveries/${id}/promote`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['discoveries'] }),
  })
}

export function useDismissDiscovery() {
  return useMutation({
    mutationFn: (id: number) =>
      api.post(`/api/discoveries/${id}/dismiss`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['discoveries'] }),
  })
}

// ── Manual AI Discovery (Perplexity) ────────────────────────────────────────

export interface DiscoverParams {
  sector?: string
  focus?: string
  limit?: number
}

export interface DiscoveredStock {
  ticker: string
  name?: string
  score: number
  reason: string
  catalyst: string
}

export interface DiscoverResult {
  success: boolean
  stocks: DiscoveredStock[]
  error?: string
  raw_analysis?: string
  timestamp?: string
  api_usage?: number
}

export function useRunDiscover() {
  return useMutation<DiscoverResult, Error, DiscoverParams>({
    mutationFn: async (params) => {
      const form = new FormData()
      if (params.sector) form.append('sector', params.sector)
      form.append('focus', params.focus ?? 'balanced')
      form.append('limit', String(params.limit ?? 5))
      // CSRF token is injected by the axios interceptor for POST requests
      const { data: csrfData } = await api.get('/api/csrf-token')
      form.append('csrf_token', csrfData.token)
      return api.post('/discover', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }).then(r => r.data)
    },
  })
}

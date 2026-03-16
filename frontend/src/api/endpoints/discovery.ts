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

import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { ApiStatus, HealthStatus, ScanProgress } from '@/types/api'

export function useApiStatus(refetchInterval: number | false = 30_000) {
  return useQuery<ApiStatus>({
    queryKey: ['status'],
    queryFn: () => api.get('/api/status').then(r => r.data),
    staleTime: 20_000,
    refetchInterval,
  })
}

export function useHealth() {
  return useQuery<HealthStatus>({
    queryKey: ['health'],
    queryFn: () => api.get('/api/health').then(r => r.data),
    staleTime: 60_000,
    refetchInterval: 60_000,
  })
}

export function useScanProgress(enabled: boolean) {
  return useQuery<ScanProgress>({
    queryKey: ['scan-progress'],
    queryFn: () => api.get('/api/scan-progress').then(r => r.data),
    staleTime: 0,
    enabled,
    refetchInterval: enabled ? 3_000 : false,
  })
}

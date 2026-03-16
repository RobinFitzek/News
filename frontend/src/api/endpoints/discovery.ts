import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { DiscoveryStats } from '@/types/api'

export function useDiscoveryStats() {
  return useQuery<DiscoveryStats>({
    queryKey: ['discovery-stats'],
    queryFn: () => api.get('/api/discovery/stats').then(r => r.data),
    staleTime: 60_000,
  })
}

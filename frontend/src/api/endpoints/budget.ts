import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { BudgetStatus } from '@/types/api'

export function useBudget() {
  return useQuery<BudgetStatus>({
    queryKey: ['budget'],
    queryFn: () => api.get('/api/budget/status').then(r => r.data),
    staleTime: 60_000,
    refetchInterval: 60_000,
  })
}

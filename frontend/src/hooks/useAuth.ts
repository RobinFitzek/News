import { useQuery } from '@tanstack/react-query'
import api from '@/api/client'
import type { ApiStatus } from '@/types/api'

export function useAuth() {
  const query = useQuery<ApiStatus>({
    queryKey: ['auth'],
    queryFn: () => api.get('/api/status').then(r => r.data),
    retry: false,
    staleTime: Infinity,
  })

  return {
    isAuthenticated: !query.isError && !query.isLoading,
    isLoading: query.isLoading,
    isError: query.isError,
  }
}

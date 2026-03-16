import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'
import type { Plugin } from '@/types/api'

export function usePlugins() {
  return useQuery<{ plugins: Plugin[] }>({
    queryKey: ['plugins'],
    queryFn: () => api.get('/api/plugins').then(r => r.data),
    staleTime: 60_000,
  })
}

export function useTogglePlugin() {
  return useMutation({
    mutationFn: (id: string) => api.post(`/api/plugins/${id}/toggle`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['plugins'] }),
  })
}

export function useRunPlugin() {
  return useMutation({
    mutationFn: (id: string) => api.post(`/api/plugins/${id}/run`).then(r => r.data),
  })
}

import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'
import type { PersonalKey } from '@/types/api'

export function usePersonalKeys() {
  return useQuery<{ keys: PersonalKey[] }>({
    queryKey: ['personal-keys'],
    queryFn: () => api.get('/api/personal-keys').then(r => r.data),
    staleTime: 60_000,
  })
}

export function useCreatePersonalKey() {
  return useMutation({
    mutationFn: (data: { label: string; scope: string }) =>
      api.post('/api/personal-keys', data).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['personal-keys'] }),
  })
}

export function useRevokePersonalKey() {
  return useMutation({
    mutationFn: (id: string) => api.delete(`/api/personal-keys/${id}`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['personal-keys'] }),
  })
}

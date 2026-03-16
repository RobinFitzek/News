import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'
import type { Provider } from '@/types/api'

export function useProviders() {
  return useQuery<{ providers: Provider[] }>({
    queryKey: ['providers'],
    queryFn: () => api.get('/api/providers').then(r => r.data),
    staleTime: 60_000,
  })
}

export function useDeleteProvider() {
  return useMutation({
    mutationFn: (id: string) => api.delete(`/api/providers/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['providers'] }),
  })
}

export function useTestProvider() {
  return useMutation({
    mutationFn: (id: string) => api.post(`/api/providers/${id}/test`).then(r => r.data),
  })
}

export function useCreateProvider() {
  return useMutation({
    mutationFn: (data: Partial<Provider> & { api_key?: string }) =>
      api.post('/api/providers', data).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['providers'] }),
  })
}

export function useUpdateProvider() {
  return useMutation({
    mutationFn: ({ id, ...data }: Partial<Provider> & { id: string; api_key?: string }) =>
      api.put(`/api/providers/${id}`, data).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['providers'] }),
  })
}

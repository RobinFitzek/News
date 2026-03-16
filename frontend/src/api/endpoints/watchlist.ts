import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'
import type { WatchlistItem } from '@/types/api'

export function useWatchlist() {
  return useQuery<WatchlistItem[]>({
    queryKey: ['watchlist'],
    queryFn: () => api.get('/api/watchlist').then(r => r.data),
    staleTime: 30_000,
  })
}

export function useAddToWatchlist() {
  return useMutation({
    mutationFn: (data: { ticker: string; name?: string; tier?: string }) =>
      api.post('/api/watchlist', data).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['watchlist'] }),
  })
}

export function useRemoveFromWatchlist() {
  return useMutation({
    mutationFn: (ticker: string) => api.delete(`/api/watchlist/${ticker}`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['watchlist'] }),
  })
}

export function useSaveWatchlistNote() {
  return useMutation({
    mutationFn: ({ ticker, note }: { ticker: string; note: string }) =>
      api.post('/api/watchlist/note', { ticker, note }).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['watchlist'] }),
  })
}

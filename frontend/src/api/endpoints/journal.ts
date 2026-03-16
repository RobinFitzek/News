import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'

// ── Types ────────────────────────────────────────────────────────────────────

export type JournalEntryType = 'ENTRY' | 'EXIT' | 'NOTE' | 'ALERT'

export interface JournalEntry {
  id: number
  ticker: string
  type: JournalEntryType
  notes: string
  price: number | null
  exit_price: number | null
  created_at: string
  closed_at: string | null
  pnl: number | null
}

export interface JournalResponse {
  entries: JournalEntry[]
}

export interface AddJournalEntryPayload {
  ticker: string
  type: JournalEntryType
  notes: string
  price?: number | null
}

export interface CloseJournalEntryPayload {
  exit_price?: number | null
  notes?: string
}

// ── Hooks ─────────────────────────────────────────────────────────────────────

export function useJournal(ticker?: string) {
  return useQuery<JournalResponse>({
    queryKey: ['journal', ticker ?? ''],
    queryFn: () => {
      const params = ticker ? `?ticker=${encodeURIComponent(ticker)}` : ''
      return api.get(`/api/journal${params}`).then(r => r.data)
    },
    staleTime: 30_000,
  })
}

export function useAddJournalEntry() {
  return useMutation({
    mutationFn: (payload: AddJournalEntryPayload) =>
      api.post('/api/journal/add', payload).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['journal'] }),
  })
}

export function useCloseJournalEntry() {
  return useMutation({
    mutationFn: ({ id, ...payload }: CloseJournalEntryPayload & { id: number }) =>
      api.post(`/api/journal/${id}/close`, payload).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['journal'] }),
  })
}

export function useDeleteJournalEntry() {
  return useMutation({
    mutationFn: (id: number) =>
      api.post(`/api/journal/${id}/delete`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['journal'] }),
  })
}

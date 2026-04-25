import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'
import type { SettingsData, SignalAccuracy } from '@/types/api'

export function useSettingsData() {
  return useQuery<SettingsData>({
    queryKey: ['settings-data'],
    queryFn: () => api.get('/api/settings-data').then(r => r.data),
    staleTime: 60_000,
  })
}

export function useSaveSettings() {
  return useMutation({
    mutationFn: (data: { section: string; [key: string]: unknown }) =>
      api.post('/api/settings/save', data).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['settings-data'] }),
  })
}

export function useChangePassword() {
  return useMutation({
    mutationFn: (data: {
      current_password: string
      new_password: string
      confirm_password: string
    }) => api.post('/change-password', new URLSearchParams(data as Record<string, string>), {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    }).then(r => r.data),
  })
}

export function useSignalAccuracy() {
  return useQuery<SignalAccuracy>({
    queryKey: ['signal-accuracy'],
    queryFn: () => api.get('/api/signal-accuracy').then(r => r.data),
    staleTime: 60_000,
  })
}

export function useSessions() {
  return useQuery({
    queryKey: ['sessions'],
    queryFn: () => api.get('/api/sessions').then(r => r.data),
    staleTime: 30_000,
  })
}

export function useLogoutOtherSessions() {
  return useMutation({
    mutationFn: () => api.post('/api/sessions/logout-others').then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sessions'] }),
  })
}

export function useLogoutSession() {
  return useMutation({
    mutationFn: (sessionId: string) =>
      api.post(`/api/sessions/logout/${sessionId}`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sessions'] }),
  })
}

export function useTestTelegram() {
  return useMutation({
    mutationFn: (data: { telegram_bot_token: string; telegram_chat_id: string }) =>
      api.post('/api/notifications/test-telegram', data).then(r => r.data),
  })
}

export function useTestDiscord() {
  return useMutation({
    mutationFn: (data: { discord_webhook_url: string }) =>
      api.post('/api/notifications/test-discord', data).then(r => r.data),
  })
}

export function useTestEmail() {
  return useMutation({
    mutationFn: () => api.post('/api/notifications/test-email').then(r => r.data),
  })
}

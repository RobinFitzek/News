import { create } from 'zustand'
import type { ToastMessage } from '@/types/api'

interface ToastStore {
  toasts: ToastMessage[]
  addToast: (message: string, type?: ToastMessage['type'], duration?: number) => void
  removeToast: (id: string) => void
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  addToast: (message, type = 'info', duration = 4000) => {
    const id = crypto.randomUUID()
    set(s => ({ toasts: [...s.toasts, { id, message, type, duration }] }))
    if (duration > 0) {
      setTimeout(() => {
        set(s => ({ toasts: s.toasts.filter(t => t.id !== id) }))
      }, duration)
    }
  },
  removeToast: (id) => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })),
}))

// Convenience export for imperative usage outside React
export const toast = {
  info:    (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'info', dur),
  success: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'success', dur),
  error:   (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'error', dur),
  warning: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'warning', dur),
}

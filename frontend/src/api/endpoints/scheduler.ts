import { useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'

function invalidateStatus() {
  queryClient.invalidateQueries({ queryKey: ['status'] })
  queryClient.invalidateQueries({ queryKey: ['scan-progress'] })
}

export function useStartScheduler() {
  return useMutation({
    mutationFn: () => api.post('/api/scheduler/start'),
    onSuccess: invalidateStatus,
  })
}

export function useStopScheduler() {
  return useMutation({
    mutationFn: () => api.post('/api/scheduler/stop'),
    onSuccess: invalidateStatus,
  })
}

export function useScanNow() {
  return useMutation({
    mutationFn: () => api.post('/api/scheduler/run-now'),
    onSuccess: invalidateStatus,
  })
}

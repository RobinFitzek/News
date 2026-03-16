import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { GeoScan, GeoExposure } from '@/types/api'

export function useGeopolitical() {
  return useQuery<{ scan: GeoScan | null }>({
    queryKey: ['geopolitical'],
    queryFn: () => api.get('/api/geopolitical').then(r => r.data),
    staleTime: 300_000,
  })
}

export function useGeoExposure() {
  return useQuery<{ exposures: GeoExposure[] }>({
    queryKey: ['geo-exposure'],
    queryFn: () => api.get('/api/geopolitical/exposure').then(r => r.data),
    staleTime: 300_000,
  })
}

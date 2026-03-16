import { usePlugins, useTogglePlugin, useRunPlugin } from '@/api/endpoints/plugins'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

export function PanelPlugins() {
  const { data, isLoading } = usePlugins()
  const toggleMut = useTogglePlugin()
  const runMut = useRunPlugin()
  const { addToast } = useToastStore()

  async function handleRun(id: string) {
    try {
      await runMut.mutateAsync(id)
      addToast('Plugin executed', 'success')
    } catch {
      addToast('Plugin execution failed', 'error')
    }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Plugins</h2>

      {isLoading && <div className={styles.loading}><Spinner /></div>}

      <div className={styles.list}>
        {data?.plugins.map(p => (
          <Card key={p.id} className={styles.section}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 'var(--space-4)' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 4 }}>
                  <span style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: 'var(--text-primary)' }}>
                    {p.name}
                  </span>
                  <Badge variant="ghost">{p.version}</Badge>
                  <Badge variant={p.is_enabled ? 'success' : 'ghost'}>
                    {p.is_enabled ? 'Enabled' : 'Disabled'}
                  </Badge>
                </div>
                <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>{p.description}</p>
              </div>
              <div style={{ display: 'flex', gap: 'var(--space-2)', flexShrink: 0 }}>
                <Button
                  variant="secondary"
                  size="sm"
                  loading={toggleMut.isPending}
                  onClick={() => toggleMut.mutate(p.id)}
                >
                  {p.is_enabled ? 'Disable' : 'Enable'}
                </Button>
                {p.is_enabled && (
                  <Button
                    variant="ghost"
                    size="sm"
                    loading={runMut.isPending}
                    onClick={() => handleRun(p.id)}
                  >
                    Run
                  </Button>
                )}
              </div>
            </div>
          </Card>
        ))}

        {data?.plugins.length === 0 && (
          <p className={styles.empty}>No plugins installed.</p>
        )}
      </div>
    </div>
  )
}

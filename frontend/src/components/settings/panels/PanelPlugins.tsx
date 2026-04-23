import { useRef } from 'react'
import { usePlugins, useTogglePlugin, useRunPlugin, useInstallPlugin } from '@/api/endpoints/plugins'
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
  const installMut = useInstallPlugin()
  const { addToast } = useToastStore()
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleRun(id: string) {
    try {
      await runMut.mutateAsync(id)
      addToast('Plugin executed', 'success')
    } catch {
      addToast('Plugin execution failed', 'error')
    }
  }

  async function handleInstall(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      await installMut.mutateAsync(file)
      addToast(`Plugin "${file.name}" installed`, 'success')
    } catch {
      addToast('Installation failed', 'error')
    } finally {
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Plugins</h2>

      <Card className={styles.section}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 className={styles.sectionTitle} style={{ margin: 0 }}>Install Plugin</h3>
          <Button variant="secondary" size="sm" loading={installMut.isPending} onClick={() => fileRef.current?.click()}>
            Upload .py file
          </Button>
        </div>
        <input ref={fileRef} type="file" accept=".py,.zip" style={{ display: 'none' }} onChange={handleInstall} />
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)', marginTop: 'var(--space-2)' }}>
          Upload a .py plugin file. It will be validated and added to the plugin list.
        </p>
      </Card>

      {isLoading && <div className={styles.loading}><Spinner /></div>}

      <div className={styles.list}>
        {data?.plugins.map(p => (
          <Card key={p.id} className={styles.section}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 'var(--space-4)' }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 4, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: 'var(--text-primary)' }}>
                    {p.name}
                  </span>
                  <Badge variant="ghost">{p.version}</Badge>
                  {p.plugin_type && (
                    <Badge variant="ghost" style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)' }}>
                      {p.plugin_type}
                    </Badge>
                  )}
                  <Badge variant={p.is_enabled ? 'success' : 'ghost'}>
                    {p.is_enabled ? 'Enabled' : 'Disabled'}
                  </Badge>
                </div>
                <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>{p.description}</p>
                {p.last_error && (
                  <div style={{ marginTop: 'var(--space-2)', padding: 'var(--space-2) var(--space-3)', background: 'rgba(212,99,75,0.08)', border: '1px solid rgba(212,99,75,0.25)', fontSize: 'var(--text-xs)', color: 'var(--signal-negative)', fontFamily: 'var(--font-mono)', wordBreak: 'break-word' }}>
                    {p.last_error}
                  </div>
                )}
                {p.last_run && (
                  <div style={{ marginTop: 'var(--space-1)', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                    Last run: {p.last_run}
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', gap: 'var(--space-2)', flexShrink: 0 }}>
                <Button variant="secondary" size="sm" loading={toggleMut.isPending} onClick={() => toggleMut.mutate(p.id)}>
                  {p.is_enabled ? 'Disable' : 'Enable'}
                </Button>
                {p.is_enabled && (
                  <Button variant="ghost" size="sm" loading={runMut.isPending} onClick={() => handleRun(p.id)}>
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

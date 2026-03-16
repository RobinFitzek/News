import { useState } from 'react'
import { useProviders, useDeleteProvider, useTestProvider } from '@/api/endpoints/providers'
import { usePersonalKeys, useCreatePersonalKey, useRevokePersonalKey } from '@/api/endpoints/personalKeys'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Modal } from '@/components/ui/Modal'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

export function PanelAPIConnections() {
  const { data: provData, isLoading: provLoading } = useProviders()
  const { data: keysData, isLoading: keysLoading } = usePersonalKeys()
  const deleteProv = useDeleteProvider()
  const testProv = useTestProvider()
  const createKey = useCreatePersonalKey()
  const revokeKey = useRevokePersonalKey()
  const { addToast } = useToastStore()

  const [newKeyModal, setNewKeyModal] = useState(false)
  const [newKeyLabel, setNewKeyLabel] = useState('')
  const [newKeyScope, setNewKeyScope] = useState('read')
  const [createdKey, setCreatedKey] = useState<string | null>(null)

  const [testingId, setTestingId] = useState<string | null>(null)

  async function handleTest(id: string) {
    setTestingId(id)
    try {
      const result = await testProv.mutateAsync(id)
      addToast(result.message ?? 'Connection OK', 'success')
    } catch {
      addToast('Connection test failed', 'error')
    } finally {
      setTestingId(null)
    }
  }

  async function handleCreateKey() {
    try {
      const result = await createKey.mutateAsync({ label: newKeyLabel, scope: newKeyScope })
      setCreatedKey(result.raw_key)
      setNewKeyLabel('')
    } catch {
      addToast('Failed to create key', 'error')
    }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>API Connections</h2>

      {/* AI Providers */}
      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>AI Providers</h3>
        {provLoading && <div className={styles.loading}><Spinner /></div>}
        <div className={styles.list}>
          {provData?.providers.map(p => (
            <div key={p.id} className={styles.providerRow}>
              <div className={styles.providerInfo}>
                <span className={styles.providerName}>{p.name}</span>
                <Badge variant={p.is_configured ? 'success' : 'ghost'}>
                  {p.is_configured ? 'Configured' : 'Not set'}
                </Badge>
                {p.test_status === 'ok' && <Badge variant="success">OK</Badge>}
                {p.test_status === 'error' && <Badge variant="danger">Error</Badge>}
              </div>
              <div className={styles.providerActions}>
                <Button
                  variant="ghost"
                  size="sm"
                  loading={testingId === p.id}
                  onClick={() => handleTest(p.id)}
                >
                  Test
                </Button>
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => deleteProv.mutate(p.id)}
                >
                  Remove
                </Button>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Personal API Keys */}
      <Card className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Personal API Keys</h3>
          <Button
            variant="primary"
            size="sm"
            onClick={() => setNewKeyModal(true)}
          >
            + New Key
          </Button>
        </div>

        {keysLoading && <div className={styles.loading}><Spinner /></div>}

        <div className={styles.list}>
          {keysData?.keys.map(k => (
            <div key={k.id} className={styles.keyRow}>
              <div className={styles.keyInfo}>
                <span className={styles.keyLabel}>{k.label}</span>
                <code className={styles.keyMask}>{k.masked_key}</code>
                <Badge variant="ghost">{k.scope}</Badge>
              </div>
              <Button
                variant="danger"
                size="sm"
                onClick={() => revokeKey.mutate(k.id)}
              >
                Revoke
              </Button>
            </div>
          ))}

          {keysData?.keys.length === 0 && (
            <p className={styles.empty}>No API keys created yet.</p>
          )}
        </div>
      </Card>

      {/* New key modal */}
      <Modal
        open={newKeyModal}
        onClose={() => { setNewKeyModal(false); setCreatedKey(null) }}
        title="Create API Key"
        size="sm"
      >
        {createdKey ? (
          <div className={styles.createdKey}>
            <p className={styles.createdKeyNote}>
              Copy this key now — it will only be shown once.
            </p>
            <code className={styles.rawKey}>{createdKey}</code>
            <Button
              variant="primary"
              size="md"
              onClick={() => {
                navigator.clipboard.writeText(createdKey)
                addToast('Key copied', 'success')
              }}
            >
              Copy Key
            </Button>
          </div>
        ) : (
          <div className={styles.form}>
            <div className={styles.field}>
              <label className={styles.fieldLabel}>Label</label>
              <input
                className={styles.input}
                placeholder="e.g. My Script"
                value={newKeyLabel}
                onChange={e => setNewKeyLabel(e.target.value)}
              />
            </div>
            <div className={styles.field}>
              <label className={styles.fieldLabel}>Scope</label>
              <select
                className={styles.input}
                value={newKeyScope}
                onChange={e => setNewKeyScope(e.target.value)}
              >
                <option value="read">Read</option>
                <option value="write">Write</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <Button
              variant="primary"
              size="md"
              loading={createKey.isPending}
              onClick={handleCreateKey}
              disabled={!newKeyLabel.trim()}
            >
              Create Key
            </Button>
          </div>
        )}
      </Modal>
    </div>
  )
}

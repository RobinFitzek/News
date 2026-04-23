import { useEffect, useState } from 'react'
import { useSettingsData, useSaveSettings } from '@/api/endpoints/settings'
import { useAutoTradeTrustGate } from '@/api/endpoints/autoTrade'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className={styles.toggle}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
      <span className={styles.toggleTrack} />
      <span className={styles.toggleThumb} />
    </label>
  )
}

function NumSlider({ label, sub, value, onChange, min, max, step, unit }: {
  label: string; sub?: string; value: number; onChange: (v: number) => void
  min: number; max: number; step?: number; unit?: string
}) {
  return (
    <div className={styles.field}>
      <div className={styles.fieldLabel}>{label}</div>
      {sub && <div className={styles.sectionSub} style={{ marginBottom: 'var(--space-2)' }}>{sub}</div>}
      <div className={styles.sliderRow}>
        <input type="range" min={min} max={max} step={step ?? 1} value={value} onChange={e => onChange(Number(e.target.value))} className={styles.slider} />
        <span className={styles.sliderValue}>{value}{unit ?? ''}</span>
      </div>
    </div>
  )
}

export function PanelAutoTrading() {
  const { data: settings } = useSettingsData()
  const { data: trustGate } = useAutoTradeTrustGate()
  const saveMut = useSaveSettings()
  const { addToast } = useToastStore()

  const [enabled, setEnabled] = useState(false)
  const [mode, setMode] = useState('paper')
  const [signalFilter, setSignalFilter] = useState('STRONG')
  const [takeProfitPct, setTakeProfitPct] = useState(8)
  const [stopLossPct, setStopLossPct] = useState(4)
  const [maxDaysOpen, setMaxDaysOpen] = useState(30)
  const [positionSizePct, setPositionSizePct] = useState(5)
  const [maxOpenPositions, setMaxOpenPositions] = useState(10)
  const [requireConfirm, setRequireConfirm] = useState(true)
  const [minTrustTrades, setMinTrustTrades] = useState(20)
  const [minWinRate, setMinWinRate] = useState(55)
  const [trustOverride, setTrustOverride] = useState(false)

  // Alpaca
  const [alpacaKey, setAlpacaKey] = useState('')
  const [alpacaSecret, setAlpacaSecret] = useState('')
  const [alpacaUrl, setAlpacaUrl] = useState('https://paper-api.alpaca.markets')

  // IBKR
  const [ibkrHost, setIbkrHost] = useState('127.0.0.1')
  const [ibkrPort, setIbkrPort] = useState(7497)
  const [ibkrClientId, setIbkrClientId] = useState(1)

  useEffect(() => {
    if (!settings) return
    const s = settings as Record<string, unknown>
    setEnabled(Boolean(s.auto_trade_enabled))
    setMode(String(s.auto_trade_mode ?? 'paper'))
    setSignalFilter(String(s.auto_trade_signal_filter ?? 'STRONG'))
    setTakeProfitPct(Number(s.auto_trade_take_profit_pct ?? 8))
    setStopLossPct(Number(s.auto_trade_stop_loss_pct ?? 4))
    setMaxDaysOpen(Number(s.auto_trade_max_days_open ?? 30))
    setPositionSizePct(Number(s.auto_trade_position_size_pct ?? 5))
    setMaxOpenPositions(Number(s.auto_trade_max_open_positions ?? 10))
    setRequireConfirm(s.auto_trade_require_confirm !== false)
    setMinTrustTrades(Number(s.auto_trade_min_trust_trades ?? 20))
    setMinWinRate(Number(s.auto_trade_min_trust_win_rate ?? 55))
    setTrustOverride(Boolean(s.auto_trade_trust_override))
    setAlpacaUrl(String(s.auto_trade_alpaca_base_url ?? 'https://paper-api.alpaca.markets'))
    setIbkrHost(String(s.auto_trade_ibkr_host ?? '127.0.0.1'))
    setIbkrPort(Number(s.auto_trade_ibkr_port ?? 7497))
    setIbkrClientId(Number(s.auto_trade_ibkr_client_id ?? 1))
  }, [settings])

  async function handleSave() {
    try {
      await saveMut.mutateAsync({
        section: 'auto_trading',
        auto_trade_enabled: enabled,
        auto_trade_mode: mode,
        auto_trade_signal_filter: signalFilter,
        auto_trade_take_profit_pct: takeProfitPct,
        auto_trade_stop_loss_pct: stopLossPct,
        auto_trade_max_days_open: maxDaysOpen,
        auto_trade_position_size_pct: positionSizePct,
        auto_trade_max_open_positions: maxOpenPositions,
        auto_trade_require_confirm: requireConfirm,
        auto_trade_min_trust_trades: minTrustTrades,
        auto_trade_min_trust_win_rate: minWinRate,
        auto_trade_trust_override: trustOverride,
        ...(mode === 'alpaca' ? {
          auto_trade_alpaca_api_key: alpacaKey || undefined,
          auto_trade_alpaca_secret: alpacaSecret || undefined,
          auto_trade_alpaca_base_url: alpacaUrl,
        } : {}),
        ...(mode === 'ibkr' ? {
          auto_trade_ibkr_host: ibkrHost,
          auto_trade_ibkr_port: String(ibkrPort),
          auto_trade_ibkr_client_id: String(ibkrClientId),
        } : {}),
      })
      addToast('Auto-trading settings saved', 'success')
    } catch { addToast('Failed to save', 'error') }
  }

  const gates = (trustGate as Record<string, unknown> | undefined)?.gates as Record<string, boolean> | undefined
  const allPassed = (trustGate as Record<string, unknown> | undefined)?.all_passed as boolean | undefined

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Auto-Trading</h2>

      {enabled && (
        <div style={{ padding: 'var(--space-3) var(--space-4)', background: 'rgba(212,99,75,0.08)', border: '1px solid rgba(212,99,75,0.3)', fontSize: 'var(--text-sm)', color: 'var(--signal-negative)' }}>
          ⚠ Auto-trading is enabled. Real trades may be executed based on AI signals.
        </div>
      )}

      <Card className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Master Switch</h3>
          <Toggle checked={enabled} onChange={setEnabled} />
        </div>

        <div className={styles.form} style={{ marginTop: 'var(--space-4)' }}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Mode</div>
            <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
              {['paper', 'alpaca', 'ibkr'].map(m => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  style={{
                    padding: '6px 14px',
                    background: mode === m ? 'var(--text-primary)' : 'var(--bg-tertiary)',
                    color: mode === m ? 'var(--bg-primary)' : 'var(--text-secondary)',
                    border: '1px solid var(--border-primary)',
                    cursor: 'pointer',
                    fontSize: 'var(--text-sm)',
                    fontFamily: 'var(--font-mono)',
                    textTransform: 'uppercase' as const,
                    letterSpacing: '0.06em',
                  }}
                >{m}</button>
              ))}
            </div>
          </div>

          <div className={styles.field}>
            <div className={styles.fieldLabel}>Signal filter</div>
            <select className={styles.input} value={signalFilter} onChange={e => setSignalFilter(e.target.value)}>
              <option value="STRONG">STRONG only (STRONG_BUY / STRONG_SELL)</option>
              <option value="ALL">All signals (BUY / SELL included)</option>
            </select>
          </div>

          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Require confirmation</div>
              <div className={styles.toggleSub}>Send email/Telegram before each trade executes</div>
            </div>
            <Toggle checked={requireConfirm} onChange={setRequireConfirm} />
          </div>
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Risk Parameters</h3>
        <div className={styles.form}>
          <NumSlider label="Take profit" value={takeProfitPct} onChange={setTakeProfitPct} min={1} max={50} unit="%" />
          <NumSlider label="Stop loss" value={stopLossPct} onChange={setStopLossPct} min={1} max={25} unit="%" />
          <NumSlider label="Max days open" value={maxDaysOpen} onChange={setMaxDaysOpen} min={1} max={90} unit="d" />
          <NumSlider label="Position size" value={positionSizePct} onChange={setPositionSizePct} min={1} max={20} unit="%" />
          <NumSlider label="Max open positions" value={maxOpenPositions} onChange={setMaxOpenPositions} min={1} max={50} />
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Trust Gate</h3>
        {gates && (
          <div style={{ marginBottom: 'var(--space-4)', display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            {Object.entries(gates).map(([key, passed]) => (
              <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', fontSize: 'var(--text-sm)' }}>
                <Badge variant={passed ? 'success' : 'danger'}>{passed ? '✓' : '✗'}</Badge>
                <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)' }}>{key}</span>
              </div>
            ))}
            <div style={{ marginTop: 'var(--space-2)', fontSize: 'var(--text-sm)', color: allPassed ? 'var(--signal-positive)' : 'var(--signal-warning)' }}>
              {allPassed ? 'All gates passed — trading allowed' : 'Trust gate blocked — not enough track record'}
            </div>
          </div>
        )}
        <div className={styles.form}>
          <NumSlider label="Min trades for trust" value={minTrustTrades} onChange={setMinTrustTrades} min={5} max={200} step={5} />
          <NumSlider label="Min win rate" value={minWinRate} onChange={setMinWinRate} min={40} max={80} unit="%" />
          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Admin trust override</div>
              <div className={styles.toggleSub}>Bypass trust gate (use with caution)</div>
            </div>
            <Toggle checked={trustOverride} onChange={setTrustOverride} />
          </div>
        </div>
      </Card>

      {mode === 'alpaca' && (
        <Card className={styles.section}>
          <h3 className={styles.sectionTitle}>Alpaca Credentials</h3>
          <div className={styles.form}>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>API Key</div>
              <input className={styles.input} value={alpacaKey} onChange={e => setAlpacaKey(e.target.value)} placeholder="(unchanged)" autoComplete="new-password" />
            </div>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Secret</div>
              <input className={styles.input} type="password" value={alpacaSecret} onChange={e => setAlpacaSecret(e.target.value)} placeholder="(unchanged)" autoComplete="new-password" />
            </div>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Base URL</div>
              <input className={styles.input} value={alpacaUrl} onChange={e => setAlpacaUrl(e.target.value)} />
            </div>
          </div>
        </Card>
      )}

      {mode === 'ibkr' && (
        <Card className={styles.section}>
          <h3 className={styles.sectionTitle}>IBKR TWS / Gateway</h3>
          <div className={styles.form}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: 'var(--space-3)' }}>
              <div className={styles.field}>
                <div className={styles.fieldLabel}>Host</div>
                <input className={styles.input} value={ibkrHost} onChange={e => setIbkrHost(e.target.value)} />
              </div>
              <div className={styles.field}>
                <div className={styles.fieldLabel}>Port</div>
                <input className={styles.input} type="number" value={ibkrPort} onChange={e => setIbkrPort(Number(e.target.value))} style={{ width: 90 }} />
              </div>
              <div className={styles.field}>
                <div className={styles.fieldLabel}>Client ID</div>
                <input className={styles.input} type="number" value={ibkrClientId} onChange={e => setIbkrClientId(Number(e.target.value))} style={{ width: 80 }} />
              </div>
            </div>
          </div>
        </Card>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button variant="primary" size="md" loading={saveMut.isPending} onClick={handleSave}>Save Auto-Trading</Button>
      </div>
    </div>
  )
}

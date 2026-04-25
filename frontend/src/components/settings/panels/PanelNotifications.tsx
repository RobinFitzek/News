import { useEffect, useState } from 'react'
import { useSettingsData, useSaveSettings, useTestEmail, useTestTelegram, useTestDiscord } from '@/api/endpoints/settings'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
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

export function PanelNotifications() {
  const { data: settings } = useSettingsData()
  const saveMut = useSaveSettings()
  const testEmail = useTestEmail()
  const testTelegram = useTestTelegram()
  const testDiscord = useTestDiscord()
  const { addToast } = useToastStore()

  // Email
  const [emailEnabled, setEmailEnabled] = useState(false)
  const [emailRecipient, setEmailRecipient] = useState('')
  const [smtpHost, setSmtpHost] = useState('smtp.gmail.com')
  const [smtpPort, setSmtpPort] = useState(587)
  const [smtpUser, setSmtpUser] = useState('')
  const [smtpPassword, setSmtpPassword] = useState('')
  const [notifyStrong, setNotifyStrong] = useState(true)
  const [dailySummary, setDailySummary] = useState(true)
  const [summaryTime, setSummaryTime] = useState('20:00')

  // Telegram
  const [telegramEnabled, setTelegramEnabled] = useState(false)
  const [telegramToken, setTelegramToken] = useState('')
  const [telegramChatId, setTelegramChatId] = useState('')
  const [telegramBotEnabled, setTelegramBotEnabled] = useState(false)

  // Discord
  const [discordEnabled, setDiscordEnabled] = useState(false)
  const [discordWebhook, setDiscordWebhook] = useState('')

  // Alert deduplication
  const [alertCooldown, setAlertCooldown] = useState(24)
  const [intradayTrigger, setIntradayTrigger] = useState(3.0)

  // Weekly report
  const [weeklyLetter, setWeeklyLetter] = useState(false)

  useEffect(() => {
    if (!settings) return
    const s = settings as Record<string, unknown>
    setEmailEnabled(Boolean(s.email_enabled))
    setEmailRecipient(String(s.email_recipient ?? ''))
    setSmtpHost(String(s.email_smtp_host ?? 'smtp.gmail.com'))
    setSmtpPort(Number(s.email_smtp_port ?? 587))
    setSmtpUser(String(s.email_smtp_user ?? ''))
    setNotifyStrong(s.notify_on_strong_signals !== false)
    setDailySummary(s.daily_summary_enabled !== false)
    setSummaryTime(String(s.daily_summary_time ?? '20:00'))
    setTelegramEnabled(Boolean(s.telegram_enabled))
    setTelegramToken(String(s.telegram_bot_token ?? ''))
    setTelegramChatId(String(s.telegram_chat_id ?? ''))
    setTelegramBotEnabled(Boolean(s.telegram_bot_enabled))
    setDiscordEnabled(Boolean(s.discord_enabled))
    setDiscordWebhook(String(s.discord_webhook_url ?? ''))
    setAlertCooldown(Number(s.alert_cooldown_hours ?? 24))
    setIntradayTrigger(Number(s.intraday_trigger_pct ?? 3.0))
    setWeeklyLetter(Boolean(s.weekly_letter_enabled))
  }, [settings])

  async function saveEmail() {
    try {
      await saveMut.mutateAsync({
        section: 'notifications_email',
        email_enabled: emailEnabled,
        email_recipient: emailRecipient,
        email_smtp_host: smtpHost,
        email_smtp_port: smtpPort,
        email_smtp_user: smtpUser,
        email_smtp_password: smtpPassword || undefined,
        notify_on_strong_signals: notifyStrong,
        daily_summary_enabled: dailySummary,
        daily_summary_time: summaryTime,
      })
      addToast('Email settings saved', 'success')
    } catch { addToast('Failed to save', 'error') }
  }

  async function saveTelegram() {
    try {
      await saveMut.mutateAsync({
        section: 'notifications_telegram',
        telegram_enabled: telegramEnabled,
        telegram_bot_token: telegramToken,
        telegram_chat_id: telegramChatId,
        telegram_bot_enabled: telegramBotEnabled,
      })
      addToast('Telegram settings saved', 'success')
    } catch { addToast('Failed to save', 'error') }
  }

  async function saveDiscord() {
    try {
      await saveMut.mutateAsync({
        section: 'notifications_discord',
        discord_enabled: discordEnabled,
        discord_webhook_url: discordWebhook,
      })
      addToast('Discord settings saved', 'success')
    } catch { addToast('Failed to save', 'error') }
  }

  async function saveAlerts() {
    try {
      await saveMut.mutateAsync({
        section: 'notifications_alerts',
        alert_cooldown_hours: alertCooldown,
        intraday_trigger_pct: intradayTrigger,
        weekly_letter_enabled: weeklyLetter,
      })
      addToast('Alert settings saved', 'success')
    } catch { addToast('Failed to save', 'error') }
  }

  async function handleTestEmail() {
    const r = await testEmail.mutateAsync()
    addToast(r.message || (r.success ? 'Test email sent' : 'Failed'), r.success ? 'success' : 'error')
  }

  async function handleTestTelegram() {
    const r = await testTelegram.mutateAsync({ telegram_bot_token: telegramToken, telegram_chat_id: telegramChatId })
    addToast(r.message || (r.success ? 'Test sent' : 'Failed'), r.success ? 'success' : 'error')
  }

  async function handleTestDiscord() {
    const r = await testDiscord.mutateAsync({ discord_webhook_url: discordWebhook })
    addToast(r.message || (r.success ? 'Test sent' : 'Failed'), r.success ? 'success' : 'error')
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Notifications</h2>

      {/* Email */}
      <Card className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Email</h3>
          <Toggle checked={emailEnabled} onChange={setEmailEnabled} />
        </div>
        {emailEnabled && (
          <div className={styles.form}>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Recipient</div>
              <input className={styles.input} type="email" value={emailRecipient} onChange={e => setEmailRecipient(e.target.value)} placeholder="you@example.com" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 'var(--space-3)' }}>
              <div className={styles.field}>
                <div className={styles.fieldLabel}>SMTP Host</div>
                <input className={styles.input} value={smtpHost} onChange={e => setSmtpHost(e.target.value)} />
              </div>
              <div className={styles.field}>
                <div className={styles.fieldLabel}>Port</div>
                <input className={styles.input} type="number" value={smtpPort} onChange={e => setSmtpPort(Number(e.target.value))} style={{ width: 80 }} />
              </div>
            </div>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>SMTP Username</div>
              <input className={styles.input} value={smtpUser} onChange={e => setSmtpUser(e.target.value)} />
            </div>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>SMTP Password</div>
              <input className={styles.input} type="password" value={smtpPassword} onChange={e => setSmtpPassword(e.target.value)} placeholder="(unchanged)" autoComplete="new-password" />
            </div>
            <div className={styles.toggleRow}>
              <div>
                <div className={styles.toggleLabel}>Notify on strong signals</div>
                <div className={styles.toggleSub}>Send email on STRONG_BUY / STRONG_SELL</div>
              </div>
              <Toggle checked={notifyStrong} onChange={setNotifyStrong} />
            </div>
            <div className={styles.toggleRow}>
              <div>
                <div className={styles.toggleLabel}>Daily summary</div>
              </div>
              <Toggle checked={dailySummary} onChange={setDailySummary} />
            </div>
            {dailySummary && (
              <div className={styles.field}>
                <div className={styles.fieldLabel}>Summary time</div>
                <input className={styles.input} type="time" value={summaryTime} onChange={e => setSummaryTime(e.target.value)} style={{ width: 120 }} />
              </div>
            )}
          </div>
        )}
        <div className={styles.saveRow} style={{ marginTop: 'var(--space-4)' }}>
          <Button variant="ghost" size="sm" loading={testEmail.isPending} onClick={handleTestEmail}>Test</Button>
          <Button variant="primary" size="md" loading={saveMut.isPending} onClick={saveEmail}>Save Email</Button>
        </div>
      </Card>

      {/* Telegram */}
      <Card className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Telegram</h3>
          <Toggle checked={telegramEnabled} onChange={setTelegramEnabled} />
        </div>
        {telegramEnabled && (
          <div className={styles.form}>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Bot Token</div>
              <input className={styles.input} type="password" value={telegramToken} onChange={e => setTelegramToken(e.target.value)} placeholder="1234567890:ABC..." autoComplete="new-password" />
            </div>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Chat ID</div>
              <input className={styles.input} value={telegramChatId} onChange={e => setTelegramChatId(e.target.value)} placeholder="-100123456789" />
            </div>
            <div className={styles.toggleRow}>
              <div>
                <div className={styles.toggleLabel}>Two-way bot commands</div>
                <div className={styles.toggleSub}>/analyze, /watchlist, /toppicks via Telegram</div>
              </div>
              <Toggle checked={telegramBotEnabled} onChange={setTelegramBotEnabled} />
            </div>
          </div>
        )}
        <div className={styles.saveRow} style={{ marginTop: 'var(--space-4)' }}>
          <Button variant="ghost" size="sm" loading={testTelegram.isPending} onClick={handleTestTelegram}>Test</Button>
          <Button variant="primary" size="md" loading={saveMut.isPending} onClick={saveTelegram}>Save Telegram</Button>
        </div>
      </Card>

      {/* Discord */}
      <Card className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Discord</h3>
          <Toggle checked={discordEnabled} onChange={setDiscordEnabled} />
        </div>
        {discordEnabled && (
          <div className={styles.form}>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Webhook URL</div>
              <input className={styles.input} type="password" value={discordWebhook} onChange={e => setDiscordWebhook(e.target.value)} placeholder="https://discord.com/api/webhooks/..." autoComplete="new-password" />
            </div>
          </div>
        )}
        <div className={styles.saveRow} style={{ marginTop: 'var(--space-4)' }}>
          <Button variant="ghost" size="sm" loading={testDiscord.isPending} onClick={handleTestDiscord}>Test</Button>
          <Button variant="primary" size="md" loading={saveMut.isPending} onClick={saveDiscord}>Save Discord</Button>
        </div>
      </Card>

      {/* Alert rules */}
      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Alert Rules</h3>
        <div className={styles.form}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Alert cooldown (hours)</div>
            <div className={styles.sliderRow}>
              <input type="range" min={1} max={168} step={1} value={alertCooldown} onChange={e => setAlertCooldown(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{alertCooldown}h</span>
            </div>
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Intraday breakout threshold (%)</div>
            <div className={styles.sliderRow}>
              <input type="range" min={1} max={10} step={0.5} value={intradayTrigger} onChange={e => setIntradayTrigger(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{intradayTrigger}%</span>
            </div>
          </div>
          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Weekly AI letter</div>
              <div className={styles.toggleSub}>Sunday evening Gemini summary of the week</div>
            </div>
            <Toggle checked={weeklyLetter} onChange={setWeeklyLetter} />
          </div>
          <div className={styles.saveRow}>
            <Button variant="primary" size="md" loading={saveMut.isPending} onClick={saveAlerts}>Save Alert Rules</Button>
          </div>
        </div>
      </Card>
    </div>
  )
}

import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import styles from './ArchitecturePage.module.css'

// ── Section data ───────────────────────────────────────────────────────────

const PIPELINE_STEPS = [
  { step: '1', title: 'Data Fetch', desc: 'Price data via yfinance, news via RSS/Perplexity, SEC filings via EDGAR. Runs on configurable schedule.', color: '#6366f1' },
  { step: '2', title: 'AI Analysis', desc: 'Multi-stage pipeline: Perplexity gathers context, Gemini/Claude produces bull/bear/synthesis with scores.', color: '#8b5cf6' },
  { step: '3', title: 'Signal Generation', desc: 'BUY / SELL / HOLD signal with confidence %, risk score 1-10, and geo-risk assessment.', color: '#10b981' },
  { step: '4', title: 'Risk Gates', desc: 'Trust score check, kill-switch (accuracy < 50%), market regime filter, position sizing limits.', color: '#f59e0b' },
  { step: '5', title: 'Trade Execution', desc: 'Auto-trade proposals with human confirmation gate. Paper or live mode via broker API.', color: '#ef4444' },
]

const DATA_SOURCES = [
  { name: 'yfinance', desc: 'Prices, fundamentals, earnings, dividends', icon: '📊' },
  { name: 'Perplexity', desc: 'Real-time news context, stock discovery', icon: '🔍' },
  { name: 'Gemini / Claude', desc: 'AI analysis, bull/bear cases, synthesis', icon: '🤖' },
  { name: 'SEC EDGAR', desc: 'Insider trades, institutional filings', icon: '📄' },
  { name: 'RSS Feeds', desc: 'Geopolitical events, market headlines', icon: '📡' },
  { name: 'Economic Calendar', desc: 'Central bank events, macro indicators', icon: '📅' },
]

const AI_AGENTS = [
  { name: 'Research Agent', role: 'Gathers real-time context from Perplexity — news, events, market data for each ticker.' },
  { name: 'Analysis Agent', role: 'Produces structured analysis via Gemini/Claude — bull case, bear case, risk score, signal.' },
  { name: 'Cross-Check Agent', role: 'Validates primary analysis against alternative AI providers. Confirms or contradicts signals.' },
  { name: 'Discovery Agent', role: 'Autonomous stock scanning — sector rotation, RSI oversold, breakouts, insider buys, AI trending.' },
  { name: 'Geo Agent', role: '6-hour geopolitical scans. Assesses per-ticker exposure to macro events. Alerts on severity ≥ 8.' },
]

const RISK_GATES = [
  { name: 'Trust Score', desc: 'Pipeline accuracy must be above 50% threshold. Falls below → auto-pause trading.', status: 'critical' },
  { name: 'Kill Switch', desc: 'Hard stop if accuracy drops below configured threshold. Prevents continued losses.', status: 'critical' },
  { name: 'Market Regime', desc: 'Adapts position sizing and signal thresholds based on current regime (bull/bear/volatile).', status: 'warning' },
  { name: 'Position Sizing', desc: 'Max position size, sector concentration limits, stop-loss enforcement.', status: 'info' },
  { name: 'Confirmation Gate', desc: 'Every auto-trade proposal requires human confirmation before execution.', status: 'info' },
]

const AUTO_TRADE_FLOW = [
  { label: 'Signal', desc: 'AI generates BUY/SELL with confidence ≥ threshold' },
  { label: 'Proposal', desc: 'System creates trade proposal with size, entry, stop-loss' },
  { label: 'Risk Check', desc: 'Trust gate + regime + concentration checks pass' },
  { label: 'Confirmation', desc: 'User receives notification, confirms or skips' },
  { label: 'Execution', desc: 'Paper trade or live order via broker API' },
  { label: 'Tracking', desc: 'Position tracked with P&L, exit rules enforced' },
]

// ── Components ─────────────────────────────────────────────────────────────

function FlowArrow() {
  return <div className={styles.flowArrow}>→</div>
}

export function ArchitecturePage() {
  return (
    <>
      <PageHeader
        title="How Stockholm Works"
        subtitle="System architecture and data flow"
      />

      {/* Pipeline flow */}
      <Card className={styles.section} delay={0}>
        <h2 className={styles.sectionTitle}>Analysis Pipeline</h2>
        <p className={styles.sectionDesc}>Every analysis flows through these stages</p>
        <div className={styles.pipelineFlow}>
          {PIPELINE_STEPS.map((s, i) => (
            <div key={s.step} className={styles.pipelineItem}>
              {i > 0 && <FlowArrow />}
              <div className={styles.pipelineCard} style={{ borderTopColor: s.color }}>
                <div className={styles.stepNum} style={{ color: s.color }}>{s.step}</div>
                <div className={styles.stepTitle}>{s.title}</div>
                <p className={styles.stepDesc}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* AI Agents + Data Sources side by side */}
      <div className={styles.twoCol}>
        <Card className={styles.section} delay={0.1}>
          <h2 className={styles.sectionTitle}>AI Agents</h2>
          <div className={styles.agentList}>
            {AI_AGENTS.map(a => (
              <div key={a.name} className={styles.agentItem}>
                <div className={styles.agentName}>{a.name}</div>
                <p className={styles.agentRole}>{a.role}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card className={styles.section} delay={0.15}>
          <h2 className={styles.sectionTitle}>Data Sources</h2>
          <div className={styles.sourceList}>
            {DATA_SOURCES.map(s => (
              <div key={s.name} className={styles.sourceItem}>
                <span className={styles.sourceIcon}>{s.icon}</span>
                <div>
                  <div className={styles.sourceName}>{s.name}</div>
                  <p className={styles.sourceDesc}>{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Risk Gates */}
      <Card className={styles.section} delay={0.2}>
        <h2 className={styles.sectionTitle}>Risk Gates</h2>
        <p className={styles.sectionDesc}>Every trade must pass through these safety checks</p>
        <div className={styles.gateGrid}>
          {RISK_GATES.map(g => (
            <div key={g.name} className={styles.gateCard}>
              <div className={styles.gateHeader}>
                <span className={styles.gateName}>{g.name}</span>
                <Badge
                  variant={g.status === 'critical' ? 'warning' : g.status === 'warning' ? 'neutral' : 'ghost'}
                >
                  {g.status}
                </Badge>
              </div>
              <p className={styles.gateDesc}>{g.desc}</p>
            </div>
          ))}
        </div>
      </Card>

      {/* Auto-Trading Flow */}
      <Card className={styles.section} delay={0.25}>
        <h2 className={styles.sectionTitle}>Auto-Trading Flow</h2>
        <p className={styles.sectionDesc}>From signal to execution — with human in the loop</p>
        <div className={styles.tradeFlow}>
          {AUTO_TRADE_FLOW.map((s, i) => (
            <div key={s.label} className={styles.tradeStep}>
              <div className={styles.tradeStepNum}>{i + 1}</div>
              <div className={styles.tradeStepContent}>
                <div className={styles.tradeStepLabel}>{s.label}</div>
                <p className={styles.tradeStepDesc}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Tech stack */}
      <Card className={styles.section} delay={0.3}>
        <h2 className={styles.sectionTitle}>Technology Stack</h2>
        <div className={styles.techGrid}>
          <div className={styles.techGroup}>
            <div className={styles.techGroupTitle}>Backend</div>
            <div className={styles.techTags}>
              {['FastAPI', 'Python 3.11+', 'SQLite', 'APScheduler', 'CrewAI'].map(t => (
                <Badge key={t} variant="ghost">{t}</Badge>
              ))}
            </div>
          </div>
          <div className={styles.techGroup}>
            <div className={styles.techGroupTitle}>Frontend</div>
            <div className={styles.techTags}>
              {['React 18', 'TypeScript', 'Vite', 'TanStack Query', 'Zustand', 'Framer Motion', 'Chart.js'].map(t => (
                <Badge key={t} variant="ghost">{t}</Badge>
              ))}
            </div>
          </div>
          <div className={styles.techGroup}>
            <div className={styles.techGroupTitle}>AI Providers</div>
            <div className={styles.techTags}>
              {['Perplexity', 'Google Gemini', 'Anthropic Claude', 'Custom LLMs'].map(t => (
                <Badge key={t} variant="ghost">{t}</Badge>
              ))}
            </div>
          </div>
          <div className={styles.techGroup}>
            <div className={styles.techGroupTitle}>Data</div>
            <div className={styles.techTags}>
              {['yfinance', 'SEC EDGAR', 'RSS/Atom', 'NewsAPI', 'Broker APIs'].map(t => (
                <Badge key={t} variant="ghost">{t}</Badge>
              ))}
            </div>
          </div>
        </div>
      </Card>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}

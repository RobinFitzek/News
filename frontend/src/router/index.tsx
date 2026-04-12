import { lazy, Suspense } from 'react'
import { createBrowserRouter } from 'react-router-dom'
import { AuthGuard } from './AuthGuard'
import { RootLayout } from '@/components/layout/RootLayout'
import { Spinner } from '@/components/ui/Spinner'

// ── Eager: auth + shell (tiny, must be synchronous) ───────────────────────────
import { LoginPage }          from '@/pages/LoginPage'
import { TotpPage }           from '@/pages/TotpPage'
import { TwoFactorSetupPage } from '@/pages/TwoFactorSetupPage'
import { NotFoundPage }       from '@/pages/NotFoundPage'

// ── Lazy: all app pages (each becomes its own chunk) ─────────────────────────
const DashboardPage        = lazy(() => import('@/pages/DashboardPage').then(m => ({ default: m.DashboardPage })))
const SettingsPage         = lazy(() => import('@/pages/SettingsPage').then(m => ({ default: m.SettingsPage })))
const WatchlistPage        = lazy(() => import('@/pages/WatchlistPage').then(m => ({ default: m.WatchlistPage })))
const AnalyzePage          = lazy(() => import('@/pages/AnalyzePage').then(m => ({ default: m.AnalyzePage })))
const HistoryPage          = lazy(() => import('@/pages/HistoryPage').then(m => ({ default: m.HistoryPage })))
const LogsPage             = lazy(() => import('@/pages/LogsPage').then(m => ({ default: m.LogsPage })))
const DiscoveriesPage      = lazy(() => import('@/pages/DiscoveriesPage').then(m => ({ default: m.DiscoveriesPage })))
const TopPicksPage         = lazy(() => import('@/pages/TopPicksPage').then(m => ({ default: m.TopPicksPage })))
const InsiderActivityPage  = lazy(() => import('@/pages/InsiderActivityPage').then(m => ({ default: m.InsiderActivityPage })))
const PortfolioPage        = lazy(() => import('@/pages/PortfolioPage').then(m => ({ default: m.PortfolioPage })))
const PaperTradingPage     = lazy(() => import('@/pages/PaperTradingPage').then(m => ({ default: m.PaperTradingPage })))
const TrustPage            = lazy(() => import('@/pages/TrustPage').then(m => ({ default: m.TrustPage })))
const LearningPage         = lazy(() => import('@/pages/LearningPage').then(m => ({ default: m.LearningPage })))
const CrosscheckPage       = lazy(() => import('@/pages/CrosscheckPage').then(m => ({ default: m.CrosscheckPage })))
const GeoHistoryPage       = lazy(() => import('@/pages/GeoHistoryPage').then(m => ({ default: m.GeoHistoryPage })))
const SectorScreenPage     = lazy(() => import('@/pages/SectorScreenPage').then(m => ({ default: m.SectorScreenPage })))
const BacktestPage         = lazy(() => import('@/pages/BacktestPage').then(m => ({ default: m.BacktestPage })))
const JournalPage          = lazy(() => import('@/pages/JournalPage').then(m => ({ default: m.JournalPage })))
const StockDetailPage      = lazy(() => import('@/pages/StockDetailPage').then(m => ({ default: m.StockDetailPage })))
const DiscoverPage         = lazy(() => import('@/pages/DiscoverPage').then(m => ({ default: m.DiscoverPage })))
const CompareStocksPage    = lazy(() => import('@/pages/CompareStocksPage').then(m => ({ default: m.CompareStocksPage })))
const GraveyardPage        = lazy(() => import('@/pages/GraveyardPage').then(m => ({ default: m.GraveyardPage })))
const ArchitecturePage     = lazy(() => import('@/pages/ArchitecturePage').then(m => ({ default: m.ArchitecturePage })))
const MacroPage            = lazy(() => import('@/pages/MacroPage').then(m => ({ default: m.MacroPage })))
const CorporateActionsPage = lazy(() => import('@/pages/CorporateActionsPage').then(m => ({ default: m.CorporateActionsPage })))
const ScenariosPage        = lazy(() => import('@/pages/ScenariosPage').then(m => ({ default: m.ScenariosPage })))
const GrahamPage           = lazy(() => import('@/pages/GrahamPage').then(m => ({ default: m.GrahamPage })))
const FearGreedPage        = lazy(() => import('@/pages/FearGreedPage').then(m => ({ default: m.FearGreedPage })))
const PoliticianTradesPage = lazy(() => import('@/pages/PoliticianTradesPage').then(m => ({ default: m.PoliticianTradesPage })))
const LSTMPage             = lazy(() => import('@/pages/LSTMPage').then(m => ({ default: m.LSTMPage })))

// ── Shared page fallback ──────────────────────────────────────────────────────
function PageFallback() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 'var(--space-16)',
    }}>
      <Spinner size="lg" />
    </div>
  )
}

function P({ C }: { C: React.ComponentType }) {
  return (
    <Suspense fallback={<PageFallback />}>
      <C />
    </Suspense>
  )
}

import type React from 'react'

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <LoginPage />,
  },
  {
    path: '/login/totp',
    element: <TotpPage />,
  },
  {
    element: (
      <AuthGuard>
        <RootLayout />
      </AuthGuard>
    ),
    children: [
      { index: true,                   element: <P C={DashboardPage} /> },
      { path: 'settings',              element: <P C={SettingsPage} /> },
      { path: 'settings/2fa/setup',    element: <TwoFactorSetupPage /> },
      { path: 'watchlist',             element: <P C={WatchlistPage} /> },
      { path: 'analyze',               element: <P C={AnalyzePage} /> },
      { path: 'history',               element: <P C={HistoryPage} /> },
      { path: 'logs',                  element: <P C={LogsPage} /> },
      { path: 'discoveries',           element: <P C={DiscoveriesPage} /> },
      { path: 'top-picks',             element: <P C={TopPicksPage} /> },
      { path: 'insider-activity',      element: <P C={InsiderActivityPage} /> },
      { path: 'portfolio',             element: <P C={PortfolioPage} /> },
      { path: 'paper-trading',         element: <P C={PaperTradingPage} /> },
      { path: 'trust',                 element: <P C={TrustPage} /> },
      { path: 'learning',              element: <P C={LearningPage} /> },
      { path: 'crosscheck',            element: <P C={CrosscheckPage} /> },
      { path: 'geo-history',           element: <P C={GeoHistoryPage} /> },
      { path: 'sector-screen',         element: <P C={SectorScreenPage} /> },
      { path: 'backtest',              element: <P C={BacktestPage} /> },
      { path: 'journal',               element: <P C={JournalPage} /> },
      { path: 'stock/:ticker',         element: <P C={StockDetailPage} /> },
      { path: 'stock/compare',         element: <P C={CompareStocksPage} /> },
      { path: 'discover',              element: <P C={DiscoverPage} /> },
      { path: 'architecture',          element: <P C={ArchitecturePage} /> },
      { path: 'graveyard',             element: <P C={GraveyardPage} /> },
      { path: 'macro',                 element: <P C={MacroPage} /> },
      { path: 'corporate-actions',     element: <P C={CorporateActionsPage} /> },
      { path: 'scenarios',             element: <P C={ScenariosPage} /> },
      { path: 'graham',                element: <P C={GrahamPage} /> },
      { path: 'fear-greed',            element: <P C={FearGreedPage} /> },
      { path: 'politician-trades',     element: <P C={PoliticianTradesPage} /> },
      { path: 'lstm',                  element: <P C={LSTMPage} /> },
      { path: '*',                     element: <NotFoundPage /> },
    ],
  },
])

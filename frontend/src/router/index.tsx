import { createBrowserRouter } from 'react-router-dom'
import { AuthGuard } from './AuthGuard'
import { RootLayout } from '@/components/layout/RootLayout'
import { LoginPage } from '@/pages/LoginPage'
import { TotpPage } from '@/pages/TotpPage'
import { TwoFactorSetupPage } from '@/pages/TwoFactorSetupPage'
import { DashboardPage } from '@/pages/DashboardPage'
import { SettingsPage } from '@/pages/SettingsPage'
import { WatchlistPage } from '@/pages/WatchlistPage'
import { HistoryPage } from '@/pages/HistoryPage'
import { AnalyzePage } from '@/pages/AnalyzePage'
import { LogsPage } from '@/pages/LogsPage'
import { DiscoveriesPage } from '@/pages/DiscoveriesPage'
import { TopPicksPage } from '@/pages/TopPicksPage'
import { InsiderActivityPage } from '@/pages/InsiderActivityPage'
import { PortfolioPage } from '@/pages/PortfolioPage'
import { PaperTradingPage } from '@/pages/PaperTradingPage'
import { TrustPage } from '@/pages/TrustPage'
import { LearningPage } from '@/pages/LearningPage'
import { CrosscheckPage } from '@/pages/CrosscheckPage'
import { GeoHistoryPage } from '@/pages/GeoHistoryPage'
import { SectorScreenPage } from '@/pages/SectorScreenPage'
import { BacktestPage } from '@/pages/BacktestPage'
import { JournalPage } from '@/pages/JournalPage'
import { StockDetailPage } from '@/pages/StockDetailPage'
import { DiscoverPage } from '@/pages/DiscoverPage'
import { CompareStocksPage } from '@/pages/CompareStocksPage'
import { GraveyardPage } from '@/pages/GraveyardPage'
import { ArchitecturePage } from '@/pages/ArchitecturePage'
import { MacroPage } from '@/pages/MacroPage'
import { CorporateActionsPage } from '@/pages/CorporateActionsPage'
import { ScenariosPage } from '@/pages/ScenariosPage'
import { NotFoundPage } from '@/pages/NotFoundPage'

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
      { index: true,                   element: <DashboardPage /> },
      { path: 'settings',              element: <SettingsPage /> },
      { path: 'settings/2fa/setup',    element: <TwoFactorSetupPage /> },
      { path: 'watchlist',             element: <WatchlistPage /> },
      { path: 'analyze',               element: <AnalyzePage /> },
      { path: 'history',               element: <HistoryPage /> },
      { path: 'logs',                  element: <LogsPage /> },
      { path: 'discoveries',           element: <DiscoveriesPage /> },
      { path: 'top-picks',             element: <TopPicksPage /> },
      { path: 'insider-activity',      element: <InsiderActivityPage /> },
      { path: 'portfolio',             element: <PortfolioPage /> },
      { path: 'paper-trading',         element: <PaperTradingPage /> },
      { path: 'trust',                 element: <TrustPage /> },
      { path: 'learning',              element: <LearningPage /> },
      { path: 'crosscheck',            element: <CrosscheckPage /> },
      { path: 'geo-history',           element: <GeoHistoryPage /> },
      { path: 'sector-screen',         element: <SectorScreenPage /> },
      { path: 'backtest',              element: <BacktestPage /> },
      { path: 'journal',               element: <JournalPage /> },
      { path: 'stock/:ticker',         element: <StockDetailPage /> },
      { path: 'stock/compare',         element: <CompareStocksPage /> },
      { path: 'discover',              element: <DiscoverPage /> },
      { path: 'architecture',          element: <ArchitecturePage /> },
      { path: 'graveyard',             element: <GraveyardPage /> },
      { path: 'macro',                 element: <MacroPage /> },
      { path: 'corporate-actions',     element: <CorporateActionsPage /> },
      { path: 'scenarios',             element: <ScenariosPage /> },
      { path: '*',                     element: <NotFoundPage /> },
    ],
  },
])

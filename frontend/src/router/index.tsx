import { createBrowserRouter } from 'react-router-dom'
import { AuthGuard } from './AuthGuard'
import { RootLayout } from '@/components/layout/RootLayout'
import { LoginPage } from '@/pages/LoginPage'
import { DashboardPage } from '@/pages/DashboardPage'
import { SettingsPage } from '@/pages/SettingsPage'
import { WatchlistPage } from '@/pages/WatchlistPage'
import { HistoryPage } from '@/pages/HistoryPage'
import { AnalyzePage } from '@/pages/AnalyzePage'
import { LogsPage } from '@/pages/LogsPage'
import { PlaceholderPage } from '@/pages/PlaceholderPage'
import { NotFoundPage } from '@/pages/NotFoundPage'

const ph = (title: string) => <PlaceholderPage title={title} />

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <LoginPage />,
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
      { path: 'watchlist',             element: <WatchlistPage /> },
      { path: 'analyze',               element: <AnalyzePage /> },
      { path: 'history',               element: <HistoryPage /> },
      { path: 'logs',                  element: <LogsPage /> },
      { path: 'discoveries',           element: ph('Auto-Discovery') },
      { path: 'top-picks',             element: ph('Top Picks') },
      { path: 'insider-activity',      element: ph('Insider Activity') },
      { path: 'portfolio',             element: ph('Portfolio') },
      { path: 'paper-trading',         element: ph('Paper Trading') },
      { path: 'trust',                 element: ph('Trust Score') },
      { path: 'learning',              element: ph('Learning') },
      { path: 'crosscheck',            element: ph('Fact-Check') },
      { path: 'geo-history',           element: ph('Geo History') },
      { path: 'stock/compare',         element: ph('Compare Stocks') },
      { path: 'sector-screen',         element: ph('Sector Screen') },
      { path: 'discover',              element: ph('AI Discover') },
      { path: 'backtest',              element: ph('Backtest') },
      { path: 'journal',               element: ph('Trade Journal') },
      { path: 'architecture',          element: ph('How it Works') },
      { path: 'graveyard',             element: ph('Graveyard') },
      { path: 'stock/:ticker',         element: ph('Stock Detail') },
      { path: '*',                     element: <NotFoundPage /> },
    ],
  },
])

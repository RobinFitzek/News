import { useState, useEffect } from 'react'
import { Card } from '../components/Card'
import { StatusBadge } from '../components/StatusBadge'
import { MetricDisplay } from '../components/MetricDisplay'
import { Table } from '../components/Table'
import * as api from '../services/api'

export function DashboardPage() {
  const [dashboardData, setDashboardData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Fetch dashboard data from backend using API service
    api.getDashboardData()
      .then(data => {
        setDashboardData(data)
        setLoading(false)
      })
      .catch(error => {
        setError(error.message)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-serif mb-4">Loading Dashboard...</h1>
          <div className="w-12 h-12 border-2 border-gray-300 border-t-transparent animate-spin"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-serif mb-4">Error Loading Dashboard</h1>
          <p className="text-gray-500">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-gray-300 text-black hover:bg-gray-500 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  // Mock data for demonstration
  const mockData = {
    systemStatus: 'running',
    portfolioValue: 100000,
    portfolioChange: '+5.2%',
    marketData: {
      SPY: 450.25,
      VIX: 18.75,
      '10Y Yield': 4.25
    },
    recentAnalyses: [
      ['AAPL', 'BUY', '85/100', '2023-05-15 14:30'],
      ['MSFT', 'HOLD', '65/100', '2023-05-15 13:45'],
      ['GOOGL', 'SELL', '45/100', '2023-05-15 12:30']
    ]
  }

  return (
    <div className="min-h-screen bg-black text-white font-sans p-4">
      <header className="mb-6">
        <h1 className="text-xl font-serif">Dashboard</h1>
        <p className="text-sm text-gray-500">Autonomous investment intelligence</p>
      </header>

      {/* System Status Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        <Card title="System Status">
          <div className="flex items-center gap-2">
            <StatusBadge status={mockData.systemStatus} />
            <span className="text-sm">Engine Active</span>
          </div>
        </Card>

        <Card title="Portfolio">
          <div className="text-2xl font-mono mb-2">${mockData.portfolioValue.toLocaleString()}</div>
          <div className={`text-sm ${mockData.portfolioChange.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>
            {mockData.portfolioChange} Today
          </div>
        </Card>

        <Card title="Market Data">
          {Object.entries(mockData.marketData).map(([label, value]) => (
            <MetricDisplay key={label} label={label} value={value} />
          ))}
        </Card>
      </div>

      {/* Recent Analyses */}
      <Card title="Recent Analyses" className="mb-6">
        <Table
          headers={['Ticker', 'Signal', 'Confidence', 'Time']}
          data={mockData.recentAnalyses}
        />
      </Card>

      {/* Status Indicators with Green/Red Colors */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card title="System Health">
          <div className="flex items-center gap-2">
            <StatusBadge status="running" />
            <span className="text-sm">All systems operational</span>
          </div>
        </Card>

        <Card title="Market Status">
          <div className="flex items-center gap-2">
            <StatusBadge status="active" />
            <span className="text-sm">Markets open</span>
          </div>
        </Card>

        <Card title="Data Freshness">
          <div className="flex items-center gap-2">
            <StatusBadge status="success" />
            <span className="text-sm">Real-time data</span>
          </div>
        </Card>
      </div>

      {/* Additional sections would go here */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Watchlist">
          <p className="text-sm text-gray-500">Watchlist content would appear here</p>
        </Card>

        <Card title="Top Picks">
          <p className="text-sm text-gray-500">Top picks would appear here</p>
        </Card>
      </div>
    </div>
  )
}
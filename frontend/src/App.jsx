import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Navigation } from './components/Navigation'
import { DashboardPage } from './pages/DashboardPage'

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-black text-white font-sans">
        <Navigation />
        
        <main className="p-4">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            {/* Additional routes would go here */}
            <Route path="*" element={<DashboardPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}
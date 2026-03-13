import { useState } from 'react'
import { Link } from 'react-router-dom'

export function Navigation() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  
  return (
    <nav className="border-b border-gray-300 p-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link to="/" className="text-xl font-serif text-white no-underline">
            Stockholm
          </Link>
          
          {/* System Status Pill */}
          <div className="flex items-center space-x-2 ml-4">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-xs text-gray-500">Running</span>
          </div>
        </div>

        <button
          className="flex flex-col space-y-1 md:hidden"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label="Menu"
        >
          <span className="w-6 h-0.5 bg-white"></span>
          <span className="w-6 h-0.5 bg-white"></span>
          <span className="w-6 h-0.5 bg-white"></span>
        </button>

        <div className="hidden md:flex items-center space-x-6">
          <Link to="/" className="text-sm text-gray-500 hover:text-white transition-colors">
            Dashboard
          </Link>
          <Link to="/watchlist" className="text-sm text-gray-500 hover:text-white transition-colors">
            Watchlist
          </Link>
          <Link to="/portfolio" className="text-sm text-gray-500 hover:text-white transition-colors">
            Portfolio
          </Link>
          <Link to="/settings" className="text-sm text-gray-500 hover:text-white transition-colors">
            Settings
          </Link>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div className="md:hidden mt-4 space-y-2">
          <Link to="/" className="block text-sm text-gray-500 hover:text-white py-1">
            Dashboard
          </Link>
          <Link to="/watchlist" className="block text-sm text-gray-500 hover:text-white py-1">
            Watchlist
          </Link>
          <Link to="/portfolio" className="block text-sm text-gray-500 hover:text-white py-1">
            Portfolio
          </Link>
          <Link to="/settings" className="block text-sm text-gray-500 hover:text-white py-1">
            Settings
          </Link>
        </div>
      )}
    </nav>
  )
}
#!/bin/bash
# start.sh - Quick start script for Stockholm Dashboard

cd "$(dirname "$0")"

# Check if we should run in development mode (with React dev server)
if [ "$1" = "dev" ]; then
    echo "Starting in DEVELOPMENT mode (separate servers)"
    echo "Building React frontend..."
    cd frontend && npm run build
    cd ..
    
    echo "Starting Python backend..."
    source venv/bin/activate
    python main.py &
    BACKEND_PID=$!
    
    echo "Starting React development server..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    
    echo "Development servers started!"
    echo "   Backend: http://localhost:8000"
    echo "   Frontend: http://localhost:5173"
    echo "   Press Ctrl+C to stop both servers"
    
    # Wait for both processes
    wait $BACKEND_PID $FRONTEND_PID
    
else
    echo "Starting in PRODUCTION mode (single server)"
    echo "Building React frontend..."
    cd frontend && npm run build
    cd ..
    
    echo "Starting Python backend with React frontend..."
    source venv/bin/activate
    python main.py
fi

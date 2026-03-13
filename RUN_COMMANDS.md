# Stockholm Dashboard - Run Commands

## Development Mode (Separate Servers)

### 1. Start Python Backend
```bash
cd /home/robin/Documents/GitHub/News
python main.py
```

The backend will run on `http://localhost:8000` and serve:
- API endpoints at `/api/*`
- Static files at `/static/*`
- Jinja templates (fallback for development)

### 2. Start React Development Server
```bash
cd /home/robin/Documents/GitHub/News/frontend
npm run dev
```

The React dev server will run on `http://localhost:5173` with:
- Hot module replacement
- Automatic proxy to backend API (`/api` → `http://localhost:8000`)
- No CORS issues

### 3. Access the Application
Open your browser to: `http://localhost:5173`

The React app will automatically proxy API requests to the Python backend.

---

## Production Mode (Single Server)

### 1. Build React Frontend
```bash
cd /home/robin/Documents/GitHub/News/frontend
npm run build
```

This creates optimized static files in `frontend/dist/`

### 2. Start Python Backend (with React)
```bash
cd /home/robin/Documents/GitHub/News
python main.py
```

The backend will now:
- Serve React static files from `/frontend/dist/`
- Handle API requests
- Serve `index.html` as fallback for SPA routing

### 3. Access the Application
Open your browser to: `http://localhost:8000`

---

## Key Features

### Development Benefits
- ✅ Hot reloading for React components
- ✅ No CORS issues (proxy configured)
- ✅ Fast iteration cycle
- ✅ Error boundaries and debugging

### Production Benefits
- ✅ Single server deployment
- ✅ Optimized static assets
- ✅ SPA routing support
- ✅ Seamless integration

### API Endpoints
All existing Python API endpoints remain unchanged:
- `/api/status` - System status
- `/api/portfolio/alerts` - Portfolio alerts
- `/api/chart-data/{ticker}` - Chart data
- And all other existing endpoints...

The React frontend communicates with these endpoints exactly as the original frontend did.

---

## Troubleshooting

### If you get CORS errors:
1. Ensure the React dev server is running
2. Check that the proxy configuration is correct in `vite.config.js`
3. Verify the Python backend is running on port 8000

### If React routing doesn't work in production:
1. Ensure `frontend/dist/` exists
2. Check that the backend is serving `index.html` as fallback
3. Verify the static files mount is working

### To clean and rebuild:
```bash
cd /home/robin/Documents/GitHub/News/frontend
rm -rf dist/
npm run build
```
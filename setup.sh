#!/bin/bash
# setup.sh - AI Investment Monitor Setup Script

echo "🚀 Setting up AI Investment Monitor..."
echo ""

# 1. Python Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

source venv/bin/activate

# 2. Dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 3. Directories
echo "📁 Creating directories..."
mkdir -p data logs templates

# 4. Secure .env file
echo "🔒 Securing .env file..."
if [ -f ".env" ]; then
    chmod 600 .env
    echo "   ✅ .env permissions set to 600"
else
    echo "   ⚠️  No .env file found - create one from .env.example"
fi

# 5. Initialize database
echo "🗄️ Initializing database..."
python -c "from database import db; print('   ✅ Database initialized')"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup complete!                                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Start the application:                                      ║"
echo "║    ./start.sh                                                ║"
echo "║                                                              ║"
echo "║  Or manually:                                                ║"
echo "║    source venv/bin/activate                                  ║"
echo "║    python main.py                                            ║"
echo "║                                                              ║"
echo "║  Then open: http://localhost:8080                            ║"
echo "║                                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  For auto-start on boot (systemd):                           ║"
echo "║    sudo cp systemd/investment-monitor.service \\              ║"
echo "║       /etc/systemd/system/                                   ║"
echo "║    sudo systemctl enable investment-monitor                  ║"
echo "║    sudo systemctl start investment-monitor                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

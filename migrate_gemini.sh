#!/bin/bash

# Migration Script: google.generativeai → google.genai
# Entfernt das alte deprecated Package und installiert das neue

echo "🔄 Migration von google.generativeai → google.genai"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Kein Virtual Environment gefunden!"
    echo "   Führe zuerst ./setup.sh aus"
    exit 1
fi

# Activate virtual environment
echo "📦 Aktiviere Virtual Environment..."
source venv/bin/activate

# Check if old package is installed
if pip show google-generativeai &> /dev/null; then
    echo "⚠️  Altes Package gefunden: google-generativeai"
    echo "   Entferne deprecated package..."
    pip uninstall google-generativeai -y
    echo "✅ Altes Package entfernt"
else
    echo "ℹ️  Altes Package nicht installiert (OK)"
fi

# Install new package
echo ""
echo "📥 Installiere neue Packages..."
pip install google-genai>=1.0.0 --upgrade

# Update all dependencies
echo ""
echo "🔄 Aktualisiere Dependencies..."
pip install -r requirements.txt --upgrade

echo ""
echo "✅ Migration abgeschlossen!"
echo ""
echo "Nächste Schritte:"
echo "  1. Starte mit: ./start.sh"
echo "  2. Öffne Dashboard: http://localhost:8080"
echo ""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).parent.parent  # Project root (parent of core/)
DATA_DIR = Path(__file__).parent / "data"  # Keep data in core/data
LOG_DIR = Path(__file__).parent / "logs"   # Keep logs in core/logs
TEMPLATES_DIR = BASE_DIR / "templates"      # Templates in project root

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Load .env if exists
load_dotenv(BASE_DIR / ".env")

# Database
DB_PATH = DATA_DIR / "investment_monitor.db"

# API Keys (loaded from DB after first setup, fallback to env)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Gemini Model Configuration (Adaptive Selection)
# Priority: Higher = prefer when available
# Updated to use Gemini 2.5 and 3.0 models (Feb 2026)
GEMINI_MODELS = {
    # Lightweight screening model (2.5 Flash Lite)
    'flash-8b': {
        'model': 'gemini-2.5-flash-lite',  # Lightweight model for bulk screening
        'rpm': 10,
        'rpd': 250_000,  # Very high daily limit
        'tpm': 1_000_000,
        'priority': 1,
        'use_for': ['scan', 'quick_check'],
        'description': 'Lightweight & fast, perfect for bulk screening'
    },
    # Standard analysis model (2.5 Flash)
    'flash-1.5': {
        'model': 'gemini-2.5-flash',
        'rpm': 5,
        'rpd': 250_000,  # High daily limit
        'tpm': 1_000_000,
        'priority': 2,
        'use_for': ['analyze', 'scan'],
        'description': 'Balanced speed/quality, high limits'
    },
    # Premium Flash (3.0 Flash - latest generation)
    'flash-2.5': {
        'model': 'gemini-3-flash',  # Latest generation
        'rpm': 5,
        'rpd': 250_000,
        'tpm': 1_000_000,
        'priority': 4,  # High priority when available
        'use_for': ['analyze', 'synthesize'],
        'description': 'Latest Gemini 3 Flash quality'
    },
    # Pro model for final synthesis (use 2.5 Flash for now)
    'pro': {
        'model': 'gemini-2.5-flash',  # Using Flash as Pro substitute
        'rpm': 5,
        'rpd': 250_000,
        'tpm': 1_000_000,
        'priority': 5,
        'use_for': ['synthesize', 'final_verdict'],
        'description': 'High quality analysis'
    },
    # Legacy alias for backward compatibility
    'flash': {
        'model': 'gemini-2.5-flash',
        'rpm': 5,
        'rpd': 250_000,
        'tpm': 1_000_000,
        'priority': 4,
        'use_for': ['analyze'],
        'description': 'Alias for 2.5 Flash'
    }
}

# Backward compatibility
GEMINI_LIMITS = GEMINI_MODELS

# Perplexity Limits
PERPLEXITY_DAILY_LIMIT = 33  # ~$5/month budget

# Default Settings (can be overridden in DB)
DEFAULT_SETTINGS = {
    # Scheduler
    "scan_interval_hours": 2,
    "active_hours_start": "08:00",
    "active_hours_end": "22:00",
    "timezone": "Europe/Berlin",
    
    # Notifications
    "email_enabled": False,
    "email_recipient": "",
    "email_smtp_host": "smtp.gmail.com",
    "email_smtp_port": 587,
    "email_smtp_user": "",
    "email_smtp_password": "",
    "notify_on_strong_signals": True,
    "daily_summary_enabled": True,
    "daily_summary_time": "20:00",
    
    # Analysis
    "analysis_depth": "standard",  # quick, standard, deep
    "include_news": True,
    "include_fundamental": True,
    "include_technical": True,
    "analysis_variant": "balanced",  # defensive, balanced, high_growth
}

# Web Server
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")  # Now safe with auth + HTTPS
WEB_PORT = int(os.getenv("WEB_PORT", "8443"))

# HTTPS Configuration
ENABLE_HTTPS = os.getenv("ENABLE_HTTPS", "false").lower() == "true"
CERT_FILE = BASE_DIR / "certs" / "cert.pem"
KEY_FILE = BASE_DIR / "certs" / "key.pem"

# === Extended Investment Algorithm Configuration ===

# Asset Categories with Risk Profiles
ASSET_CATEGORIES = {
    'etf': {
        'risk_range': (1, 3),
        'volatility': 'low',
        'description': 'Exchange Traded Funds - Diversifiziert, stabile Erträge',
        'analysis_focus': ['expense_ratio', 'tracking_error', 'holdings_diversity']
    },
    'blue_chip': {
        'risk_range': (2, 4),
        'volatility': 'low-medium',
        'description': 'Etablierte Großunternehmen - Stabil, solide Dividenden',
        'analysis_focus': ['dividend_yield', 'market_position', 'debt_ratio']
    },
    'growth': {
        'risk_range': (4, 6),
        'volatility': 'medium',
        'description': 'Wachstumsunternehmen - Mittleres Risiko, höheres Potential',
        'analysis_focus': ['revenue_growth', 'market_expansion', 'competitive_moat']
    },
    'startup': {
        'risk_range': (7, 9),
        'volatility': 'high',
        'description': 'Junge Unternehmen - Hohes Risiko, hohes Potential',
        'analysis_focus': ['cash_runway', 'funding_rounds', 'market_disruption']
    },
    'speculative': {
        'risk_range': (8, 10),
        'volatility': 'extreme',
        'description': 'Spekulative Investments - Extrem riskant, möglicher Totalverlust',
        'analysis_focus': ['price_momentum', 'volume_spikes', 'news_sentiment']
    }
}

# Time Horizons for Investment Strategies
TIME_HORIZONS = {
    'short_term': {
        'days': 30,
        'focus': 'technical',
        'scan_frequency': 'daily',
        'description': 'Kurzfristig (1-4 Wochen) - Technische Analyse dominiert'
    },
    'medium_term': {
        'days': 180,
        'focus': 'balanced',
        'scan_frequency': 'weekly',
        'description': 'Mittelfristig (1-6 Monate) - Ausgewogene Analyse'
    },
    'long_term': {
        'days': 365,
        'focus': 'fundamental',
        'scan_frequency': 'monthly',
        'description': 'Langfristig (1+ Jahr) - Fundamentale Analyse dominiert'
    }
}

# Cycle Configuration with Adaptive API Budgets
# Budget is variable and can be overridden in DB settings
# Models are auto-selected based on task type when budget allows
CYCLE_CONFIG = {
    'daily': {
        # Conservative daily budget - prioritize high-limit models
        'api_budget': {
            'flash-8b': 30,    # Bulk screening
            'flash-1.5': 15,   # Standard analysis (high limits)
            'flash-2.5': 5,    # Premium only when needed
            'pro': 3           # Final synthesis only
        },
        'focus': 'quick_screening',
        'model_preference': ['flash-8b', 'flash-1.5'],  # Prefer high-limit models
        'description': 'Täglicher Quick-Scan - nutzt hauptsächlich Flash-8b und Flash-1.5'
    },
    'weekly': {
        # More premium budget for weekly deep dive
        'api_budget': {
            'flash-8b': 30,
            'flash-1.5': 20,
            'flash-2.5': 15,   # More premium for quality
            'pro': 8
        },
        'focus': 'deep_analysis',
        'model_preference': ['flash-1.5', 'flash-2.5'],  # Balance quality/volume
        'description': 'Wöchentliche Tiefenanalyse - mehr Premium-Modelle für Qualität'
    },
    'monthly': {
        # Full premium budget for portfolio review
        'api_budget': {
            'flash-8b': 40,
            'flash-1.5': 30,
            'flash-2.5': 25,   # Heavy premium usage
            'pro': 15          # Multiple pro syntheses
        },
        'focus': 'portfolio_review',
        'model_preference': ['flash-2.5', 'pro'],  # Prioritize quality
        'description': 'Monatliche Portfolio-Review - maximale Qualität'
    }
}

# Default Daily Budget Allocation (can be overridden in DB)
DEFAULT_DAILY_BUDGET = {
    'flash-8b': 50,    # High volume, low quality - screening
    'flash-1.5': 30,   # Balanced - main workhorse
    'flash-2.5': 15,   # Best quality but limited preview
    'flash': 15,       # Alias for flash-2.5
    'pro': 5           # Rare, final synthesis only
}

# Strategy Presets
STRATEGY_PRESETS = {
    'conservative': {
        'risk_tolerance': 'low',
        'time_horizon': 'long_term',
        'asset_mix': {'etf': 50, 'blue_chip': 40, 'growth': 10},
        'max_risk_score': 4
    },
    'balanced': {
        'risk_tolerance': 'medium',
        'time_horizon': 'medium_term',
        'asset_mix': {'etf': 30, 'blue_chip': 40, 'growth': 25, 'startup': 5},
        'max_risk_score': 6
    },
    'aggressive': {
        'risk_tolerance': 'high',
        'time_horizon': 'short_term',
        'asset_mix': {'blue_chip': 20, 'growth': 40, 'startup': 30, 'speculative': 10},
        'max_risk_score': 9
    }
}


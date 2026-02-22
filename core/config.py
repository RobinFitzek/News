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
# Priority: Higher = prefer when available, cost_per_1m = USD per 1M tokens
# Updated for Paid Tier 1 (Feb 2026)
GEMINI_MODELS = {
    'flash-8b': {
        'model': 'gemini-2.5-flash-lite',
        'rpm': 300,
        'rpd': 1_500,
        'tpm': 1_000_000,
        'priority': 1,
        'use_for': ['scan', 'quick_check'],
        'cost_per_1m_input': 0.10,
        'cost_per_1m_output': 0.40,
        'description': 'Cheapest - bulk screening'
    },
    'flash-1.5': {
        'model': 'gemini-2.5-flash',
        'rpm': 300,
        'rpd': 1_500,
        'tpm': 1_000_000,
        'priority': 2,
        'use_for': ['analyze', 'scan'],
        'cost_per_1m_input': 0.15,
        'cost_per_1m_output': 0.60,
        'description': 'Balanced speed/quality'
    },
    'flash-2.5': {
        'model': 'gemini-3-flash',
        'rpm': 300,
        'rpd': 1_500,
        'tpm': 1_000_000,
        'priority': 4,
        'use_for': ['analyze', 'synthesize'],
        'cost_per_1m_input': 0.50,
        'cost_per_1m_output': 3.00,
        'description': 'Highest quality, most expensive'
    },
    'flash': {
        'model': 'gemini-2.5-flash',
        'rpm': 300,
        'rpd': 1_500,
        'tpm': 1_000_000,
        'priority': 2,
        'use_for': ['analyze'],
        'cost_per_1m_input': 0.15,
        'cost_per_1m_output': 0.60,
        'description': 'Alias for 2.5 Flash'
    },
    'pro': {
        'model': 'gemini-3-flash',
        'rpm': 300,
        'rpd': 1_500,
        'tpm': 1_000_000,
        'priority': 4,
        'use_for': ['synthesize', 'final_verdict'],
        'cost_per_1m_input': 0.50,
        'cost_per_1m_output': 3.00,
        'description': 'Alias for Gemini 3 Flash (synthesis)'
    }
}

# Backward compatibility
GEMINI_LIMITS = GEMINI_MODELS

# Perplexity Pricing (USD per 1M tokens + per 1000 searches)
PERPLEXITY_PRICING = {
    'sonar': {
        'cost_per_1m_input': 1.0,
        'cost_per_1m_output': 1.0,
        'cost_per_1000_searches': 5.0,
    }
}

# Default monthly budgets (EUR)
DEFAULT_MONTHLY_BUDGET = {
    'perplexity': 5.0,
    'gemini': 5.0,
}

# EUR to USD conversion estimate (used for budget calculations)
EUR_TO_USD = 1.08

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

    # Monthly API Budgets (EUR)
    "perplexity_monthly_budget": 5.0,
    "gemini_monthly_budget": 5.0,

    # Portfolio Management Rules
    "portfolio_max_position_pct": 10.0,
    "portfolio_stop_loss_pct": 15.0,
    "portfolio_max_sector_pct": 30.0,
    "portfolio_rebalance_drift_pct": 5.0,
    "portfolio_risk_guard_enabled": True,
    "portfolio_global_loss_limit_pct": 10.0,
    "portfolio_risk_cooldown_hours": 24,
    "portfolio_risk_guard_triggered_at": None,

    # Authentication Safety
    "auth_max_failed_attempts": 5,
    "auth_attempt_window_minutes": 15,
    "auth_lockout_minutes": 15,

    # Learning System
    "learning_verification_days": 90,

    # Auto-Discovery
    "discovery_enabled": True,
    "discovery_daily_time": "06:00",
    "discovery_weekly_day": "wed",
    "discovery_weekly_time": "12:00",
    "discovery_promotion_threshold": 55,
    "discovery_max_promote_per_run": 5,
    "discovery_max_watchlist_size": 50,
    "discovery_strategies": ["volume_spike", "breakout", "oversold", "sector_rotation", "insider_buy", "value_screen"],
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

# Quantitative Screener Configuration (replaces AI Stage 1)
QUANT_SCREENER_CONFIG = {
    'composite_weights': {'valuation': 0.30, 'technical': 0.25, 'momentum': 0.25, 'quality': 0.20},
    'anomaly_z_threshold': 2.0,
    'opportunity_threshold': 70,
    'caution_threshold': 30,
    'benchmark_ticker': 'SPY',
}

# Pipeline stage split ratios (how to distribute daily Gemini budget across stages)
# Stage 1 is now free (quant screener), so budget goes to Stage 2 + 3
PIPELINE_STAGE_SPLIT = {
    'stage1': 0.0,   # Free - quant screener, no API cost
    'stage2': 0.60,  # 60% for news summarization (Perplexity)
    'stage3': 0.40,  # 40% for research notes (Gemini flash)
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


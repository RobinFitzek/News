#!/usr/bin/env python3
"""
Test script for Investment Algorithm System
Validates all new modules and their integration.
"""
import sys

def test_config():
    """Test config extensions"""
    print("\n=== Testing Config ===")
    from config import ASSET_CATEGORIES, TIME_HORIZONS, CYCLE_CONFIG, STRATEGY_PRESETS
    
    assert 'etf' in ASSET_CATEGORIES
    assert 'blue_chip' in ASSET_CATEGORIES
    assert 'startup' in ASSET_CATEGORIES
    print(f"✓ Asset Categories: {list(ASSET_CATEGORIES.keys())}")
    
    assert 'short_term' in TIME_HORIZONS
    assert 'long_term' in TIME_HORIZONS
    print(f"✓ Time Horizons: {list(TIME_HORIZONS.keys())}")
    
    assert 'daily' in CYCLE_CONFIG
    assert 'weekly' in CYCLE_CONFIG
    assert 'monthly' in CYCLE_CONFIG
    print(f"✓ Cycle Config: {list(CYCLE_CONFIG.keys())}")
    
    assert 'conservative' in STRATEGY_PRESETS
    assert 'balanced' in STRATEGY_PRESETS
    assert 'aggressive' in STRATEGY_PRESETS
    print(f"✓ Strategy Presets: {list(STRATEGY_PRESETS.keys())}")
    
    return True

def test_database():
    """Test database extensions"""
    print("\n=== Testing Database ===")
    from database import db
    
    # Test strategies
    strategies = db.get_strategies()
    print(f"✓ Strategies in DB: {len(strategies)}")
    
    # Test category methods
    db.set_category("TEST", "growth", 5, "medium", "Test ticker")
    cat = db.get_category("TEST")
    assert cat is not None
    assert cat['category'] == 'growth'
    print("✓ Category CRUD works")
    
    return True

def test_strategy_engine():
    """Test strategy engine"""
    print("\n=== Testing Strategy Engine ===")
    from strategy_engine import strategy_manager, risk_classifier, prompt_builder
    
    # Test strategy manager
    strategy = strategy_manager.get_strategy('balanced')
    assert strategy is not None
    print(f"✓ Strategy Manager: {strategy['name']}")
    
    # Test prompt builder
    prompt = prompt_builder.build_scan_prompt("AAPL", "balanced", "blue_chip")
    assert "AAPL" in prompt
    assert "balanced" in prompt.lower() or "Ausgewogen" in prompt
    print("✓ Prompt Builder works")
    
    return True

def test_discovery_engine():
    """Test discovery engine"""
    print("\n=== Testing Discovery Engine ===")
    from discovery_engine import discovery_engine
    
    # Test suggestions
    suggestions = discovery_engine.get_discovery_suggestions()
    assert 'trending' in suggestions
    assert 'sectors' in suggestions
    print(f"✓ Discovery suggestions: {len(suggestions['categories']['growth'])} growth stocks")
    
    return True

def test_cycle_processor():
    """Test cycle processor (structure only)"""
    print("\n=== Testing Cycle Processor ===")
    from cycle_processor import cycle_processor
    
    assert hasattr(cycle_processor, 'run_daily_cycle')
    assert hasattr(cycle_processor, 'run_weekly_cycle')
    assert hasattr(cycle_processor, 'run_monthly_cycle')
    print("✓ Cycle Processor structure OK")
    
    return True

def test_scheduler_integration():
    """Test scheduler integration"""
    print("\n=== Testing Scheduler ===")
    from scheduler import scheduler
    
    status = scheduler.get_status()
    assert 'is_running' in status
    print(f"✓ Scheduler status: running={status['is_running']}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Investment Algorithm System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Config", test_config),
        ("Database", test_database),
        ("Strategy Engine", test_strategy_engine),
        ("Discovery Engine", test_discovery_engine),
        ("Cycle Processor", test_cycle_processor),
        ("Scheduler", test_scheduler_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

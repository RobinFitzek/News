#!/usr/bin/env python3
"""
Test script for Perplexity-powered stock discovery
Run this to test the new discovery functionality
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clients.perplexity_client import pplx_client
from engine.discovery_engine import discovery_engine
import json


def test_perplexity_discovery():
    """Test the Perplexity discovery feature"""
    print("=" * 70)
    print("🧪 TESTING PERPLEXITY-POWERED STOCK DISCOVERY")
    print("=" * 70)

    # Check if Perplexity is configured
    if not pplx_client.is_configured():
        print("\n❌ Perplexity API not configured!")
        print("   Please set PERPLEXITY_API_KEY in your .env file")
        return False

    usage = pplx_client.get_usage()
    print(f"\n📊 API Status:")
    print(f"   Used today: {usage['used_today']}/{usage['daily_limit']}")
    print(f"   Remaining: {usage['remaining']}")

    if usage['remaining'] <= 0:
        print("\n⚠️ Daily limit reached! Cannot test discovery.")
        return False

    # Test 1: Direct Perplexity call
    print("\n" + "=" * 70)
    print("TEST 1: Direct Perplexity Discovery Call")
    print("=" * 70)

    print("\n🔍 Discovering trending stocks (balanced focus)...")
    result = pplx_client.discover_trending_stocks(
        sector=None,
        focus="balanced",
        limit=5
    )

    if result['success']:
        print(f"\n✅ Discovery successful!")
        print(f"\n📋 Full Analysis:")
        print("-" * 70)
        print(result['raw_analysis'])
        print("-" * 70)

        print(f"\n📊 Parsed Stocks ({len(result['stocks'])}):")
        for stock in result['stocks']:
            score_emoji = "🟢" if stock['score'] >= 70 else "🟡" if stock['score'] >= 50 else "🔴"
            print(f"\n   {score_emoji} {stock['ticker']} - Score: {stock['score']}/100 ({stock['confidence']} confidence)")
            print(f"      Reason: {stock['reason']}")
            print(f"      Catalyst: {stock['catalyst']}")
    else:
        print(f"\n❌ Discovery failed: {result.get('error', 'Unknown error')}")
        return False

    # Test 2: Discovery Engine integration
    print("\n" + "=" * 70)
    print("TEST 2: Discovery Engine Integration")
    print("=" * 70)

    print("\n🔍 Using discovery engine with Perplexity...")
    engine_result = discovery_engine.discover_with_perplexity(
        sector=None,
        focus="growth",
        limit=5
    )

    if engine_result['success']:
        print(f"\n✅ Engine discovery successful!")
        print(f"   Found {engine_result['filtered_count']} new stocks")
        print(f"   Filtered {engine_result.get('watchlist_duplicates', 0)} watchlist duplicates")

        print(f"\n📊 Top Discoveries:")
        for i, stock in enumerate(engine_result['stocks'][:5], 1):
            print(f"   {i}. {stock['ticker']}: {stock['reason'][:60]}...")
    else:
        print(f"\n❌ Engine discovery failed: {engine_result.get('error', 'Unknown error')}")
        return False

    # Test 3: Trending discovery (fallback test)
    print("\n" + "=" * 70)
    print("TEST 3: Trending Discovery (with Perplexity integration)")
    print("=" * 70)

    print("\n🔍 Discovering trending stocks...")
    trending = discovery_engine.discover_trending(limit=5)

    print(f"\n✅ Found {len(trending)} trending stocks:")
    for ticker in trending:
        print(f"   • {ticker}")

    # Test 4: Dashboard suggestions
    print("\n" + "=" * 70)
    print("TEST 4: Dashboard Discovery Suggestions")
    print("=" * 70)

    print("\n🔍 Getting discovery suggestions for dashboard...")
    suggestions = discovery_engine.get_discovery_suggestions()

    print(f"\n📊 Trending: {', '.join(suggestions['trending'])}")

    if 'perplexity_picks' in suggestions:
        print(f"\n🌐 Perplexity Picks:")
        for pick in suggestions['perplexity_picks']:
            print(f"   • {pick['ticker']} (Score: {pick['score']}) - {pick['reason']}")

    # Final API usage
    final_usage = pplx_client.get_usage()
    print("\n" + "=" * 70)
    print(f"📊 Final API Usage: {final_usage['used_today']}/{final_usage['daily_limit']}")
    print(f"   Remaining: {final_usage['remaining']}")
    print("=" * 70)

    print("\n✅ All tests completed successfully!")
    return True


def test_parsing():
    """Test the parsing logic with sample data"""
    print("\n" + "=" * 70)
    print("🧪 TESTING PARSING LOGIC (NO API CALL)")
    print("=" * 70)

    sample_response = """
    Basierend auf den aktuellen Markttrends und Breaking News, hier sind die interessantesten Aktien:

    Die Technologiebranche zeigt starke Momentum mit KI-fokussierten Unternehmen...

    RECOMMENDED_STOCKS:
    - NVDA: [AI-Marktführer mit starkem Wachstum] | Score: 92 | Catalyst: Q4 Earnings am 21. Feb
    - PLTR: [Regierungsverträge + KI-Expansion] | Score: 85 | Catalyst: Army Contract Announcement
    - COIN: [Crypto-Rally profitiert von regulatorischer Klarheit] | Score: 78 | Catalyst: Bitcoin ETF Flows
    - AMD: [Datacenter-Wachstum, MI300 Ramp-up] | Score: 81 | Catalyst: Data Center Revenue Update
    - AVGO: [AI-Networking, VMware Synergien] | Score: 76 | Catalyst: Integration Milestone
    """

    print("\n📄 Sample Response:")
    print("-" * 70)
    print(sample_response)
    print("-" * 70)

    # Test parsing
    parsed = pplx_client._parse_stock_recommendations(sample_response, limit=10)

    print(f"\n✅ Parsed {len(parsed)} stocks:")
    for stock in parsed:
        print(f"\n   {stock['ticker']} - {stock['score']}/100 ({stock['confidence']} confidence)")
        print(f"      {stock['reason']}")
        print(f"      Catalyst: {stock['catalyst']}")

    return len(parsed) > 0


if __name__ == "__main__":
    print("\n🚀 Starting Perplexity Discovery Tests...\n")

    # Run parsing test first (no API call)
    parsing_ok = test_parsing()

    if not parsing_ok:
        print("\n❌ Parsing test failed!")
        sys.exit(1)

    # Ask user if they want to run live tests
    print("\n" + "=" * 70)
    answer = input("\n🌐 Run live tests with Perplexity API? (uses 2-3 API calls) [y/N]: ")

    if answer.lower() in ['y', 'yes']:
        success = test_perplexity_discovery()
        sys.exit(0 if success else 1)
    else:
        print("\n✅ Parsing test passed. Skipping live API tests.")
        print("   Run with 'python test_discovery.py' and answer 'y' to test live API.")
        sys.exit(0)

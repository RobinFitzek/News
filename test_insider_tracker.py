#!/usr/bin/env python3
"""
Test script for Insider Trading Tracker
Tests SEC Edgar client and insider analysis functionality
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clients.sec_edgar_client import sec_client
from engine.insider_tracker import insider_tracker
from core.database import db
import json


def test_sec_edgar_client():
    """Test SEC Edgar client functionality"""
    print("=" * 70)
    print("🧪 TEST 1: SEC EDGAR CLIENT")
    print("=" * 70)

    # Test with a well-known ticker
    test_ticker = "NVDA"

    print(f"\n1. Testing CIK lookup for {test_ticker}...")
    cik = sec_client.get_company_cik(test_ticker)

    if cik:
        print(f"   ✅ Found CIK: {cik}")
    else:
        print(f"   ❌ Could not find CIK for {test_ticker}")
        return False

    print(f"\n2. Fetching insider transactions for {test_ticker} (last 90 days)...")
    print("   Note: This may take 30-60 seconds due to SEC rate limiting...")

    transactions = sec_client.get_insider_transactions(test_ticker, days_back=90)

    if transactions:
        print(f"   ✅ Found {len(transactions)} transactions")

        # Show first transaction as example
        if len(transactions) > 0:
            txn = transactions[0]
            print(f"\n   Example Transaction:")
            print(f"   - Insider: {txn['insider_name']} ({txn['title']})")
            print(f"   - Type: {txn['transaction_type']}")
            print(f"   - Date: {txn['transaction_date']}")
            print(f"   - Shares: {txn['shares']:,.0f} @ ${txn['price']:.2f}")
            print(f"   - Value: ${txn['value']:,.0f}")
            print(f"   - Significance: {txn['significance_score']}/100")
    else:
        print(f"   ⚠️ No recent transactions found (this is normal if no insider activity)")

    print(f"\n3. Getting insider summary for {test_ticker}...")
    summary = sec_client.get_insider_summary(test_ticker, days_back=90)

    if summary['transactions_count'] > 0:
        print(f"   ✅ Summary generated")
        print(f"   - Total Transactions: {summary['transactions_count']}")
        print(f"   - Purchases: {summary['purchases_count']}")
        print(f"   - Sales: {summary['sales_count']}")
        print(f"   - Net Signal: {summary['net_signal']} (Score: {summary['signal_score']})")
        print(f"   - Net Value: ${summary['net_value']:,.0f}")
    else:
        print(f"   ⚠️ No activity to summarize")

    return True


def test_insider_tracker():
    """Test insider tracker engine"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: INSIDER TRACKER ENGINE")
    print("=" * 70)

    test_ticker = "AAPL"

    print(f"\n1. Getting comprehensive analysis for {test_ticker}...")
    analysis = insider_tracker.get_insider_analysis(test_ticker, days_back=180)

    if not analysis['has_activity']:
        print(f"   ⚠️ No activity found for {test_ticker}")
        print(f"   Message: {analysis['message']}")
        print("\n   This is normal - trying another ticker...")

        # Try another common ticker
        test_ticker = "TSLA"
        print(f"\n   Trying {test_ticker}...")
        analysis = insider_tracker.get_insider_analysis(test_ticker, days_back=180)

    if analysis['has_activity']:
        print(f"   ✅ Analysis complete!")
        print(f"\n   Summary:")
        print(f"   - Transactions: {analysis['summary']['transactions_count']}")
        print(f"   - Signal: {analysis['summary']['net_signal']}")
        print(f"   - Score: {analysis['summary']['signal_score']}")

        if analysis['patterns']:
            print(f"\n   Patterns Detected:")
            for pattern, detected in analysis['patterns'].items():
                if detected:
                    print(f"   - {pattern.replace('_', ' ').title()}: ✓")

        if analysis['gemini_analysis']:
            print(f"\n   Gemini Analysis:")
            print(f"   {analysis['gemini_analysis'][:200]}...")

        return True
    else:
        print(f"   ⚠️ No insider activity found")
        print("   Note: Many stocks have no recent insider transactions")
        return True  # Still a success - just no data


def test_database_integration():
    """Test database storage and retrieval"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: DATABASE INTEGRATION")
    print("=" * 70)

    # Create a test transaction
    test_txn = {
        'ticker': 'TEST',
        'insider_name': 'Test Insider',
        'title': 'CEO',
        'transaction_date': '2024-02-01',
        'filing_date': '2024-02-03',
        'transaction_type': 'Purchase',
        'transaction_code': 'P',
        'shares': 10000,
        'price': 50.00,
        'value': 500000,
        'significance_score': 85,
        'form4_url': 'https://sec.gov/test'
    }

    print("\n1. Saving test transaction...")
    db.save_insider_transaction(test_txn)
    print("   ✅ Transaction saved")

    print("\n2. Retrieving transactions for TEST ticker...")
    transactions = db.get_insider_transactions('TEST', days_back=90)

    if transactions:
        print(f"   ✅ Retrieved {len(transactions)} transaction(s)")
        print(f"   - Insider: {transactions[0]['insider_name']}")
        print(f"   - Type: {transactions[0]['transaction_type']}")
        print(f"   - Value: ${transactions[0]['value']:,.0f}")
    else:
        print("   ❌ Failed to retrieve transaction")
        return False

    print("\n3. Getting top insider signals...")
    signals = db.get_top_insider_signals(limit=5)

    if signals:
        print(f"   ✅ Retrieved {len(signals)} signals")
        for signal in signals[:3]:
            print(f"   - {signal['ticker']}: {signal['signal']} ({signal['transaction_count']} txns)")
    else:
        print("   ⚠️ No signals found (normal if database is empty)")

    return True


def test_watchlist_scan():
    """Test scanning entire watchlist"""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: WATCHLIST SCAN")
    print("=" * 70)

    # Check if watchlist has items
    watchlist = db.get_watchlist(active_only=True)

    if not watchlist:
        print("   ⚠️ Watchlist is empty - add some stocks first")
        print("   Skipping watchlist scan test")
        return True

    print(f"\n   Watchlist has {len(watchlist)} stocks")
    print(f"   Note: This test will make multiple SEC API calls (rate limited)")
    print(f"   Expected time: ~{len(watchlist) * 3} seconds")

    answer = input("\n   Proceed with watchlist scan? [y/N]: ")

    if answer.lower() != 'y':
        print("   Skipping watchlist scan")
        return True

    print("\n   Scanning...")
    results = insider_tracker.scan_watchlist_insiders(days_back=90)

    print(f"\n   ✅ Scan complete!")
    print(f"   - Stocks scanned: {len(watchlist)}")
    print(f"   - Stocks with activity: {len(results)}")

    if results:
        print(f"\n   Top 3 signals:")
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result['ticker']}: {result['net_signal']} ({result['transactions_count']} txns)")

    return True


def run_all_tests():
    """Run all tests"""
    print("\n🚀 Starting Insider Trading Tracker Tests...\n")

    tests = [
        ("SEC Edgar Client", test_sec_edgar_client),
        ("Insider Tracker Engine", test_insider_tracker),
        ("Database Integration", test_database_integration),
        ("Watchlist Scan", test_watchlist_scan),
    ]

    results = []

    for name, test_func in tests:
        try:
            print(f"\n{'=' * 70}")
            print(f"Running: {name}")
            print(f"{'=' * 70}")

            success = test_func()
            results.append((name, success))

            if success:
                print(f"\n✅ {name}: PASSED")
            else:
                print(f"\n❌ {name}: FAILED")

        except Exception as e:
            print(f"\n❌ {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔍 INSIDER TRADING TRACKER - TEST SUITE")
    print("=" * 70)
    print("\nThis will test:")
    print("1. SEC Edgar API integration")
    print("2. Insider transaction parsing")
    print("3. Database storage/retrieval")
    print("4. Watchlist scanning (optional)")
    print("\nNote: SEC Edgar has rate limits (10 requests/second)")
    print("Tests may take several minutes to complete.")
    print("=" * 70)

    answer = input("\nProceed with tests? [y/N]: ")

    if answer.lower() in ['y', 'yes']:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        print("\n✅ Tests cancelled")
        sys.exit(0)

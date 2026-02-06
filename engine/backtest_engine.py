"""
Backtest Validation Utility
Tests if the 70+ composite score threshold actually predicts outperformance.
Validates weight combinations to find what actually works vs arbitrary guesses.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from engine.quant_screener import QuantScreener
from core.database import db
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtest quant screener performance against historical data."""

    def __init__(self):
        self.screener = QuantScreener()
        self.benchmark_ticker = 'SPY'

    def backtest_threshold(self, 
                          tickers: List[str],
                          threshold: int = 70,
                          lookback_days: int = 90,
                          test_period_days: int = 730) -> Dict:
        """
        Test if stocks scoring above threshold outperform SPY.
        
        Args:
            tickers: List of tickers to test
            threshold: Score threshold (e.g., 70)
            lookback_days: Days to hold after signal
            test_period_days: Historical period to test (default 2 years)
        """
        logger.info(f"Starting backtest: {len(tickers)} tickers, threshold={threshold}, "
                   f"hold={lookback_days}d, period={test_period_days}d")

        start_date = datetime.now() - timedelta(days=test_period_days)
        results = []

        # Download benchmark data
        spy_data = yf.Ticker(self.benchmark_ticker).history(
            start=start_date, 
            end=datetime.now()
        )
        if spy_data.empty:
            return {'error': 'Could not fetch benchmark data'}

        # Test each ticker at multiple historical dates
        test_dates = pd.date_range(
            start=start_date + timedelta(days=365),  # Need 1yr history for screener
            end=datetime.now() - timedelta(days=lookback_days),
            freq='30D'  # Test monthly
        )

        passed_trades = []
        failed_trades = []

        for ticker in tickers:
            try:
                # Get full history
                stock_hist = yf.Ticker(ticker).history(
                    start=start_date,
                    end=datetime.now()
                )
                
                if stock_hist.empty or len(stock_hist) < 400:
                    continue

                for test_date in test_dates:
                    # Get data up to test date (simulate historical screening)
                    hist_up_to_date = stock_hist[stock_hist.index <= test_date]
                    
                    if len(hist_up_to_date) < 365:
                        continue

                    # Run screener on historical data
                    try:
                        # This is simplified - in reality we'd need to reconstruct
                        # historical info data, which yfinance doesn't provide
                        # For now, use current screening as proxy
                        screen_result = self.screener.screen_ticker(ticker)
                        
                        if not screen_result or 'error' in screen_result:
                            continue

                        score = screen_result['composite_score']
                        
                        # If score passes threshold, simulate trade
                        if score >= threshold:
                            # Get entry price
                            entry_price = float(hist_up_to_date['Close'].iloc[-1])
                            
                            # Get exit price after lookback_days
                            exit_date = test_date + timedelta(days=lookback_days)
                            future_data = stock_hist[stock_hist.index >= exit_date]
                            
                            if future_data.empty:
                                continue
                            
                            exit_price = float(future_data['Close'].iloc[0])
                            
                            # Calculate return
                            stock_return = ((exit_price - entry_price) / entry_price) * 100
                            
                            # Get SPY return over same period
                            spy_entry_data = spy_data[spy_data.index <= test_date]
                            spy_exit_data = spy_data[spy_data.index >= exit_date]
                            
                            if spy_entry_data.empty or spy_exit_data.empty:
                                continue
                            
                            spy_entry = float(spy_entry_data['Close'].iloc[-1])
                            spy_exit = float(spy_exit_data['Close'].iloc[0])
                            spy_return = ((spy_exit - spy_entry) / spy_entry) * 100
                            
                            alpha = stock_return - spy_return
                            
                            trade = {
                                'ticker': ticker,
                                'date': test_date.strftime('%Y-%m-%d'),
                                'score': score,
                                'entry_price': round(entry_price, 2),
                                'exit_price': round(exit_price, 2),
                                'stock_return': round(stock_return, 2),
                                'spy_return': round(spy_return, 2),
                                'alpha': round(alpha, 2),
                                'outperformed': stock_return > spy_return,
                            }
                            
                            results.append(trade)
                            
                            if trade['outperformed']:
                                passed_trades.append(trade)
                            else:
                                failed_trades.append(trade)

                    except Exception as e:
                        logger.debug(f"Error screening {ticker} at {test_date}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Error backtesting {ticker}: {e}")
                continue

        # Calculate statistics
        if not results:
            return {'error': 'No valid backtest results'}

        total_trades = len(results)
        win_rate = (len(passed_trades) / total_trades) * 100 if total_trades > 0 else 0
        avg_return = np.mean([t['stock_return'] for t in results])
        avg_alpha = np.mean([t['alpha'] for t in results])
        
        summary = {
            'threshold': threshold,
            'lookback_days': lookback_days,
            'test_period_days': test_period_days,
            'total_trades': total_trades,
            'winning_trades': len(passed_trades),
            'losing_trades': len(failed_trades),
            'win_rate_pct': round(win_rate, 1),
            'avg_return_pct': round(avg_return, 2),
            'avg_alpha_pct': round(avg_alpha, 2),
            'avg_spy_return': round(np.mean([t['spy_return'] for t in results]), 2),
            'best_trade': max(results, key=lambda x: x['stock_return']) if results else None,
            'worst_trade': min(results, key=lambda x: x['stock_return']) if results else None,
            'tested_tickers': len(set(t['ticker'] for t in results)),
        }

        # Store results
        self._store_backtest_results(summary, results)

        return {
            'summary': summary,
            'trades': results[:50],  # Return sample of trades
        }

    def _store_backtest_results(self, summary: Dict, trades: List[Dict]):
        """Store backtest results in database."""
        try:
            # Store summary
            db.execute("""
                INSERT INTO backtest_results (
                    threshold, lookback_days, test_period_days,
                    total_trades, win_rate_pct, avg_return_pct, avg_alpha_pct,
                    tested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary['threshold'],
                summary['lookback_days'],
                summary['test_period_days'],
                summary['total_trades'],
                summary['win_rate_pct'],
                summary['avg_return_pct'],
                summary['avg_alpha_pct'],
                datetime.now().isoformat(),
            ))
            
            logger.info(f"Stored backtest results: {summary['total_trades']} trades, "
                       f"{summary['win_rate_pct']}% win rate")

        except Exception as e:
            logger.warning(f"Could not store backtest results: {e}")

    def test_weight_combinations(self, tickers: List[str]) -> List[Dict]:
        """
        Test different weight combinations to find optimal allocation.
        Current: val=30%, tech=25%, mom=25%, qual=20%
        """
        weight_variants = [
            {'name': 'current', 'valuation': 0.30, 'technical': 0.25, 'momentum': 0.25, 'quality': 0.20},
            {'name': 'value_heavy', 'valuation': 0.40, 'technical': 0.20, 'momentum': 0.20, 'quality': 0.20},
            {'name': 'momentum_heavy', 'valuation': 0.20, 'technical': 0.20, 'momentum': 0.40, 'quality': 0.20},
            {'name': 'quality_focus', 'valuation': 0.25, 'technical': 0.20, 'momentum': 0.20, 'quality': 0.35},
            {'name': 'technical_only', 'valuation': 0.15, 'technical': 0.45, 'momentum': 0.25, 'quality': 0.15},
            {'name': 'equal_weight', 'valuation': 0.25, 'technical': 0.25, 'momentum': 0.25, 'quality': 0.25},
        ]

        results = []
        
        for variant in weight_variants:
            logger.info(f"Testing weight variant: {variant['name']}")
            
            # Temporarily override screener weights
            original_weights = self.screener.config['composite_weights'].copy()
            self.screener.config['composite_weights'] = {
                'valuation': variant['valuation'],
                'technical': variant['technical'],
                'momentum': variant['momentum'],
                'quality': variant['quality'],
            }

            # Run backtest with these weights
            backtest_result = self.backtest_threshold(
                tickers=tickers[:20],  # Sample for speed
                threshold=70,
                lookback_days=90,
                test_period_days=365
            )

            # Restore original weights
            self.screener.config['composite_weights'] = original_weights

            if 'summary' in backtest_result:
                result = {
                    'variant_name': variant['name'],
                    'weights': variant,
                    'performance': backtest_result['summary'],
                }
                results.append(result)

        # Sort by alpha
        results.sort(key=lambda x: x['performance']['avg_alpha_pct'], reverse=True)

        return results

    def validate_current_settings(self) -> Dict:
        """Quick validation: test current settings on recent picks."""
        try:
            # Get recent analyses with strong signals
            recent = db.query("""
                SELECT ticker, composite_score, timestamp
                FROM analyses
                WHERE signal IN ('Opportunity', 'STRONG_BUY', 'BUY')
                AND composite_score >= 70
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            """, ((datetime.now() - timedelta(days=180)).isoformat(),))

            if not recent:
                return {'message': 'No recent picks to validate'}

            tickers = [r['ticker'] for r in recent]
            
            result = self.backtest_threshold(
                tickers=tickers,
                threshold=70,
                lookback_days=90,
                test_period_days=180
            )

            return result

        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return {'error': str(e)}


# Singleton
backtest_engine = BacktestEngine()

"""
AI-Generated Weekly Investment Letter
Summarizes the past 7 days: top signals, geo events, accuracy stats.
Sent every Sunday evening via existing email infrastructure.
"""
from datetime import datetime, timedelta
from typing import Dict, List
from core.database import db
from core.notifications import notifications
import logging

logger = logging.getLogger(__name__)


class WeeklyLetterGenerator:

    def collect_week_data(self) -> Dict:
        """Collect all data for the past 7 days."""
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()

        analyses = db.query("""
            SELECT ticker, signal, confidence, timestamp, risk_score, geo_risk_score
            FROM analysis_history
            WHERE timestamp >= ?
            ORDER BY confidence DESC
        """, (cutoff,))

        geo_scans = db.query("""
            SELECT timestamp, severity_avg, is_delta, raw_summary
            FROM geopolitical_events
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 5
        """, (cutoff,))

        try:
            from engine.learning_optimizer import learning_optimizer
            learning_stats = learning_optimizer.get_learning_stats()
        except Exception:
            learning_stats = {}

        # Auto-trade weekly digest
        auto_trade_digest = {}
        try:
            from engine.auto_paper_trader import auto_paper_trader
            perf = auto_paper_trader.get_performance_summary()
            opened_rows = db.query(
                "SELECT COUNT(*) as c FROM auto_paper_trades WHERE entry_date >= ? AND status != 'blocked'",
                (cutoff,)
            )
            closed_rows = db.query(
                "SELECT COUNT(*) as c, AVG(pnl_pct) as avg_pnl FROM auto_paper_trades WHERE exit_date >= ? AND status = 'closed'",
                (cutoff,)
            )
            opened = opened_rows[0]['c'] if opened_rows else 0
            closed_row = closed_rows[0] if closed_rows else {}
            closed = closed_row.get('c') or 0
            avg_pnl = (closed_row.get('avg_pnl') or 0) * 100
            auto_trade_digest = {
                'opened': opened,
                'closed': closed,
                'avg_pnl_pct': round(avg_pnl, 2),
                'total_closed': perf['total_closed'],
                'win_rate_pct': perf['win_rate_pct'],
                'total_pnl_pct': perf['total_pnl_pct'],
                'open_positions': perf['open_positions'],
            }
        except Exception:
            pass

        strong_buys = [a for a in analyses if 'BUY' in (a.get('signal') or '')]
        strong_sells = [a for a in analyses if 'SELL' in (a.get('signal') or '')]

        return {
            'analyses': analyses,
            'geo_scans': geo_scans,
            'learning_stats': learning_stats,
            'auto_trade_digest': auto_trade_digest,
            'strong_buys': strong_buys[:5],
            'strong_sells': strong_sells[:5],
            'week_start': cutoff[:10],
            'week_end': datetime.now().strftime('%Y-%m-%d'),
            'total_analyses': len(analyses),
        }

    def build_prompt(self, data: Dict) -> str:
        buys = ', '.join(a['ticker'] for a in data['strong_buys']) or 'none'
        sells = ', '.join(a['ticker'] for a in data['strong_sells']) or 'none'
        geo_summary = data['geo_scans'][0].get('raw_summary', '')[:500] if data['geo_scans'] else 'No geo scans this week.'
        accuracy = data['learning_stats'].get('avg_accuracy', 0)
        total_verified = data['learning_stats'].get('total_verified', 0)

        at = data.get('auto_trade_digest', {})
        if at:
            sign = '+' if at.get('avg_pnl_pct', 0) >= 0 else ''
            auto_line = (
                f"- Auto-trader this week: {at.get('opened', 0)} opened, "
                f"{at.get('closed', 0)} closed, avg {sign}{at.get('avg_pnl_pct', 0):.1f}% "
                f"(all-time: {at.get('total_closed', 0)} closed, {at.get('win_rate_pct', 0):.0f}% win rate, "
                f"{'+' if at.get('total_pnl_pct', 0) >= 0 else ''}{at.get('total_pnl_pct', 0):.1f}% total)"
            )
        else:
            auto_line = "- Auto-trader: no data available"

        return f"""You are an AI investment analyst. Write a concise weekly investment letter for the week of {data['week_start']} to {data['week_end']}.

DATA:
- Total analyses run: {data['total_analyses']}
- Strong buy signals: {buys}
- Strong sell signals: {sells}
- System accuracy: {accuracy:.0%} over {total_verified} verified predictions
- Latest geo risk summary: {geo_summary}
{auto_line}

Write a professional, concise 4-paragraph HTML letter:
1. Market overview and key signals this week
2. Geopolitical risk highlights and portfolio implications
3. System accuracy and outlook for next week
4. Auto-trader performance digest (one sentence summary)

Format as clean HTML paragraphs. Be direct and specific. No generic filler."""

    def generate_and_send(self):
        """Generate via Gemini and send via existing email infrastructure."""
        try:
            data = self.collect_week_data()
            if data['total_analyses'] == 0:
                logger.info("Weekly letter skipped: no analyses this week")
                return

            from clients.gemini_client import gemini_client
            prompt = self.build_prompt(data)
            letter_html = gemini_client.generate(prompt, max_tokens=800)

            if not letter_html:
                logger.warning("Weekly letter: Gemini returned empty response")
                return

            subject = f"Stockholm Weekly Letter — {data['week_end']}"
            # Send via existing email method
            notifications.reload_settings()
            notifications._send_email(subject, letter_html)
            logger.info(f"Weekly letter sent for week ending {data['week_end']}")

        except Exception as e:
            logger.error(f"Weekly letter generation failed: {e}")


weekly_letter_generator = WeeklyLetterGenerator()

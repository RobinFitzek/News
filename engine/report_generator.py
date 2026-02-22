"""
Weekly Report Generator
Compiles portfolio summary, new discoveries, upcoming earnings, sector changes,
and risk metrics into a formatted HTML email report. Sends via the notification
system (email + webhooks).
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
from core.notifications import notifications
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate and send periodic investment reports."""

    def __init__(self):
        self._last_report_time = None

    def generate_weekly_report(self) -> Dict:
        """
        Build the full weekly report.
        Returns dict with html_content and metadata. Optionally sends it.
        """
        logger.info("Generating weekly report...")

        try:
            portfolio_section = self._build_portfolio_section()
            discoveries_section = self._build_discoveries_section()
            earnings_section = self._build_earnings_section()
            sector_section = self._build_sector_section()
            risk_section = self._build_risk_section()

            report_date = datetime.now().strftime('%d.%m.%Y')

            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto; background: #f9fafb; }}
                    .header {{ background: linear-gradient(135deg, #1e3a5f, #2563eb); color: white; padding: 30px; text-align: center; }}
                    .header h1 {{ margin: 0; font-size: 24px; }}
                    .header p {{ margin: 8px 0 0; opacity: 0.85; }}
                    .section {{ padding: 20px; background: white; margin: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                    .section h2 {{ margin-top: 0; color: #1e3a5f; font-size: 18px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                    th {{ text-align: left; padding: 8px 10px; background: #f3f4f6; font-size: 12px; text-transform: uppercase; color: #6b7280; }}
                    td {{ padding: 8px 10px; border-bottom: 1px solid #f3f4f6; font-size: 14px; }}
                    .positive {{ color: #10b981; font-weight: bold; }}
                    .negative {{ color: #ef4444; font-weight: bold; }}
                    .neutral {{ color: #6b7280; }}
                    .metric-row {{ display: flex; justify-content: space-between; padding: 6px 0; }}
                    .footer {{ text-align: center; padding: 20px; color: #9ca3af; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Weekly Investment Report</h1>
                    <p>Week ending {report_date}</p>
                </div>
                {portfolio_section}
                {discoveries_section}
                {earnings_section}
                {sector_section}
                {risk_section}
                <div class="footer">
                    <p>AI Investment Monitor | Generated {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
                </div>
            </body>
            </html>
            """

            # Auto-send if configured
            auto_send = db.get_setting('weekly_report_auto_send')
            sent = False
            if auto_send:
                sent = self.send_report(html)

            self._last_report_time = datetime.now()

            logger.info(f"Weekly report generated {'and sent' if sent else '(not sent)'}.")

            return {
                'html_content': html,
                'sent': sent,
                'generated_at': datetime.now().isoformat(),
                'sections': ['portfolio', 'discoveries', 'earnings', 'sectors', 'risk'],
            }

        except Exception as e:
            logger.error(f"Weekly report generation failed: {e}")
            return {'html_content': '', 'sent': False, 'error': str(e)}

    def _build_portfolio_section(self) -> str:
        """Portfolio summary: holdings, total value, top gainers/losers."""
        try:
            holdings = db.get_portfolio_holdings()
            active = [h for h in holdings if h.get('shares', 0) > 0]

            if not active:
                return """
                <div class="section">
                    <h2>Portfolio Summary</h2>
                    <p class="neutral">No active holdings.</p>
                </div>
                """

            total_value = 0
            total_cost = 0
            rows = ""

            for h in active:
                ticker = h.get('ticker', '')
                shares = h.get('shares', 0)
                avg_cost = h.get('avg_cost', 0)
                current_price = h.get('current_price', avg_cost)

                position_value = shares * current_price
                cost_basis = shares * avg_cost
                total_value += position_value
                total_cost += cost_basis

                gain_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
                gain_class = 'positive' if gain_pct >= 0 else 'negative'

                rows += f"""
                <tr>
                    <td><strong>{ticker}</strong></td>
                    <td>{shares}</td>
                    <td>${current_price:.2f}</td>
                    <td>${position_value:,.0f}</td>
                    <td class="{gain_class}">{gain_pct:+.1f}%</td>
                </tr>
                """

            total_gain_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
            total_class = 'positive' if total_gain_pct >= 0 else 'negative'

            # Cash balance
            cash_rows = db.query("SELECT SUM(amount) as total FROM cash_positions WHERE amount > 0")
            cash = cash_rows[0]['total'] if cash_rows and cash_rows[0]['total'] else 0

            return f"""
            <div class="section">
                <h2>Portfolio Summary</h2>
                <p>Total Value: <strong>${total_value:,.0f}</strong> + ${cash:,.0f} cash
                | Overall: <span class="{total_class}">{total_gain_pct:+.1f}%</span></p>
                <table>
                    <thead>
                        <tr><th>Ticker</th><th>Shares</th><th>Price</th><th>Value</th><th>Gain</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        except Exception as e:
            logger.warning(f"Portfolio section error: {e}")
            return '<div class="section"><h2>Portfolio Summary</h2><p class="neutral">Could not load portfolio data.</p></div>'

    def _build_discoveries_section(self) -> str:
        """New discoveries from the past 7 days."""
        try:
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            discoveries = db.query(
                "SELECT ticker, strategy, score, discovered_at FROM discoveries "
                "WHERE discovered_at >= ? ORDER BY score DESC LIMIT 10",
                (week_ago,)
            )

            if not discoveries:
                return """
                <div class="section">
                    <h2>New Discoveries (7d)</h2>
                    <p class="neutral">No new discoveries this week.</p>
                </div>
                """

            rows = ""
            for d in discoveries:
                score = d.get('score', 0)
                score_class = 'positive' if score >= 70 else 'neutral'
                rows += f"""
                <tr>
                    <td><strong>{d.get('ticker', '')}</strong></td>
                    <td>{d.get('strategy', 'N/A')}</td>
                    <td class="{score_class}">{score}</td>
                    <td>{d.get('discovered_at', '')[:10]}</td>
                </tr>
                """

            return f"""
            <div class="section">
                <h2>New Discoveries (7d)</h2>
                <p>{len(discoveries)} stocks discovered this week.</p>
                <table>
                    <thead>
                        <tr><th>Ticker</th><th>Strategy</th><th>Score</th><th>Date</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        except Exception as e:
            logger.warning(f"Discoveries section error: {e}")
            return '<div class="section"><h2>New Discoveries</h2><p class="neutral">Could not load discoveries.</p></div>'

    def _build_earnings_section(self) -> str:
        """Upcoming earnings for watchlist and portfolio tickers."""
        try:
            # Gather tickers from portfolio and watchlist
            holdings = db.get_portfolio_holdings()
            portfolio_tickers = [h['ticker'] for h in holdings if h.get('shares', 0) > 0]

            watchlist = db.query("SELECT ticker FROM watchlist WHERE active = 1")
            watchlist_tickers = [w['ticker'] for w in watchlist] if watchlist else []

            all_tickers = list(set(portfolio_tickers + watchlist_tickers))

            if not all_tickers:
                return """
                <div class="section">
                    <h2>Upcoming Earnings</h2>
                    <p class="neutral">No tickers to track.</p>
                </div>
                """

            # Try to use earnings tracker
            upcoming = []
            try:
                from engine.earnings_tracker import earnings_tracker
                for ticker in all_tickers[:30]:  # Limit API calls
                    info = earnings_tracker.get_earnings_info(ticker)
                    if info and info.get('days_until') is not None and 0 <= info['days_until'] <= 14:
                        upcoming.append({
                            'ticker': ticker,
                            'days_until': info['days_until'],
                            'date': info.get('earnings_date', 'N/A'),
                            'in_portfolio': ticker in portfolio_tickers,
                        })
            except ImportError:
                logger.debug("Earnings tracker not available for report.")

            if not upcoming:
                return """
                <div class="section">
                    <h2>Upcoming Earnings (14d)</h2>
                    <p class="neutral">No upcoming earnings in the next 2 weeks.</p>
                </div>
                """

            upcoming.sort(key=lambda x: x['days_until'])

            rows = ""
            for e in upcoming:
                portfolio_badge = ' [HELD]' if e['in_portfolio'] else ''
                rows += f"""
                <tr>
                    <td><strong>{e['ticker']}</strong>{portfolio_badge}</td>
                    <td>{e.get('date', 'N/A')}</td>
                    <td>{e['days_until']} days</td>
                </tr>
                """

            return f"""
            <div class="section">
                <h2>Upcoming Earnings (14d)</h2>
                <p>{len(upcoming)} earnings report(s) in the next 2 weeks.</p>
                <table>
                    <thead>
                        <tr><th>Ticker</th><th>Date</th><th>Days Away</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        except Exception as e:
            logger.warning(f"Earnings section error: {e}")
            return '<div class="section"><h2>Upcoming Earnings</h2><p class="neutral">Could not load earnings data.</p></div>'

    def _build_sector_section(self) -> str:
        """Sector momentum and rotation signals."""
        try:
            from engine.sector_momentum import sector_momentum

            rankings = sector_momentum.get_sector_rankings()
            rotation = sector_momentum.get_rotation_signals()

            if not rankings:
                return """
                <div class="section">
                    <h2>Sector Overview</h2>
                    <p class="neutral">No sector data available.</p>
                </div>
                """

            rows = ""
            for r in rankings[:6]:  # Top and bottom sectors
                return_class = 'positive' if r['return_1mo'] >= 0 else 'negative'
                rows += f"""
                <tr>
                    <td><strong>{r['name']}</strong> ({r['etf']})</td>
                    <td class="{return_class}">{r['return_1mo']:+.1f}%</td>
                    <td>{r['return_1wk']:+.1f}%</td>
                    <td>{r['relative_strength']:+.1f}</td>
                </tr>
                """

            rotation_text = ""
            if rotation.get('gaining'):
                gaining_names = ', '.join(r['name'] for r in rotation['gaining'])
                rotation_text += f"<p class='positive'>Money flowing into: {gaining_names}</p>"
            if rotation.get('losing'):
                losing_names = ', '.join(r['name'] for r in rotation['losing'])
                rotation_text += f"<p class='negative'>Money flowing out of: {losing_names}</p>"

            return f"""
            <div class="section">
                <h2>Sector Overview</h2>
                {rotation_text}
                <table>
                    <thead>
                        <tr><th>Sector</th><th>1M Return</th><th>1W Return</th><th>Rel. Strength</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        except Exception as e:
            logger.warning(f"Sector section error: {e}")
            return '<div class="section"><h2>Sector Overview</h2><p class="neutral">Could not load sector data.</p></div>'

    def _build_risk_section(self) -> str:
        """Portfolio risk metrics summary."""
        try:
            # Concentration check
            holdings = db.get_portfolio_holdings()
            active = [h for h in holdings if h.get('shares', 0) > 0]

            if not active:
                return ""

            total_value = sum(h.get('shares', 0) * h.get('current_price', h.get('avg_cost', 0)) for h in active)

            alerts = []

            if total_value > 0:
                for h in active:
                    value = h.get('shares', 0) * h.get('current_price', h.get('avg_cost', 0))
                    pct = (value / total_value) * 100
                    if pct > 15:
                        alerts.append(f"{h.get('ticker', '?')} is {pct:.0f}% of portfolio (concentration risk)")

            # Drawdown check
            try:
                from engine.drawdown_tracker import drawdown_tracker
                for h in active[:10]:
                    ticker = h.get('ticker', '')
                    avg_cost = h.get('avg_cost', 0)
                    current = h.get('current_price', avg_cost)
                    if avg_cost > 0:
                        loss_pct = ((current - avg_cost) / avg_cost) * 100
                        if loss_pct < -15:
                            alerts.append(f"{ticker} down {loss_pct:.1f}% from cost basis")
            except ImportError:
                pass

            if not alerts:
                return """
                <div class="section">
                    <h2>Risk Alerts</h2>
                    <p class="positive">No risk alerts this week.</p>
                </div>
                """

            alert_items = ''.join(f'<li style="padding: 4px 0;">{a}</li>' for a in alerts)

            return f"""
            <div class="section">
                <h2>Risk Alerts</h2>
                <ul style="margin: 0; padding-left: 20px;">{alert_items}</ul>
            </div>
            """

        except Exception as e:
            logger.warning(f"Risk section error: {e}")
            return ""

    def send_report(self, html_content: str) -> bool:
        """Send the report HTML via the notification system (email)."""
        if not html_content:
            logger.warning("No report content to send.")
            return False

        try:
            notifications.reload_settings()

            if not notifications.enabled or not notifications.recipient:
                logger.info("Email not configured — report not sent.")
                return False

            subject = f"Weekly Investment Report - {datetime.now().strftime('%d.%m.%Y')}"
            sent = notifications._send_email(subject, html_content)

            if sent:
                logger.info("Weekly report sent successfully.")
                db.log_alert("", "WEEKLY_REPORT", subject, notifications.recipient)
            else:
                logger.warning("Weekly report email failed to send.")

            return sent

        except Exception as e:
            logger.error(f"Failed to send weekly report: {e}")
            return False


# Singleton
report_generator = ReportGenerator()

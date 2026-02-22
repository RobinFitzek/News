"""
Notification System — Email + Telegram + Discord
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        self._load_settings()
    
    def _load_settings(self):
        """Load email settings from database"""
        self.enabled = db.get_setting("email_enabled")
        self.recipient = db.get_setting("email_recipient")
        self.smtp_host = db.get_setting("email_smtp_host")
        self.smtp_port = db.get_setting("email_smtp_port")
        self.smtp_user = db.get_setting("email_smtp_user")
        self.smtp_password = db.get_setting("email_smtp_password")
        self.notify_strong_signals = db.get_setting("notify_on_strong_signals")
    
    def reload_settings(self):
        """Reload settings from DB (called after settings change)"""
        self._load_settings()
    
    def should_notify(self, signal: str) -> bool:
        """Check if we should send notification for this signal"""
        if not self.enabled or not self.recipient:
            return False
        
        if self.notify_strong_signals:
            return signal in ["STRONG_BUY", "STRONG_SELL"]
        return signal in ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL"]
    
    def send_alert(self, ticker: str, signal: str, recommendation: str,
                   confidence: int = 0) -> bool:
        """Send alert to email + webhooks for a stock signal."""
        sent = False

        # Email channel
        if self.should_notify(signal):
            subject = f"\U0001f6a8 {signal}: {ticker}"
            signal_color = '#10b981' if 'BUY' in signal else '#ef4444'
            html = f"""
            <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: {signal_color}; color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0;">{signal}</h1>
                    <h2 style="margin: 10px 0 0 0;">{ticker}</h2>
                </div>
                <div style="padding: 20px; background: #f9fafb;">
                    <h3>\U0001f4ca Analysis</h3>
                    <div style="background: white; padding: 15px; border-radius: 8px;">{recommendation}</div>
                    <p style="color: #6b7280; font-size: 12px; margin-top: 20px;">
                        {datetime.now().strftime('%d.%m.%Y %H:%M')} | AI Investment Monitor
                    </p>
                </div>
            </body></html>
            """
            sent |= self._send_email(subject, html)

        # Webhook channels (Telegram / Discord)
        try:
            from engine.webhook_notifier import webhook_notifier
            webhook_notifier.reload()
            if webhook_notifier.any_configured():
                sent |= webhook_notifier.send_signal_alert(
                    ticker=ticker,
                    signal=signal,
                    confidence=confidence,
                    recommendation=recommendation[:300],
                )
        except Exception as e:
            logger.debug(f"Webhook send skipped: {e}")

        return sent
    
    def send_daily_summary(self, analyses: list) -> bool:
        """Send daily summary email"""
        if not self.enabled or not self.recipient:
            return False
        
        subject = f"📊 Daily Investment Summary - {datetime.now().strftime('%d.%m.%Y')}"
        
        # Build summary table
        rows = ""
        for a in analyses:
            signal_color = "#10b981" if "BUY" in a.get('signal', '') else (
                "#ef4444" if "SELL" in a.get('signal', '') else "#6b7280"
            )
            rows += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; font-weight: bold;">
                    {a.get('ticker', 'N/A')}
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; 
                           color: {signal_color}; font-weight: bold;">
                    {a.get('signal', 'N/A')}
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">
                    {a.get('confidence', 0)}%
                </td>
            </tr>
            """
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #1f2937; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">📊 Daily Summary</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.8;">
                    {datetime.now().strftime('%d.%m.%Y')}
                </p>
            </div>
            <div style="padding: 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #f3f4f6;">
                            <th style="padding: 10px; text-align: left;">Ticker</th>
                            <th style="padding: 10px; text-align: left;">Signal</th>
                            <th style="padding: 10px; text-align: left;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows if rows else '<tr><td colspan="3" style="padding: 20px; text-align: center;">Keine Analysen heute</td></tr>'}
                    </tbody>
                </table>
            </div>
            <div style="padding: 20px; background: #f9fafb; text-align: center;">
                <p style="color: #6b7280; font-size: 12px;">
                    AI Investment Monitor | Dein N100 Homeserver
                </p>
            </div>
        </body>
        </html>
        """
        
        return self._send_email(subject, html)
    
    def _send_email(self, subject: str, html_content: str) -> bool:
        """Send email via SMTP"""
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.recipient]):
            print("⚠️ Email not configured properly")
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = self.recipient
            
            msg.attach(MIMEText(html_content, "html"))
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, self.recipient, msg.as_string())
            
            # Log the alert
            db.log_alert("", "", subject, self.recipient)
            print(f"✅ Email sent: {subject}")
            return True
            
        except Exception as e:
            print(f"❌ Email error: {e}")
            return False

# Singleton
notifications = NotificationService()

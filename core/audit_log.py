"""Security audit logging"""
from datetime import datetime
from pathlib import Path
import json

class AuditLogger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(exist_ok=True)

    def log(self, event_type: str, username: str = None,
            ip: str = None, details: dict = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "username": username,
            "ip": ip,
            "details": details or {}
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

audit_log = AuditLogger(Path(__file__).parent.parent / "logs" / "security.log")

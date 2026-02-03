# Security Documentation

## Authentication
- Session-based authentication with 24-hour timeout
- Bcrypt password hashing with automatic salt
- Default credentials: admin / changeme123
- **CHANGE DEFAULT PASSWORD IMMEDIATELY**

## Encryption
- Fernet (AES-128) encryption for API keys and passwords
- Encryption key: `/data/.encryption_key` (600 permissions)
- **BACKUP ENCRYPTION KEY** - cannot decrypt without it

## HTTPS/TLS
- Self-signed certificate for home network use
- Located: `/certs/cert.pem` and `/certs/key.pem`
- Valid for 10 years
- Browsers will show warning - accept to proceed
- Enable in `.env`: `ENABLE_HTTPS=true`

## CSRF Protection
- Token-based CSRF protection on all POST forms
- Tokens valid for 1 hour
- Automatic token injection via middleware

## Rate Limiting
- Login: 5 attempts per minute
- Analysis: 10 per hour
- Scheduler triggers: 3 per hour

## Security Headers
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- Strict-Transport-Security: 1 year (when HTTPS enabled)
- Content-Security-Policy: Restrictive

## File Permissions
- `.env`: 600 (owner read/write only)
- `.encryption_key`: 600 (owner read/write only)
- `key.pem`: 600 (owner read/write only)
- Database: 644 (contains encrypted secrets)

## Audit Logging
- Security events logged to `logs/security.log`
- Tracks: login attempts, API key changes, CSRF failures
- JSON format for easy parsing

## Maintenance
Weekly:
- Review security audit logs
- Update Python dependencies
- Check for security patches

Monthly:
- Change admin password
- Rotate API keys (optional)
- Test backup restoration

## Incident Response
If system is compromised:
1. Stop application immediately
2. Change all passwords and API keys
3. Regenerate encryption key (will require re-entering secrets)
4. Review audit logs for unauthorized access
5. Update to latest security patches

## Configuration
To enable full security, update `.env`:
```bash
# Enable HTTPS
ENABLE_HTTPS=true
WEB_HOST=0.0.0.0
WEB_PORT=8443
```

Then install dependencies:
```bash
pip install itsdangerous cryptography slowapi
```

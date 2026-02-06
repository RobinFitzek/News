# Deployment Checklist - Investment Monitor

## Pre-Deployment Checks

### 1. Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Directory permissions correct

### 2. Configuration
- [ ] API keys configured (Gemini, Perplexity)
- [ ] Database path verified
- [ ] Encryption key set (for API key storage)
- [ ] HTTPS certificates generated (if using HTTPS)
- [ ] Environment variables set

### 3. Database
- [ ] Database initialized
- [ ] Default admin user created
- [ ] **Admin password changed from default!**
- [ ] Database backup strategy in place
- [ ] Database directory has write permissions

### 4. Testing
- [ ] All unit tests passing
- [ ] Database tests passing
- [ ] Manual smoke tests completed
- [ ] Error handling verified
- [ ] Performance tests acceptable

### 5. Security
- [ ] Default admin password changed
- [ ] API keys encrypted
- [ ] HTTPS enabled (for production)
- [ ] CSRF protection enabled
- [ ] Rate limiting configured
- [ ] Security headers enabled

### 6. Monitoring & Logging
- [ ] Logs directory created
- [ ] Log rotation configured
- [ ] Health check endpoint tested
- [ ] Error alerting configured (optional)

### 7. Scheduler
- [ ] Scheduler settings configured
- [ ] Active hours set correctly
- [ ] Scan interval configured
- [ ] Email notifications configured (optional)

## Deployment Steps

### 1. Initial Setup

```bash
# Clone/update repository
cd /home/robin/Documents/GitHub/News

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from core.database import db; print('Database initialized')"
```

### 2. Configure Application

```bash
# Set up .env file (if using)
cp .env.example .env
nano .env

# Update settings:
# - API keys
# - Database path
# - Security settings
# - Email settings (optional)
```

### 3. Security Setup

```python
# Change admin password
from core.database import db
db.update_password('admin', 'YOUR_SECURE_PASSWORD_HERE')

# Add API keys
db.set_api_key('gemini', 'YOUR_GEMINI_API_KEY')
db.set_api_key('perplexity', 'YOUR_PERPLEXITY_API_KEY')
```

### 4. Test Configuration

```bash
# Run tests
python -m unittest discover tests -v

# Test health endpoint
python app.py &
sleep 5
curl http://localhost:8000/health
pkill -f "python app.py"
```

### 5. Production Deployment

#### Option A: Direct Run (Development)

```bash
python app.py
```

#### Option B: Production Server (Recommended)

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn app:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  --daemon
```

#### Option C: Systemd Service

Create `/etc/systemd/system/investment-monitor.service`:

```ini
[Unit]
Description=Investment Monitor Service
After=network.target

[Service]
Type=notify
User=robin
WorkingDirectory=/home/robin/Documents/GitHub/News
Environment="PATH=/home/robin/Documents/GitHub/News/venv/bin"
ExecStart=/home/robin/Documents/GitHub/News/venv/bin/gunicorn app:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable investment-monitor
sudo systemctl start investment-monitor
sudo systemctl status investment-monitor
```

### 6. Verify Deployment

```bash
# Check service status
systemctl status investment-monitor

# Check logs
tail -f logs/application.log

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/login

# Check scheduler
tail -f logs/application.log | grep -i scheduler
```

## Post-Deployment

### 1. Initial Configuration

- [ ] Log in to web interface
- [ ] Change admin password (if not done)
- [ ] Configure settings
- [ ] Add stocks to watchlist
- [ ] Set up analysis preferences
- [ ] Configure email notifications

### 2. Start Scheduler

- [ ] Start scheduler from web interface
- [ ] Verify scheduler is running
- [ ] Check scheduled job times
- [ ] Monitor first run

### 3. Monitoring Setup

```bash
# Set up log monitoring
tail -f logs/application.log

# Set up cron job for health checks
crontab -e

# Add:
*/5 * * * * curl -s http://localhost:8000/health > /dev/null || echo "Health check failed" | mail -s "Investment Monitor Alert" your@email.com
```

### 4. Backup Setup

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/robin/backups/investment-monitor"
mkdir -p $BACKUP_DIR

# Backup database
cp core/data/investment_monitor.db "$BACKUP_DIR/db_$DATE.db"

# Backup logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/ \;

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

chmod +x backup.sh

# Add to crontab (daily at 2 AM)
0 2 * * * /home/robin/Documents/GitHub/News/backup.sh >> /home/robin/backups/backup.log 2>&1
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
journalctl -u investment-monitor -n 50

# Check application logs
tail -100 logs/errors.log

# Verify Python path
which python
python --version

# Test manually
source venv/bin/activate
python app.py
```

### Database Locked

```bash
# Check for stuck processes
ps aux | grep investment-monitor

# Kill if necessary
sudo systemctl restart investment-monitor
```

### API Errors

```bash
# Check API keys
python -c "from core.database import db; print('Gemini:', bool(db.get_api_key('gemini'))); print('Perplexity:', bool(db.get_api_key('perplexity')))"

# Test API connectivity
curl -I https://api.gemini.google.com
curl -I https://api.perplexity.ai
```

### High Memory Usage

```bash
# Check memory
free -h

# Check process memory
ps aux | grep investment-monitor

# Reduce workers if needed
# Edit systemd service: --workers 2
```

## Maintenance

### Daily
- Monitor logs for errors
- Check scheduler runs
- Verify API quota usage

### Weekly
- Review analysis results
- Check database size
- Update watchlist
- Review performance

### Monthly
- Run full test suite
- Update dependencies
- Review and archive logs
- Database vacuum/optimize
- Security audit

### Quarterly
- Dependency security audit
- Performance optimization
- Feature review
- Backup verification

## Rollback Procedure

If deployment fails:

```bash
# Stop service
sudo systemctl stop investment-monitor

# Restore database backup
cp /home/robin/backups/investment-monitor/db_YYYYMMDD_HHMMSS.db core/data/investment_monitor.db

# Restore previous code version
git checkout previous-version

# Restart service
sudo systemctl start investment-monitor

# Verify
systemctl status investment-monitor
curl http://localhost:8000/health
```

## Success Criteria

✅ Service running without errors
✅ Health check returns "healthy"
✅ Can log in to web interface
✅ Scheduler executing on schedule
✅ Database operations working
✅ Logs being written
✅ API calls succeeding
✅ No critical errors in logs

**If all criteria met: Deployment successful! 🎉**

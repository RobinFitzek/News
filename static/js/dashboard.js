/**
 * Stockholm — Dashboard JavaScript
 * Clean, minimal interactions for investment intelligence
 */

// ============================================
// Theme & Performance Manager
// ============================================

class ThemeManager {
    constructor() {
        this.storageKey = 'stockholm-theme';
        this.init();
    }

    init() {
        const savedTheme = localStorage.getItem(this.storageKey);
        const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
        const initialTheme = savedTheme || (prefersLight ? 'light' : 'dark');

        this.applyTheme(initialTheme, false);
        this.bindToggle();
        this.watchSystemPreference();
    }

    applyTheme(theme, persist = true) {
        const normalized = theme === 'light' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', normalized);
        this.syncToggle(normalized);
        this.updateMetaThemeColor(normalized);

        if (persist) {
            localStorage.setItem(this.storageKey, normalized);
        }
    }

    bindToggle() {
        const toggle = document.getElementById('theme-toggle');
        if (!toggle) return;

        toggle.addEventListener('change', (event) => {
            const nextTheme = event.target.checked ? 'light' : 'dark';
            this.applyTheme(nextTheme);
        });
    }

    syncToggle(theme) {
        const toggle = document.getElementById('theme-toggle');
        if (toggle) {
            toggle.checked = theme === 'light';
        }
    }

    watchSystemPreference() {
        if (!window.matchMedia) return;

        const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
        const handleChange = (event) => {
            if (localStorage.getItem(this.storageKey)) return;
            this.applyTheme(event.matches ? 'light' : 'dark', false);
        };

        if (mediaQuery.addEventListener) {
            mediaQuery.addEventListener('change', handleChange);
        } else if (mediaQuery.addListener) {
            mediaQuery.addListener(handleChange);
        }
    }

    updateMetaThemeColor(theme) {
        const meta = document.querySelector('meta[name="theme-color"]');
        if (!meta) return;
        meta.setAttribute('content', theme === 'light' ? '#f8f8f7' : '#080810'); // adjusted for Breathe
    }
}

// ============================================
// Parallax & Depth Manager
// ============================================

class ParallaxManager {
    constructor() {
        this.isActive = true;
        this.mouseX = window.innerWidth / 2;
        this.mouseY = window.innerHeight / 2;
        this.currentX = this.mouseX;
        this.currentY = this.mouseY;
        
        this.luminary = document.getElementById('luminary');
        this.cards = new Map();
        
        this.init();
    }
    
    init() {
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        const settingOff = localStorage.getItem('stockholm-parallax') === 'false';
        
        if (prefersReducedMotion || isTouch || settingOff) {
            this.isActive = false;
            // No return here, we still want to monitor for setting changes if possible
            // but for now we follow existing logic
        }
        
        document.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
        });
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !this.cards.has(entry.target)) {
                    const rect = entry.target.getBoundingClientRect();
                    this.cards.set(entry.target, {
                        baseY: window.scrollY + rect.top
                    });
                    if (this.isActive) entry.target.style.willChange = 'transform';
                }
            });
        }, { rootMargin: '100px 0px' });
        
        const attachCards = () => {
            document.querySelectorAll('.card, .glass, .glass-card').forEach(card => {
                if (!this.cards.has(card)) observer.observe(card);
            });
        };
        attachCards();
        
        const domObserver = new MutationObserver(mutations => {
            let added = false;
            mutations.forEach(m => {
                m.addedNodes.forEach(node => {
                    if (node.nodeType === 1) added = true;
                });
            });
            if (added) attachCards();
        });
        domObserver.observe(document.body, { childList: true, subtree: true });
        
        requestAnimationFrame(() => this.tick());
    }

    setIsActive(val) {
        this.isActive = val;
        document.querySelectorAll('.card, .glass, .glass-card').forEach(card => {
            card.style.willChange = val ? 'transform' : 'auto';
            if (!val) card.style.transform = '';
        });
    }
    
    tick() {
        if (!this.isActive) return;
        
        this.currentX += (this.mouseX - this.currentX) * 0.08;
        this.currentY += (this.mouseY - this.currentY) * 0.08;
        
        if (this.luminary) {
            const shiftX = (this.currentX - window.innerWidth / 2) * 0.04;
            const shiftY = (this.currentY - window.innerHeight / 2) * 0.04;
            this.luminary.style.transform = `translate(${shiftX}px, ${shiftY}px)`;
        }
        
        const scrollY = window.scrollY;
        const maxShift = 8;
        
        this.cards.forEach((data, card) => {
            const distance = scrollY + window.innerHeight - data.baseY;
            if (distance > -window.innerHeight && distance < window.innerHeight * 2) {
                let shift = (distance - window.innerHeight / 2) * 0.06;
                shift = Math.max(-maxShift, Math.min(maxShift, shift));
                
                card.style.setProperty('--parallax-y', `${shift}px`);
            }
        });
        
        requestAnimationFrame(() => this.tick());
    }
}

// ============================================
// Chart Configuration — Minimal B&W Styling
// ============================================

function getThemeColor(variableName, fallback) {
    const value = getComputedStyle(document.documentElement)
        .getPropertyValue(variableName)
        .trim();
    return value || fallback;
}

function getThemeChartColors() {
    return {
        primary: getThemeColor('--text-primary', '#ffffff'),
        secondary: getThemeColor('--text-secondary', '#a0a0a0'),
        muted: getThemeColor('--text-muted', '#666666'),
        border: getThemeColor('--border-light', 'rgba(255, 255, 255, 0.08)'),
        background: getThemeColor('--bg-card', '#0a0a0a'),
        text: getThemeColor('--text-secondary', '#a0a0a0'),
        textPrimary: getThemeColor('--text-primary', '#ffffff')
    };
}

function buildChartDefaults() {
    const chartColors = getThemeChartColors();
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const tooltipBg = isLight ? 'rgba(255, 255, 255, 0.75)' : 'rgba(10, 10, 10, 0.75)';

    Chart.defaults.font.family = 'JetBrains Mono';
    Chart.defaults.color = chartColors.secondary;

    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                backgroundColor: tooltipBg,
                borderColor: chartColors.border,
                borderWidth: 1,
                titleColor: chartColors.textPrimary,
                bodyColor: chartColors.muted,
                padding: 12,
                cornerRadius: 0,
                displayColors: false,
                boxPadding: 4
            }
        },
        scales: {
            x: {
                grid: {
                    color: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)',
                    drawBorder: false
                },
                border: { display: false },
                ticks: {
                    color: chartColors.secondary,
                    font: {
                        family: 'JetBrains Mono',
                        size: 10
                    }
                }
            },
            y: {
                grid: {
                    color: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)',
                    drawBorder: false
                },
                border: { display: false },
                ticks: {
                    color: chartColors.secondary,
                    font: {
                        family: 'JetBrains Mono',
                        size: 10
                    }
                }
            }
        }
    };
}

function createLineChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const chartDefaults = buildChartDefaults();
    const context2d = ctx.getContext('2d');
    const positiveColor = getThemeColor('--signal-positive', '#4ade80');
    const negativeColor = getThemeColor('--signal-negative', '#ef4444');

    data.datasets.forEach(ds => {
        const values = ds.data;
        const isPositive = values.length > 1 ? values[values.length - 1] >= values[0] : true;
        const signalColor = isPositive ? positiveColor : negativeColor;

        ds.borderColor = signalColor;
        ds.borderWidth = 1.5;
        ds.pointRadius = 0;
        ds.pointHoverRadius = 4;
        ds.pointHoverBorderWidth = 2;
        ds.fill = true;
        
        const gradient = context2d.createLinearGradient(0, 0, 0, ctx.parentElement ? ctx.parentElement.clientHeight : 300);
        gradient.addColorStop(0, `${signalColor}40`);
        gradient.addColorStop(1, `${signalColor}00`);
        
        ds.backgroundColor = gradient;
        ds.tension = 0.3;
    });

    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            ...chartDefaults,
            ...options
        }
    });
}

function createBarChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const chartDefaults = buildChartDefaults();
    const positiveColor = getThemeColor('--signal-positive', '#4ade80');
    const negativeColor = getThemeColor('--signal-negative', '#ef4444');
    const neutralColor = getThemeColor('--glow-neutral', '#e2e2e2');

    data.datasets.forEach(ds => {
        ds.borderRadius = 0;
        ds.barPercentage = 0.9;
        ds.categoryPercentage = 0.9;
        
        ds.backgroundColor = function(context) {
            const chart = context.chart;
            const {ctx: chartCtx, chartArea} = chart;
            if (!chartArea) return neutralColor;
            
            const value = context.raw;
            const targetColor = value >= 0 ? positiveColor : negativeColor;
            
            const gradient = chartCtx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
            gradient.addColorStop(0, neutralColor);
            gradient.addColorStop(1, targetColor);
            return gradient;
        };
        ds.hoverBackgroundColor = '#ffffff';
    });

    return new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...chartDefaults,
            hover: { mode: 'index', intersect: false },
            ...options
        }
    });
}

function createDoughnutChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const chartDefaults = buildChartDefaults();
    const chartColors = getThemeChartColors();

    return new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: chartColors.textPrimary,
                        padding: 16,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: chartDefaults.plugins.tooltip
            },
            ...options
        }
    });
}
// ============================================
// Micro-Charts & KPIs (BREATHE-3)
// ============================================

class SparkBar {
    constructor(container, data) {
        this.container = container;
        this.data = data;
        this.render();
    }
    
    render() {
        const width = 40;
        const height = 16;
        const gap = 1;
        const barWidth = (width - gap * (this.data.length - 1)) / this.data.length;
        
        const min = Math.min(...this.data);
        const max = Math.max(...this.data);
        const range = max - min || 1;
        
        const positiveColor = getThemeColor('--signal-positive', '#4ade80');
        const negativeColor = getThemeColor('--signal-negative', '#ef4444');
        const isPositive = this.data[this.data.length - 1] >= 0;
        const baseColor = isPositive ? positiveColor : negativeColor;
        
        let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
        
        this.data.forEach((val, i) => {
            const hNormalized = Math.max(1, ((val - min) / range) * height);
            const x = i * (barWidth + gap);
            const opacity = 0.3 + (0.7 * (i / (this.data.length - 1)));
            
            svg += `<rect x="${x}" y="${height - hNormalized}" width="${barWidth}" height="${hNormalized}" fill="${baseColor}" opacity="${opacity}" />`;
        });
        svg += '</svg>';
        
        this.container.innerHTML = svg;
    }
}

class SignalMatrix {
    constructor(container, valuePct) {
        this.container = container;
        this.valuePct = Math.max(0, Math.min(100, valuePct));
        this.render();
    }
    
    render() {
        const totalDots = 10;
        const filledDots = Math.round((this.valuePct / 100) * totalDots);
        const size = 6;
        const gap = 3;
        const width = (size * totalDots) + (gap * (totalDots - 1));
        const height = size;
        
        const positiveColor = getThemeColor('--signal-positive', '#4ade80');
        const negativeColor = getThemeColor('--signal-negative', '#ef4444');
        const signalColor = this.valuePct >= 50 ? positiveColor : negativeColor;
        const emptyColor = getThemeColor('--border-light', 'rgba(255,255,255,0.08)');
        
        let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
        for (let i = 0; i < totalDots; i++) {
            const x = i * (size + gap);
            const color = i < filledDots ? signalColor : emptyColor;
            svg += `<rect x="${x}" y="0" width="${size}" height="${size}" fill="${color}" />`;
        }
        svg += '</svg>';
        
        this.container.innerHTML = svg;
    }
}

class HorizonChart {
    constructor(container, data) {
        this.container = container;
        this.data = data;
        this.render();
    }
    
    render() {
        if (!this.data || this.data.length === 0) return;
        
        const absMax = Math.max(...this.data.map(Math.abs)) || 1;
        
        function hexToRgb(hex) {
            let color = hex.replace('#', '');
            if (color.length === 3) color = color[0]+color[0]+color[1]+color[1]+color[2]+color[2];
            return `${parseInt(color.substring(0, 2), 16)}, ${parseInt(color.substring(2, 4), 16)}, ${parseInt(color.substring(4, 6), 16)}`;
        }
        
        const posHex = getThemeColor('--signal-positive', '#4ade80').trim();
        const negHex = getThemeColor('--signal-negative', '#ef4444').trim();
        const posRgb = hexToRgb(posHex);
        const negRgb = hexToRgb(negHex);
        
        const step = 100 / this.data.length;
        let stops = [];
        
        this.data.forEach((val, i) => {
            const intensity = Math.abs(val) / absMax;
            const rgb = val >= 0 ? posRgb : negRgb;
            const alpha = (0.2 + (0.8 * intensity)).toFixed(2);
            const start = (i * step).toFixed(2);
            const end = ((i + 1) * step).toFixed(2);
            
            stops.push(`rgba(${rgb}, ${alpha}) ${start}%`);
            stops.push(`rgba(${rgb}, ${alpha}) ${end}%`);
        });
        
        this.container.style.height = '20px';
        this.container.style.width = '100%';
        this.container.style.background = `linear-gradient(to right, ${stops.join(', ')})`;
        this.container.style.borderRadius = '0';
    }
}

class ConstellationGraph {
    constructor(container, nodes, edges) {
        this.container = container;
        this.nodes = nodes;
        this.edges = edges;
        this.width = container.clientWidth || 300;
        this.height = container.clientHeight || 200;
        this.render();
    }
    
    render() {
        if (!this.nodes || this.nodes.length === 0) return;
        
        const iterations = 50;
        const positions = {};
        
        this.nodes.forEach(n => {
            positions[n.id] = {
                x: 10 + Math.random() * (this.width - 20),
                y: 10 + Math.random() * (this.height - 20),
                vx: 0, vy: 0
            };
        });
        
        const k = Math.sqrt((this.width * this.height) / this.nodes.length);
        const repel = (dist) => (k * k) / dist;
        const attract = (dist) => (dist * dist) / k;
        
        for (let i = 0; i < iterations; i++) {
            for (let j = 0; j < this.nodes.length; j++) {
                for (let l = j + 1; l < this.nodes.length; l++) {
                    const u = positions[this.nodes[j].id];
                    const v = positions[this.nodes[l].id];
                    let dx = u.x - v.x;
                    let dy = u.y - v.y;
                    let dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
                    const force = repel(dist);
                    const fx = (dx / dist) * force;
                    const fy = (dy / dist) * force;
                    u.vx += fx; u.vy += fy;
                    v.vx -= fx; v.vy -= fy;
                }
            }
            this.edges.forEach(e => {
                const u = positions[e.source];
                const v = positions[e.target];
                if (!u || !v) return;
                let dx = u.x - v.x;
                let dy = u.y - v.y;
                let dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
                const force = attract(dist) * Math.max(0.1, e.strength);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                u.vx -= fx; u.vy -= fy;
                v.vx += fx; v.vy += fy;
            });
            this.nodes.forEach(n => {
                const pos = positions[n.id];
                pos.x = Math.max(10, Math.min(this.width - 10, pos.x + pos.vx * 0.1));
                pos.y = Math.max(10, Math.min(this.height - 10, pos.y + pos.vy * 0.1));
                pos.vx = 0; pos.vy = 0;
            });
        }
        
        let svg = `<svg width="100%" height="100%" viewBox="0 0 ${this.width} ${this.height}" xmlns="http://www.w3.org/2000/svg">`;
        const lineCol = getThemeColor('--border-light', 'rgba(255,255,255,0.1)');
        
        this.edges.forEach(e => {
            const u = positions[e.source];
            const v = positions[e.target];
            if (!u || !v) return;
            const w = Math.max(1, (e.strength || 0.5) * 3);
            svg += `<line x1="${u.x}" y1="${u.y}" x2="${v.x}" y2="${v.y}" stroke="${lineCol}" stroke-width="${w}" opacity="0.5" />`;
        });
        
        const pCol = getThemeColor('--signal-positive', '#4ade80');
        const nCol = getThemeColor('--signal-negative', '#ef4444');
        const neuCol = getThemeColor('--glow-neutral', '#e2e2e2');
        
        this.nodes.forEach(n => {
            const pos = positions[n.id];
            const size = Math.max(6, (n.weight || 0.5) * 20);
            const hs = size / 2;
            let col = neuCol;
            if (n.signal > 0) col = pCol;
            else if (n.signal < 0) col = nCol;
            
            svg += `<rect x="${pos.x - hs}" y="${pos.y - hs}" width="${size}" height="${size}" fill="${col}" style="filter: drop-shadow(0 0 ${size/2}px ${col})" />`;
        });
        
        svg += '</svg>';
        this.container.innerHTML = svg;
    }
}

class CountUp {
    constructor(element, endVal, duration = 400) {
        this.element = element;
        this.endVal = endVal;
        this.duration = duration;
        this.noiseChars = '0123456789#%&?'.split('');
        this.originalText = element.textContent.trim();
        this.start();
    }

    start() {
        const startTime = performance.now();
        const startVal = this.originalText;
        const targetStr = String(this.endVal);

        const tick = (now) => {
            const progress = Math.min((now - startTime) / this.duration, 1);

            if (progress < 0.4) {
                // Phase 1: Defuse to noise
                this.element.textContent = this.generateNoise(startVal.length || targetStr.length);
                this.element.style.color = 'var(--glow-gold)';
            } else if (progress < 0.9) {
                // Phase 2: Shimmering to resolve
                this.element.textContent = this.generateNoise(targetStr.length);
            } else {
                // Final
                let displayVal = targetStr;
                if (this.originalText.startsWith('$')) displayVal = '$' + displayVal;
                if (this.originalText.endsWith('%')) displayVal = displayVal + '%';
                this.element.textContent = displayVal;
                this.element.style.color = '';
                return;
            }

            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }

    generateNoise(len) {
        let s = '';
        for(let i=0; i<len; i++) s += this.noiseChars[Math.floor(Math.random()*this.noiseChars.length)];
        return s;
    }
}// ============================================
// Toast Notifications — Minimal Alerts
// ============================================

function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    const container = document.getElementById('toast-container') || createToastContainer();
    container.appendChild(toast);

    // Auto-remove with fade out
    setTimeout(() => {
        toast.classList.add('hiding');
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = `
        position: fixed;
        top: 80px;
        right: 24px;
        z-index: 10000;
        display: flex;
        flex-direction: column;
        gap: 12px;
        pointer-events: none;
    `;
    document.body.appendChild(container);
    return container;
}

// ============================================
// API Utilities
// ============================================

async function fetchJSON(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showToast(`Error: ${error.message}`, 'danger');
        throw error;
    }
}

// ============================================
// Real-time Updates
// ============================================

class RealtimeUpdater {
    constructor(interval = 30000) {
        this.interval = interval;
        this.timerId = null;
        this.updateCallbacks = [];
    }

    start() {
        if (this.timerId) return;

        this.timerId = setInterval(() => {
            this.updateCallbacks.forEach(callback => {
                try {
                    callback();
                } catch (error) {
                    console.error('Update callback error:', error);
                }
            });
        }, this.interval);

        console.log(`Stockholm: Real-time updates active (${this.interval / 1000}s interval)`);
    }

    stop() {
        if (this.timerId) {
            clearInterval(this.timerId);
            this.timerId = null;
        }
    }

    addCallback(callback) {
        this.updateCallbacks.push(callback);
    }

    removeCallback(callback) {
        this.updateCallbacks = this.updateCallbacks.filter(cb => cb !== callback);
    }
}

// ============================================
// Dashboard Functions
// ============================================

function updateAPIStatus() {
    fetchJSON('/api/status')
        .then(data => {
            const apiUsage = data.api_usage || {};

            function updateApiCard(key, used, limit) {
                const safeLimit = Math.max(limit || 1, 1);
                const usageEl = document.getElementById('api-usage-' + key);
                const progressEl = document.getElementById('api-progress-' + key);

                if (usageEl) {
                    usageEl.innerHTML = used + '<span style="color: var(--text-muted); font-weight: 300;">/' + safeLimit + '</span>';
                }
                if (progressEl) {
                    const pct = Math.min(100, (used / safeLimit) * 100);
                    progressEl.style.width = pct + '%';
                }
            }

            // Update Perplexity usage
            const perplexityUsed = apiUsage.perplexity?.used_today || 0;
            const perplexityLimit = apiUsage.perplexity?.daily_limit || 100;
            updateApiCard('perplexity', perplexityUsed, perplexityLimit);

            // Update Gemini Flash
            const flashUsed = apiUsage.gemini?.flash?.used_today || 0;
            const flashLimit = apiUsage.gemini?.flash?.daily_limit || 50;
            updateApiCard('gemini-flash', flashUsed, flashLimit);

            // Update Gemini Pro
            const proUsed = apiUsage.gemini?.pro?.used_today || 0;
            const proLimit = apiUsage.gemini?.pro?.daily_limit || 50;
            updateApiCard('gemini-pro', proUsed, proLimit);

            // Update dynamic provider cards
            const providers = data.providers || [];
            providers.forEach(p => {
                updateApiCard('provider-' + p.id, p.used_today || 0, p.daily_limit || 1);
            });

            // Top bar counters
            updateElement('sys-stale-count', data.stale_analyses || 0);
        })
        .catch(error => console.error('Failed to update API status:', error));
}

function updateProgressBar(id, percent) {
    const bar = document.getElementById(id);
    if (bar) {
        bar.style.width = `${Math.min(percent, 100)}%`;
    }
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

// ============================================
// Animation Utilities
// ============================================

function animateValue(element, start, end, duration = 400) {
    if (!element) return;

    const startTime = performance.now();
    const diff = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out quad
        const easeProgress = 1 - (1 - progress) * (1 - progress);
        const current = start + (diff * easeProgress);

        element.textContent = Math.round(current);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Intersection Observer for reveal animations
function observeAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    document.querySelectorAll('.animate-on-scroll, .card, .grid').forEach(el => {
        observer.observe(el);
    });
}

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    const lsEnabled = localStorage.getItem('stockholm-loading-screen') !== 'false' && !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const startDelay = lsEnabled ? 2200 : 0; // Wait for loading screen if active

    // Initialize core managers
    window.themeManager = new ThemeManager();
    window.parallaxManager = new ParallaxManager();

    // 7.1 Page Load Sequence
    const runSequence = () => {
        // t=0: Shell fades in
        document.body.classList.add('breathe-ready');
        
        // t=200ms: Luminary orbs drift in (via CSS class on luminary)
        setTimeout(() => {
            const luminary = document.getElementById('luminary');
            if (luminary) luminary.classList.add('is-active');
        }, 200);

        // t=400ms: Navigation materializes
        setTimeout(() => {
            const nav = document.querySelector('nav');
            if (nav) nav.classList.add('is-visible');
        }, 400);

        // t=600ms: Start observing card entrances
        setTimeout(() => {
            observeAnimations();
        }, 600);
    };

    if (lsEnabled) {
        document.addEventListener('diffusionComplete', () => {
            setTimeout(runSequence, 400); // Small pause after sword shatter
        });
    } else {
        runSequence();
    }

    // Setup real-time updates for dashboard
    if (document.getElementById('dashboard-page')) {
        window.realtimeUpdater = new RealtimeUpdater(30000);
        window.realtimeUpdater.addCallback(updateAPIStatus);
        window.realtimeUpdater.start();

        // Initial update
        updateAPIStatus();
    }

    // Initialize progress bars from data attributes
    document.querySelectorAll('[data-width]').forEach(el => {
        const targetWidth = el.getAttribute('data-width');
        // Animate progress bar
        el.style.width = '0%';
        requestAnimationFrame(() => {
            el.style.width = targetWidth + '%';
        });
    });

    // Observe scroll animations
    observeAnimations();

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    console.log('%cStockholm', 'font-size: 16px; font-weight: 400; color: #0a0a0a; font-family: "EB Garamond", Georgia, serif;');
    console.log('%cInvestment Intelligence', 'color: #737373;');
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.realtimeUpdater) {
        window.realtimeUpdater.stop();
    }
});

// Export utilities
window.dashboardUtils = {
    showToast,
    fetchJSON,
    createLineChart,
    createBarChart,
    createDoughnutChart,
    animateValue,
    chartColors: getThemeChartColors(),
    getThemeChartColors,
    buildChartDefaults,
    SparkBar,
    SignalMatrix,
    HorizonChart,
    ConstellationGraph,
    CountUp
};

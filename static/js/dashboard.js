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
        meta.setAttribute('content', theme === 'light' ? '#f8f8f7' : '#000000');
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

    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: chartColors.textPrimary,
                    font: {
                        family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
                        size: 12
                    }
                }
            },
            tooltip: {
                backgroundColor: 'rgba(10, 10, 10, 0.95)',
                borderColor: chartColors.border,
                borderWidth: 1,
                titleColor: '#ffffff',
                bodyColor: chartColors.muted,
                padding: 12,
                cornerRadius: 4,
                displayColors: true,
                boxPadding: 4
            }
        },
        scales: {
            x: {
                grid: {
                    color: chartColors.border,
                    drawBorder: false
                },
                ticks: {
                    color: chartColors.text,
                    font: {
                        family: 'SF Mono, Monaco, monospace',
                        size: 10
                    }
                }
            },
            y: {
                grid: {
                    color: chartColors.border,
                    drawBorder: false
                },
                ticks: {
                    color: chartColors.text,
                    font: {
                        family: 'SF Mono, Monaco, monospace',
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

    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            ...chartDefaults,
            ...options,
            elements: {
                line: {
                    tension: 0.3,
                    borderWidth: 1.5
                },
                point: {
                    radius: 0,
                    hoverRadius: 4,
                    hoverBorderWidth: 2
                }
            }
        }
    });
}

function createBarChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const chartDefaults = buildChartDefaults();

    return new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...chartDefaults,
            ...options,
            borderRadius: 2
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
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-8px)';
        setTimeout(() => toast.remove(), 200);
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

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme
    window.themeManager = new ThemeManager();

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
    buildChartDefaults
};

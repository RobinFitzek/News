/**
 * AI Investment Monitor - Dashboard JavaScript
 * Handles charts, real-time updates, and interactivity
 */

// ============================================
// Theme Management
// ============================================

class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'dark';
        this.performanceMode = localStorage.getItem('performanceMode') === 'true';
        this.init();
    }

    init() {
        this.applyTheme();
        this.applyPerformanceMode();
        this.setupToggleListeners();
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
    }

    applyPerformanceMode() {
        if (this.performanceMode) {
            document.documentElement.setAttribute('data-performance', 'true');
        } else {
            document.documentElement.removeAttribute('data-performance');
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('theme', this.theme);
        this.applyTheme();

        // Animate transition
        document.body.style.transition = 'background-color 0.3s ease';
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }

    togglePerformanceMode() {
        this.performanceMode = !this.performanceMode;
        localStorage.setItem('performanceMode', this.performanceMode);
        this.applyPerformanceMode();

        // Show toast notification
        showToast(
            this.performanceMode ? '⚡ Performance Mode Enabled' : '✨ Visual Effects Enabled',
            'info'
        );
    }

    setupToggleListeners() {
        const themeToggle = document.getElementById('theme-toggle');
        const perfToggle = document.getElementById('performance-toggle');

        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        if (perfToggle) {
            perfToggle.addEventListener('click', () => this.togglePerformanceMode());
            perfToggle.checked = this.performanceMode;
        }
    }
}

// ============================================
// Chart Utilities (Chart.js)
// ============================================

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                color: getComputedStyle(document.documentElement)
                    .getPropertyValue('--text-primary').trim(),
                font: {
                    family: 'Inter, sans-serif',
                    size: 12
                }
            }
        }
    },
    scales: {
        x: {
            grid: {
                color: 'rgba(255, 255, 255, 0.05)'
            },
            ticks: {
                color: getComputedStyle(document.documentElement)
                    .getPropertyValue('--text-secondary').trim()
            }
        },
        y: {
            grid: {
                color: 'rgba(255, 255, 255, 0.05)'
            },
            ticks: {
                color: getComputedStyle(document.documentElement)
                    .getPropertyValue('--text-secondary').trim()
            }
        }
    }
};

function createLineChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: { ...chartDefaults, ...options }
    });
}

function createBarChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'bar',
        data: data,
        options: { ...chartDefaults, ...options }
    });
}

function createDoughnutChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: getComputedStyle(document.documentElement)
                            .getPropertyValue('--text-primary').trim()
                    }
                }
            },
            ...options
        }
    });
}

// ============================================
// Toast Notifications
// ============================================

function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} animate-fade-in`;
    toast.textContent = message;

    const container = document.getElementById('toast-container') || createToastContainer();
    container.appendChild(toast);

    // Auto-remove
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    gap: 10px;
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

        console.log(`Real-time updates started (every ${this.interval}ms)`);
    }

    stop() {
        if (this.timerId) {
            clearInterval(this.timerId);
            this.timerId = null;
            console.log('Real-time updates stopped');
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
// Dashboard-specific Functions
// ============================================

function updateAPIStatus() {
    fetchJSON('/api/status')
        .then(data => {
            // Update Perplexity usage
            const perplexityUsed = data.perplexity?.used_today || 0;
            const perplexityLimit = data.perplexity?.daily_limit || 100;
            const perplexityPercent = (perplexityUsed / perplexityLimit) * 100;

            updateProgressBar('perplexity-progress', perplexityPercent);
            updateElement('perplexity-usage', `${perplexityUsed}/${perplexityLimit}`);

            // Update Gemini usage
            const flashUsed = data.gemini?.flash?.used_today || 0;
            const flashLimit = data.gemini?.flash?.daily_limit || 50;
            const flashPercent = (flashUsed / flashLimit) * 100;

            updateProgressBar('gemini-flash-progress', flashPercent);
            updateElement('gemini-flash-usage', `${flashUsed}/${flashLimit}`);

            const proUsed = data.gemini?.pro?.used_today || 0;
            const proLimit = data.gemini?.pro?.daily_limit || 50;
            const proPercent = (proUsed / proLimit) * 100;

            updateProgressBar('gemini-pro-progress', proPercent);
            updateElement('gemini-pro-usage', `${proUsed}/${proLimit}`);
        })
        .catch(error => console.error('Failed to update API status:', error));
}

function updateProgressBar(id, percent) {
    const bar = document.getElementById(id);
    if (bar) {
        bar.style.width = `${Math.min(percent, 100)}%`;

        // Change color based on usage
        if (percent >= 90) {
            bar.style.background = 'var(--gradient-danger)';
        } else if (percent >= 70) {
            bar.style.background = 'var(--gradient-warning)';
        } else {
            bar.style.background = 'var(--gradient-primary)';
        }
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

function animateValue(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16); // 60 FPS
    let current = start;

    const timer = setInterval(() => {
        current += increment;

        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }

        element.textContent = Math.round(current);
    }, 16);
}

function observeAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.card, .glass-card').forEach(card => {
        observer.observe(card);
    });
}

// ============================================
// Initialize on DOM Ready
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme manager
    window.themeManager = new ThemeManager();

    // Setup real-time updates if on dashboard
    if (document.getElementById('dashboard-page')) {
        window.realtimeUpdater = new RealtimeUpdater(30000);
        window.realtimeUpdater.addCallback(updateAPIStatus);
        window.realtimeUpdater.start();

        // Initial update
        updateAPIStatus();
    }

    // Observe animations
    observeAnimations();

    // Add smooth scroll to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.realtimeUpdater) {
        window.realtimeUpdater.stop();
    }
});

// Export for use in other scripts
window.dashboardUtils = {
    showToast,
    fetchJSON,
    createLineChart,
    createBarChart,
    createDoughnutChart,
    animateValue
};

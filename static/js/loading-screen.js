/**
 * BREATHE-4: Mercury Diffusion & Ascendant Sword
 * The UI IS the loading animation.
 */

class TextDiffuser {
    constructor() {
        this.noiseChars = '░▒▓█╔╗╝╚║═╠╣╦╩╬│─┼@#$%&?!'.split('');
        this.nodesToAnimate = [];
        this.isComplete = false;
        
        // Exclude specific heavy nodes or elements that shouldn't diffuse
        this.excludeTags = ['SCRIPT', 'STYLE', 'SVG', 'IMG', 'CANVAS', 'INPUT', 'TEXTAREA', 'SELECT', 'CODE', 'PRE'];
        
        this.prepareDocument();
        this.startDiffusion();
    }
    
    prepareDocument() {
        const textNodes = [];
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        
        let node;
        while (node = walker.nextNode()) {
            const parent = node.parentNode;
            if (this.excludeTags.includes(parent.nodeName)) continue;
            
            // Only process non-empty text
            const text = node.nodeValue;
            if (text.trim().length === 0) continue;
            
            // Re-evaluating: doing this on EVERY deeply nested span could be expensive.
            // But spec says: "Every text node on the page is wrapped". 
            // We'll wrap individual non-space characters in spans.
            textNodes.push({ node, text, parent });
        }
        
        const pageHeight = Math.max(document.body.scrollHeight, window.innerHeight);
        
        textNodes.forEach(({ node, text, parent }) => {
            const fragment = document.createDocumentFragment();
            let hasChars = false;
            
            // Approximate vertical position (0 to 1) for the wave effect
            let normalizedPos = 0;
            if (parent.getBoundingClientRect) {
                const rect = parent.getBoundingClientRect();
                normalizedPos = Math.max(0, Math.min(1, (rect.top + window.scrollY) / pageHeight));
            }
            
            for (let i = 0; i < text.length; i++) {
                const char = text[i];
                if (char.trim() === '') {
                    fragment.appendChild(document.createTextNode(char));
                    continue;
                }
                
                const span = document.createElement('span');
                span.className = 'diffuser-char';
                span.setAttribute('data-real', char);
                // Numbers resolve last
                const isNumber = /[0-9]/.test(char);
                const resolveBonus = isNumber ? -0.1 : 0;
                
                // Base probability of resolving each tick
                // Top of page resolves faster
                const prob = 0.05 + ((1 - normalizedPos) * 0.04) + resolveBonus;
                
                span.setAttribute('data-prob', prob.toString());
                span.textContent = this.getRandomNoise();
                
                // Start with glowing noise style
                span.style.color = 'var(--glow-gold)';
                span.style.opacity = '0.4';
                
                fragment.appendChild(span);
                this.nodesToAnimate.push({ span, real: char, prob, resolved: false, isNumber });
                hasChars = true;
            }
            
            if (hasChars) {
                parent.replaceChild(fragment, node);
            }
        });
    }
    
    getRandomNoise() {
        return this.noiseChars[Math.floor(Math.random() * this.noiseChars.length)];
    }
    
    startDiffusion() {
        const frameInterval = 60; // 60ms between loops to avoid burning CPU while still shimmering
        let lastTime = 0;
        
        const tick = (now) => {
            if (this.isComplete) return;
            
            // Only update on interval
            if (now - lastTime < frameInterval) {
                requestAnimationFrame(tick);
                return;
            }
            lastTime = now;
            
            let allResolved = true;
            
            this.nodesToAnimate.forEach(target => {
                if (target.resolved) return;
                allResolved = false;
                
                if (Math.random() < target.prob) {
                    target.resolved = true;
                    target.span.textContent = target.real;
                    target.span.classList.add('resolved');
                    target.span.style.color = '';
                    target.span.style.opacity = '';
                    
                    if (target.isNumber) {
                        target.span.style.textShadow = '0 0 12px var(--glow-gold)';
                        setTimeout(() => { target.span.style.textShadow = ''; }, 200);
                    }
                } else {
                    // re-randomize
                    if (Math.random() > 0.6) {
                        target.span.textContent = this.getRandomNoise();
                    }
                }
                
                // Increase probability slightly over time to ensure it finishes
                target.prob += 0.005;
            });
            
            if (allResolved) {
                this.isComplete = true;
                const event = new CustomEvent('diffusionComplete');
                document.dispatchEvent(event);
            } else {
                requestAnimationFrame(tick);
            }
        };
        
        requestAnimationFrame(tick);
    }
}

class AscendantSword {
    constructor() {
        this.container = document.createElement('div');
        this.container.className = 'ascendant-sword';
        // Wait, where does the sword exist? "position: fixed center screen"
        this.container.style.cssText = `
            position: fixed;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            font-family: var(--font-data, 'JetBrains Mono', monospace);
            font-size: 14px;
            color: var(--glow-gold);
            text-align: center;
            line-height: 1.1;
            z-index: 9999;
            pointer-events: none;
            white-space: pre;
            text-shadow: 0 0 15px var(--glow-gold);
            opacity: 0.8;
            transition: opacity 0.3s ease, transform 0.8s cubic-bezier(.34, 1.56, .64, 1);
        `;
        
        const swordArt = `
      /| ________________
O|===|* >________________>
      \\|
`;
        this.container.textContent = swordArt;
        document.body.appendChild(this.container);
        
        document.addEventListener('diffusionComplete', () => this.shatter());
    }
    
    shatter() {
        // Flash amber then shatter
        this.container.style.color = 'var(--glow-amber)';
        this.container.style.textShadow = '0 0 30px var(--glow-amber)';
        this.container.style.transform = 'translate(-50%, -50%) scale(1.1)';
        
        setTimeout(() => {
            this.container.style.opacity = '0';
            this.spawnParticles();
            setTimeout(() => this.container.remove(), 500);
        }, 300);
    }
    
    spawnParticles() {
        const luminary = document.getElementById('luminary');
        if (!luminary) return;
        
        const rect = this.container.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'luminary-particle shard';
            const size = Math.random() * 4 + 2;
            
            const angle = Math.random() * Math.PI * 2;
            const dist = Math.random() * 80 + 20;
            const tx = Math.cos(angle) * dist;
            const ty = Math.sin(angle) * dist;
            
            particle.style.cssText = `
                position: absolute;
                left: ${cx}px;
                top: ${cy}px;
                width: ${size}px;
                height: ${size}px;
                background: var(--glow-amber);
                box-shadow: 0 0 ${size}px var(--glow-amber);
                border-radius: 50%;
                opacity: 0.8;
                pointer-events: none;
                transition: transform 1.5s cubic-bezier(.1, .8, .2, 1), opacity 1.5s ease-out;
            `;
            
            luminary.appendChild(particle);
            
            // Trigger animation
            requestAnimationFrame(() => {
                particle.style.transform = `translate(${tx}px, ${ty}px) scale(0)`;
                particle.style.opacity = '0';
                setTimeout(() => particle.remove(), 1500);
            });
        }
    }
}

// Disable diffusion completely if user prefers reduced motion or turned it off in settings
if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    // skip
} else if (localStorage.getItem('stockholm-loading-screen') === 'false') {
    // skipped via Appearance settings
} else {
    document.addEventListener('DOMContentLoaded', () => {
        new TextDiffuser();
        // new AscendantSword(); // Disabled per user request
    });
}

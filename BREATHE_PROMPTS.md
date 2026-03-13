# Breathe Design — Implementation Prompts

Structured XML prompts for implementing the "Breathe" glasmorphic design language rewrite.
Feed each prompt individually to an LLM in phase order. Each prompt is self-contained.
Complete and verify each phase before starting the next — layers depend on prior work.

**Phase order:** A → B → C → D → E → F
**Reference:** `TODO.md` sections BREATHE-1 through BREATHE-9

---

## Phase A — Foundation

### Prompt A1 · CSS Token System

```xml
<prompt id="A1" phase="A" title="Breathe CSS Token System">

  <objective>
    Replace the existing CSS custom property system in static/css/modern.css with the
    complete Breathe design token vocabulary. This is purely additive token work —
    no component styles change yet. All existing variable names must be preserved as
    aliases so nothing breaks while the new system is introduced.
  </objective>

  <read_first>
    <file>static/css/modern.css</file>
    <file>templates/base.html</file>
  </read_first>

  <context>
    The current design system is called "Nordisch Klar" — flat, no glassmorphism,
    no depth layers. It uses CSS custom properties on :root and [data-theme="light"].
    The new system is called "Breathe". It introduces five new token categories:
    depth (Z-space), glass materials, glow light sources, motion curves, and
    an expanded color palette. Existing tokens like --bg-primary, --text-primary,
    --signal-positive must remain functional throughout the transition.
  </context>

  <token_spec>

    <category name="depth-z-space">
      --z-luminary: 0;
      --z-shell: 1;
      --z-void: 2;
      --z-glass: 3;
      --z-elevated: 4;
      --z-overlay: 5;
    </category>

    <category name="glass-materials">
      --glass-blur-near: blur(8px);
      --glass-blur-mid: blur(20px);
      --glass-blur-far: blur(36px);
      --glass-saturation: saturate(160%);
      --glass-tint-dark: rgba(255, 255, 255, 0.05);
      --glass-tint-light: rgba(255, 255, 255, 0.62);
      --glass-border-highlight: rgba(255, 255, 255, 0.14);
      --glass-border-shadow: rgba(0, 0, 0, 0.18);
      --glass-specular: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 50%);
    </category>

    <category name="glow-system">
      --glow-positive: #6BFF9E;
      --glow-negative: #FF6B6B;
      --glow-neutral: #6BB8FF;
      --glow-gold: #FFD87A;
      --glow-ice: #A8D8FF;
      --glow-amber: #FFB347;
      --glow-sky: #B8DAFF;
      --glow-intensity: 0.6;
      --glow-radius-near: 120px;
      --glow-radius-far: 400px;
      --glow-positive-rgb: 107, 255, 158;
      --glow-negative-rgb: 255, 107, 107;
      --glow-neutral-rgb: 107, 184, 255;
      --glow-gold-rgb: 255, 216, 122;
    </category>

    <category name="motion">
      --ease-breathe: cubic-bezier(0.34, 1.2, 0.64, 1);
      --ease-defuse: cubic-bezier(0.16, 1, 0.3, 1);
      --ease-sink: cubic-bezier(0.4, 0, 1, 1);
      --duration-diffuse: 600ms;
      --duration-breathe: 4000ms;
      --duration-float: 6000ms;
      --parallax-strength: 8px;
    </category>

    <category name="color-dark" selector=":root, [data-theme='dark']">
      --bg-primary: #080810;
      --bg-primary-rgb: 8, 8, 16;
      --bg-secondary: #0E0E1A;
      --bg-tertiary: #141426;
      --text-primary: #EEE8DC;
      --text-secondary: #9990A0;
      --text-tertiary: #5A5468;
      --border-primary: rgba(255, 255, 255, 0.07);
      --border-highlight: rgba(255, 255, 255, 0.14);
      --signal-positive: #4EE88A;
      --signal-negative: #E86060;
      --signal-neutral: #60A8E8;
      --text-display: clamp(3rem, 6vw, 5rem);
    </category>

    <category name="color-light" selector="[data-theme='light']">
      --bg-primary: #F5F0E8;
      --bg-primary-rgb: 245, 240, 232;
      --bg-secondary: #EDE8DF;
      --bg-tertiary: #E4DFD4;
      --text-primary: #18141E;
      --text-secondary: #5C5468;
      --text-tertiary: #998EA8;
      --border-primary: rgba(0, 0, 0, 0.07);
      --border-highlight: rgba(255, 255, 255, 0.80);
      --signal-positive: #0D7A3C;
      --signal-negative: #A01818;
      --signal-neutral: #1456A0;
    </category>

  </token_spec>

  <constraints>
    <item>Do not remove any existing CSS variables — only add new ones and alias where needed.</item>
    <item>Do not change any component styles in this prompt. Tokens only.</item>
    <item>Add tokens into the existing :root block. Do not create a second :root.</item>
    <item>All --glow-* tokens must have corresponding --*-rgb variants for rgba() composition.</item>
    <item>Preserve the existing ThemeManager in dashboard.js — it reads the data-theme attribute.</item>
  </constraints>

  <verification>
    <check>DevTools computed panel on :root shows all new --z-*, --glass-*, --glow-*, --ease-* tokens.</check>
    <check>Switching dark/light mode still works. Existing components unchanged visually.</check>
    <check>No CSS parse errors in DevTools console.</check>
  </verification>

  <output>Modified static/css/modern.css with new token blocks in :root and [data-theme="light"].</output>

</prompt>
```

---

## Phase B — Layer System

### Prompt B1 · DOM Structure + Shell Layer

```xml
<prompt id="B1" phase="B" title="Layer DOM Structure and Shell Frosted Background">

  <objective>
    Insert the two fixed background layers (Luminary placeholder and Shell) into the page DOM,
    and implement the Shell frosted material in CSS. Establishes the visual foundation
    all glass cards float above. Luminary orbs come in B2 — this prompt is DOM and Shell only.
  </objective>

  <prerequisite>Prompt A1 complete — all Breathe tokens exist in modern.css.</prerequisite>

  <read_first>
    <file>templates/base.html</file>
    <file>static/css/modern.css</file>
  </read_first>

  <dom_changes file="templates/base.html">
    <item>
      Insert as the FIRST two children of body, before any nav or content:

      &lt;svg class="svg-defs" aria-hidden="true"
           style="position:absolute;width:0;height:0;overflow:hidden"&gt;
        &lt;defs&gt;
          &lt;filter id="shell-refraction"&gt;
            &lt;feTurbulence type="fractalNoise" baseFrequency="0.65"
              numOctaves="3" seed="2" result="noise"/&gt;
            &lt;feDisplacementMap in="SourceGraphic" in2="noise"
              scale="8" xChannelSelector="R" yChannelSelector="G"/&gt;
          &lt;/filter&gt;
          &lt;filter id="card-bulge"&gt;
            &lt;feTurbulence type="turbulence" baseFrequency="0.05"
              numOctaves="2" seed="5" result="turb"&gt;
              &lt;animate attributeName="baseFrequency" dur="800ms"
                values="0.05;0.08;0.05" begin="indefinite" id="bulge-anim"/&gt;
            &lt;/feTurbulence&gt;
            &lt;feDisplacementMap in="SourceGraphic" in2="turb"
              scale="0" xChannelSelector="R" yChannelSelector="G" id="bulge-map"/&gt;
          &lt;/filter&gt;
        &lt;/defs&gt;
      &lt;/svg&gt;

      &lt;div class="luminary" aria-hidden="true"&gt;
        &lt;div class="luminary__orb luminary__orb--primary"&gt;&lt;/div&gt;
        &lt;div class="luminary__orb luminary__orb--secondary"&gt;&lt;/div&gt;
        &lt;div class="luminary__particles" id="luminary-particles"&gt;&lt;/div&gt;
      &lt;/div&gt;
      &lt;div class="shell" aria-hidden="true"&gt;&lt;/div&gt;
    </item>
    <item>
      On the &lt;html&gt; tag add attributes: data-depth="on" data-parallax="on"
    </item>
  </dom_changes>

  <css_spec file="static/css/modern.css">

    <section name="body — transparent so fixed layers show through">
      body { background: transparent; position: relative; z-index: var(--z-glass); }
    </section>

    <section name="luminary — container only, orbs styled in B2">
      .luminary {
        position: fixed; inset: 0;
        z-index: var(--z-luminary);
        pointer-events: none; overflow: hidden;
      }
      [data-depth="off"] .luminary { display: none; }
    </section>

    <section name="shell — frosted site background">
      .shell {
        position: fixed; inset: 0;
        z-index: var(--z-shell);
        pointer-events: none;
        background-color: rgba(var(--bg-primary-rgb), 0.88);
        backdrop-filter: blur(2px) saturate(120%);
        -webkit-backdrop-filter: blur(2px) saturate(120%);
      }
      .shell::before {
        content: ''; position: absolute; inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
        opacity: 0.03; pointer-events: none;
      }
      [data-depth="off"] .shell {
        backdrop-filter: none; -webkit-backdrop-filter: none;
        background-color: var(--bg-primary);
      }
      [data-depth="off"] .shell::before { display: none; }
    </section>

  </css_spec>

  <constraints>
    <item>luminary and shell must both have aria-hidden="true".</item>
    <item>body must be transparent — shell provides the background.</item>
    <item>data-depth="off" on html element cleanly removes all blur effects.</item>
    <item>Do not change any nav, card, or content HTML in this prompt.</item>
  </constraints>

  <verification>
    <check>Page background has a faint frosted texture — no longer a hard flat colour.</check>
    <check>Nav and all content still visible. No z-index conflicts.</check>
    <check>Adding data-depth="off" to html shows solid background colour, removes blur.</check>
  </verification>

  <output>
    Modified templates/base.html (SVG defs, luminary div, shell div, html attributes).
    Modified static/css/modern.css (body bg, luminary base, shell material, depth-off overrides).
  </output>

</prompt>
```

---

### Prompt B2 · Luminary — Orbs, Particles, Parallax

```xml
<prompt id="B2" phase="B" title="Luminary Light Orbs, Particles, and Mouse Parallax">

  <objective>
    Bring the Luminary layer to life: dual animated glow orbs, SVG particle field,
    and mouse-driven parallax. Creates the living light source that all upper layers
    are illuminated by. Warm gold (top-right) and cool ice-blue (bottom-left).
  </objective>

  <prerequisite>Prompt B1 complete — luminary DOM and shell layer exist.</prerequisite>

  <read_first>
    <file>static/css/modern.css</file>
    <file>static/js/dashboard.js</file>
  </read_first>

  <css_spec file="static/css/modern.css">

    <section name="Orbs">
      .luminary__orb { position: absolute; border-radius: 50%; pointer-events: none; will-change: transform; }

      .luminary__orb--primary {
        width: 800px; height: 800px; top: -200px; right: -100px;
        background: radial-gradient(ellipse at center,
          rgba(var(--glow-gold-rgb), calc(0.25 * var(--glow-intensity))) 0%, transparent 70%);
        animation: orb-drift-a var(--duration-float) ease-in-out infinite alternate;
      }
      .luminary__orb--secondary {
        width: 600px; height: 600px; bottom: -150px; left: -80px;
        background: radial-gradient(ellipse at center,
          rgba(var(--glow-neutral-rgb), calc(0.18 * var(--glow-intensity))) 0%, transparent 70%);
        animation: orb-drift-b calc(var(--duration-float) * 1.22) ease-in-out infinite alternate;
      }
      [data-theme="light"] .luminary__orb--primary {
        background: radial-gradient(ellipse at center,
          rgba(255,179,71, calc(0.15 * var(--glow-intensity))) 0%, transparent 70%);
      }
      [data-theme="light"] .luminary__orb--secondary {
        background: radial-gradient(ellipse at center,
          rgba(184,218,255, calc(0.10 * var(--glow-intensity))) 0%, transparent 70%);
      }
      @keyframes orb-drift-a { from { transform: translate(0,0); } to { transform: translate(-80px,60px); } }
      @keyframes orb-drift-b { from { transform: translate(0,0); } to { transform: translate(60px,-80px); } }
      @media (prefers-reduced-motion: reduce) { .luminary__orb { animation: none; } }
    </section>

    <section name="Particles">
      .luminary__particles { position: absolute; inset: 0; pointer-events: none; }
      .luminary__particle { position: absolute; border-radius: 50%; background: var(--text-primary); pointer-events: none; }
      @media (prefers-reduced-motion: reduce) { .luminary__particle { animation: none !important; } }
    </section>

  </css_spec>

  <js_spec file="static/js/dashboard.js">
    Add ParallaxManager class before the DOMContentLoaded listener.
    Then inside DOMContentLoaded: window.parallaxManager = new ParallaxManager();

    class ParallaxManager {
      constructor() {
        this.enabled = localStorage.getItem('stockholm-parallax') !== 'false'
          && !('ontouchstart' in window);
        this.orbA = document.querySelector('.luminary__orb--primary');
        this.orbB = document.querySelector('.luminary__orb--secondary');
        this.tx = 0; this.ty = 0; this.cx = 0; this.cy = 0;
        this.raf = null;
        if (this.enabled) this._listen();
        this._particles();
      }
      _listen() {
        document.addEventListener('mousemove', e => {
          this.tx = (e.clientX / window.innerWidth  - 0.5) * 80;
          this.ty = (e.clientY / window.innerHeight - 0.5) * 60;
        });
        this._tick();
      }
      _tick() {
        this.cx += (this.tx - this.cx) * 0.06;
        this.cy += (this.ty - this.cy) * 0.06;
        if (this.orbA) this.orbA.style.transform = `translate(${this.cx*.4}px,${this.cy*.3}px)`;
        if (this.orbB) this.orbB.style.transform = `translate(${-this.cx*.25}px,${-this.cy*.2}px)`;
        this.raf = requestAnimationFrame(() => this._tick());
      }
      setEnabled(val) {
        this.enabled = val;
        localStorage.setItem('stockholm-parallax', val);
        if (val && !this.raf) this._listen();
        else if (!val) {
          cancelAnimationFrame(this.raf); this.raf = null;
          if (this.orbA) this.orbA.style.transform = '';
          if (this.orbB) this.orbB.style.transform = '';
        }
      }
      _particles() {
        const c = document.getElementById('luminary-particles');
        if (!c || 'ontouchstart' in window) return;
        for (let i = 0; i < 40; i++) {
          const p = document.createElement('div');
          p.className = 'luminary__particle';
          const sz = 1 + Math.random() * 2;
          const dur = 8 + Math.random() * 12;
          const del = -Math.random() * 20;
          const dx = (Math.random()-.5)*60; const dy = (Math.random()-.5)*60;
          p.style.cssText = `width:${sz}px;height:${sz}px;left:${Math.random()*100}%;`
            + `top:${Math.random()*100}%;opacity:${0.06+Math.random()*0.09};`
            + `animation:pdrift${i} ${dur}s ${del}s ease-in-out infinite alternate;`;
          const s = document.createElement('style');
          s.textContent = `@keyframes pdrift${i}{from{transform:translate(0,0)}to{transform:translate(${dx}px,${dy}px)}}`;
          document.head.appendChild(s);
          c.appendChild(p);
        }
      }
    }
  </js_spec>

  <constraints>
    <item>prefers-reduced-motion: orb animations and particles stop, parallax disabled.</item>
    <item>Touch devices: parallax auto-disabled, particles skipped.</item>
    <item>ParallaxManager.setEnabled() is the public API called by Settings page.</item>
    <item>data-depth="off" hides entire luminary — no orphan orbs.</item>
  </constraints>

  <verification>
    <check>Dual warm-gold and cool-blue glows gently drift in dark mode.</check>
    <check>Mouse movement shifts orbs subtly — parallax depth is perceptible but not jarring.</check>
    <check>40 tiny particles float independently across the background.</check>
    <check>prefers-reduced-motion: all orb and particle motion stops completely.</check>
  </verification>

  <output>
    Modified static/css/modern.css (orb styles, particle styles, drift keyframes).
    Modified static/js/dashboard.js (ParallaxManager class).
  </output>

</prompt>
```

---

### Prompt B3 · Glass Cards — Material + The Bulb

```xml
<prompt id="B3" phase="B" title="Glassmorphic Card Material and The Bulb Glow">

  <objective>
    Transform every .card into a sharp-edged glassmorphic slab with specular highlight
    and a living signal-colored glow source (The Bulb) behind it.
    Sharp edges (border-radius: 0) are non-negotiable throughout.
  </objective>

  <prerequisite>B1 and B2 complete — Luminary and Shell are active.</prerequisite>

  <read_first>
    <file>static/css/modern.css</file>
    <file>templates/dashboard.html</file>
  </read_first>

  <css_spec file="static/css/modern.css">

    <section name="Glass card base">
      .card {
        position: relative;
        border-radius: 0;
        background: var(--glass-tint-dark);
        backdrop-filter: var(--glass-blur-mid) var(--glass-saturation);
        -webkit-backdrop-filter: var(--glass-blur-mid) var(--glass-saturation);
        border: none;
        box-shadow:
          inset 1px 1px 0 var(--glass-border-highlight),
          inset -1px -1px 0 var(--glass-border-shadow),
          0 8px 32px rgba(0,0,0,0.4),
          0 2px 8px rgba(0,0,0,0.2),
          0 16px 48px rgba(var(--card-glow-rgb, var(--glow-neutral-rgb)), calc(0.15 * var(--glow-intensity)));
        padding: var(--space-6);
        transition: transform var(--duration-fast) var(--ease-breathe),
                    box-shadow var(--duration-fast) var(--ease-breathe),
                    backdrop-filter var(--duration-fast) ease;
        --card-glow-rgb: var(--glow-neutral-rgb);
      }
      [data-theme="light"] .card { background: var(--glass-tint-light); }
    </section>

    <section name="Specular highlight — top-left corner shine">
      .card::before {
        content: ''; position: absolute; inset: 0;
        background: var(--glass-specular);
        pointer-events: none; z-index: 1; opacity: 0.8;
        transition: opacity var(--duration-fast) ease;
      }
    </section>

    <section name="The Bulb — glow behind each card">
      .card::after {
        content: ''; position: absolute; inset: -40px;
        background: radial-gradient(ellipse at center,
          rgba(var(--card-glow-rgb), calc(0.12 * var(--glow-intensity))) 0%, transparent 65%);
        z-index: -1; pointer-events: none;
        animation: breathing-glow var(--duration-breathe) ease-in-out infinite alternate;
        transition: background var(--duration-normal) ease;
      }
      @keyframes breathing-glow {
        from { opacity: 0.7; transform: scale(0.95); }
        to   { opacity: 1.0; transform: scale(1.05); }
      }
      @media (prefers-reduced-motion: reduce) { .card::after { animation: none; } }
    </section>

    <section name="Hover — lift, clear, brighten">
      .card:hover {
        transform: translateY(-3px);
        backdrop-filter: blur(12px) var(--glass-saturation);
        -webkit-backdrop-filter: blur(12px) var(--glass-saturation);
        box-shadow:
          inset 1px 1px 0 var(--glass-border-highlight),
          inset -1px -1px 0 var(--glass-border-shadow),
          0 16px 48px rgba(0,0,0,0.5),
          0 4px 12px rgba(0,0,0,0.3),
          0 24px 64px rgba(var(--card-glow-rgb, var(--glow-neutral-rgb)), calc(0.22 * var(--glow-intensity)));
      }
      .card:hover::before { opacity: 1; }
    </section>

    <section name="Signal colours via data-signal attribute">
      .card[data-signal="positive"] { --card-glow-rgb: var(--glow-positive-rgb); }
      .card[data-signal="negative"] { --card-glow-rgb: var(--glow-negative-rgb); }
      .card[data-signal="neutral"]  { --card-glow-rgb: var(--glow-neutral-rgb); }
      .card[data-signal="gold"]     { --card-glow-rgb: var(--glow-gold-rgb); }
    </section>

    <section name="Depth effects off fallback">
      [data-depth="off"] .card {
        backdrop-filter: none; -webkit-backdrop-filter: none;
        background: var(--bg-secondary);
        box-shadow: 0 1px 0 var(--border-primary);
      }
      [data-depth="off"] .card::before,
      [data-depth="off"] .card::after { display: none; }
    </section>

  </css_spec>

  <html_task>
    Add data-signal attributes to cards in dashboard.html, watchlist.html,
    portfolio.html, top_picks.html:
    - Positive return / bullish metric cards → data-signal="positive"
    - Loss / risk / bearish cards             → data-signal="negative"
    - Info / chart / neutral cards            → data-signal="neutral"
    - Watch / alert cards                     → data-signal="gold"
  </html_task>

  <constraints>
    <item>border-radius must be 0 on .card — no exceptions.</item>
    <item>The ::after Bulb must have z-index: -1 — it renders behind the card.</item>
    <item>Card inner content must have position: relative; z-index: 2 to sit above ::before specular.</item>
    <item>breathing-glow must respect prefers-reduced-motion.</item>
    <item>Existing card grid layout and padding must not break.</item>
  </constraints>

  <verification>
    <check>Cards blur the content visible behind them — glassmorphic confirmed.</check>
    <check>Top-left rim is faintly bright; bottom-right is darker — specular asymmetry.</check>
    <check>Positive cards have green glow behind them; negative red; neutral blue.</check>
    <check>Hovering lifts card 3px, brightens specular, deepens shadow.</check>
    <check>data-depth="off" shows flat opaque cards, no glow, no blur.</check>
  </verification>

  <output>
    Modified static/css/modern.css (glass card system, bulb, hover states).
    Modified templates/dashboard.html + watchlist.html + portfolio.html + top_picks.html
    (data-signal attributes on cards).
  </output>

</prompt>
```

---

## Phase C — Data Visualization

### Prompt C1 · Chart Theme + SparkBar + Delta Arrows

```xml
<prompt id="C1" phase="C" title="Chart.js Theme, SparkBar Micro-Charts, Delta Arrow Components">

  <objective>
    Override Chart.js global defaults with Breathe visual language. Implement the SparkBar
    micro-chart class for table cells. Replace text delta patterns (▲ +2.4%) with
    proportional SVG directional arrows. Data must be understood before it is read.
  </objective>

  <prerequisite>Phase A tokens active in CSS.</prerequisite>

  <read_first>
    <file>static/js/dashboard.js</file>
    <file>static/css/modern.css</file>
  </read_first>

  <js_spec file="static/js/dashboard.js">

    <section name="Chart.js global theme — call before any chart initialises">
      function applyChartTheme() {
        const get = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
        Chart.defaults.font.family = 'JetBrains Mono, monospace';
        Chart.defaults.font.size = 11;
        Chart.defaults.color = get('--text-secondary');
        Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
        Chart.defaults.plugins.legend.display = false;
        Chart.defaults.plugins.tooltip.backgroundColor = get('--bg-secondary');
        Chart.defaults.plugins.tooltip.borderColor = get('--border-highlight');
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.titleColor = get('--text-primary');
        Chart.defaults.plugins.tooltip.bodyColor = get('--text-secondary');
        Chart.defaults.plugins.tooltip.padding = 12;
        Chart.defaults.plugins.tooltip.cornerRadius = 0;
        Chart.defaults.elements.line.borderWidth = 1.5;
        Chart.defaults.elements.point.radius = 0;
        Chart.defaults.elements.point.hoverRadius = 4;
        Chart.defaults.elements.bar.borderRadius = 0;
        Chart.defaults.scale.grid.color = 'rgba(255,255,255,0.04)';
        Chart.defaults.scale.grid.drawBorder = false;
        Chart.defaults.scale.ticks.maxTicksLimit = 5;
      }
      document.addEventListener('themechange', applyChartTheme);
      // Call applyChartTheme() inside DOMContentLoaded before chart init.
    </section>

    <section name="SparkBar — 8-bar inline SVG for table cells">
      class SparkBar {
        constructor(container, data, signal = 'neutral') {
          this.c = container; this.data = data; this.signal = signal; this.render();
        }
        render() {
          const get = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
          const colors = { positive: get('--glow-positive'), negative: get('--glow-negative'), neutral: get('--glow-neutral') };
          const color = colors[this.signal] || colors.neutral;
          const vals = this.data.slice(-8);
          const max = Math.max(...vals.map(Math.abs)) || 1;
          const W = 40, H = 16, bw = 3, gap = 2;
          const ns = 'http://www.w3.org/2000/svg';
          const svg = document.createElementNS(ns, 'svg');
          svg.setAttribute('width', W); svg.setAttribute('height', H);
          svg.setAttribute('aria-hidden', 'true');
          svg.style.cssText = 'display:inline-block;vertical-align:middle';
          vals.forEach((v, i) => {
            const h = Math.max(2, (Math.abs(v) / max) * H);
            const r = document.createElementNS(ns, 'rect');
            r.setAttribute('x', i * (bw + gap));
            r.setAttribute('y', H - h);
            r.setAttribute('width', bw); r.setAttribute('height', h);
            r.setAttribute('fill', color);
            r.setAttribute('opacity', i === vals.length - 1 ? '1' : '0.5');
            svg.appendChild(r);
          });
          this.c.innerHTML = ''; this.c.appendChild(svg);
        }
      }
      window.SparkBar = SparkBar;
    </section>

    <section name="Delta arrows — proportional SVG for data-delta cells">
      function renderDeltaArrows() {
        document.querySelectorAll('[data-delta]').forEach(el => {
          const val = parseFloat(el.dataset.delta);
          if (isNaN(val)) return;
          const get = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
          const color = val >= 0 ? get('--signal-positive') : get('--signal-negative');
          const shaft = Math.min(24, Math.max(6, Math.abs(val) * 4));
          const up = val >= 0;
          const ns = 'http://www.w3.org/2000/svg';
          const svg = document.createElementNS(ns, 'svg');
          svg.setAttribute('width', '14'); svg.setAttribute('height', shaft + 10);
          svg.setAttribute('aria-hidden', 'true');
          svg.style.cssText = 'display:inline-block;vertical-align:middle;margin-right:4px';
          const line = document.createElementNS(ns, 'line');
          line.setAttribute('x1','7'); line.setAttribute('x2','7');
          line.setAttribute('y1', up ? shaft + 8 : 2);
          line.setAttribute('y2', up ? 8 : shaft + 2);
          line.setAttribute('stroke', color); line.setAttribute('stroke-width','2');
          const poly = document.createElementNS(ns, 'polygon');
          poly.setAttribute('fill', color);
          poly.setAttribute('points', up
            ? `7,0 1,9 13,9`
            : `7,${shaft+10} 1,${shaft+1} 13,${shaft+1}`);
          svg.appendChild(line); svg.appendChild(poly);
          const span = document.createElement('span');
          span.textContent = `${val>=0?'+':''}${val.toFixed(2)}%`;
          span.style.cssText = `font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:${color}`;
          el.innerHTML = ''; el.appendChild(svg); el.appendChild(span);
        });
      }
      // Call renderDeltaArrows() after any table data is rendered or updated.
    </section>

  </js_spec>

  <constraints>
    <item>applyChartTheme() must be called BEFORE any Chart instance is created.</item>
    <item>Tooltip cornerRadius must be 0 — sharp corners match the card language.</item>
    <item>SparkBar: SVG only, no canvas, no library.</item>
    <item>Delta shaft length is proportional to magnitude, capped at 24px.</item>
    <item>Add data-delta="VALUE" on table cells in templates rather than inline ▲▼ text.</item>
  </constraints>

  <verification>
    <check>Chart tooltips: sharp corners, dark glass background, JetBrains Mono font.</check>
    <check>Chart grid lines are ghost-level faint — structural but invisible.</check>
    <check>SparkBar in table cells: 8 bars, last bar full opacity, correct signal colour.</check>
    <check>Delta cells: SVG arrow + percentage, shaft visibly longer for larger values.</check>
  </verification>

  <output>
    Modified static/js/dashboard.js (applyChartTheme, SparkBar, renderDeltaArrows).
    Modified static/css/modern.css ([data-delta] inline-flex styles).
    Modified relevant templates (data-delta attributes on table cells).
  </output>

</prompt>
```

---

## Phase D — Loading Screen

### Prompt D1 · TextDiffuser — Mercury Effect on Page Text

```xml
<prompt id="D1" phase="D" title="Mercury Text Diffusion on Live Page Content">

  <objective>
    Implement the TextDiffuser class. The Mercury diffusion effect runs on the website's
    own text — nav labels, card headings, KPI values, table cells. There is no overlay.
    No separate loading screen sitting on top. The glassmorphic card structure and depth
    layers are fully visible from frame zero. Only the text within them diffuses from
    noise characters into real content. The layout IS the loading animation.
  </objective>

  <prerequisite>Phase B complete — glass cards visible.</prerequisite>

  <read_first>
    <file>static/js/dashboard.js</file>
    <file>templates/base.html</file>
    <file>static/css/modern.css</file>
  </read_first>

  <design_rules>
    <rule>Top of page (nav, h1, h2) resolves first: 0–400ms.</rule>
    <rule>Card titles and KPI labels resolve mid: 300–800ms.</rule>
    <rule>Table data and sub-labels resolve last: 600–1200ms.</rule>
    <rule>Digit characters (0–9) receive an additional 200ms delay within their node —
      numbers crystallise after surrounding label text, making the data reveal feel intentional.</rule>
    <rule>Unresolved characters shimmer — re-randomised every 2–3 frames.</rule>
    <rule>The glass card shapes, backgrounds, and glow are never affected — only text diffuses.</rule>
  </design_rules>

  <new_file path="static/js/loading-screen.js">
    (() => {
      const NOISE = '░▒▓█╔╗╝╚║═╠╣╦╩╬│─┼@#$%&?!;:~^*+=-';
      const rnd = () => NOISE[Math.floor(Math.random() * NOISE.length)];

      class TextDiffuser {
        constructor() { this.nodes = []; this.started = false; }

        wrap() {
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
            acceptNode(node) {
              const p = node.parentElement;
              if (!p) return NodeFilter.FILTER_REJECT;
              const tag = p.tagName.toLowerCase();
              if (['script','style','input','textarea','select','noscript','svg','pre'].includes(tag))
                return NodeFilter.FILTER_REJECT;
              if (p.closest('[aria-live], #loading-sword')) return NodeFilter.FILTER_REJECT;
              if (!node.textContent.trim()) return NodeFilter.FILTER_REJECT;
              return NodeFilter.FILTER_ACCEPT;
            }
          });
          let node;
          while ((node = walker.nextNode())) {
            const rect = node.parentElement.getBoundingClientRect();
            const yNorm = Math.min(1, (rect.top + window.scrollY) / Math.max(1, document.body.scrollHeight));
            const baseDelay = yNorm * 800;
            const text = node.textContent;
            const frag = document.createDocumentFragment();
            const chars = [];
            for (let i = 0; i < text.length; i++) {
              const ch = text[i];
              if (/\s/.test(ch)) { frag.appendChild(document.createTextNode(ch)); continue; }
              const span = document.createElement('span');
              span.className = 'diffuse-char';
              span.dataset.delay = /[0-9]/.test(ch) ? baseDelay + 200 : baseDelay;
              span.textContent = rnd();
              chars.push({ span, real: ch, resolved: false });
              frag.appendChild(span);
            }
            node.parentElement.replaceChild(frag, node);
            this.nodes.push(chars);
          }
        }

        start() {
          const t0 = performance.now();
          const skip = () => { this._resolveAll(); };
          document.addEventListener('keydown', skip, { once: true });
          document.addEventListener('click', skip, { once: true });
          const tick = () => {
            const elapsed = performance.now() - t0;
            let done = true;
            for (const chars of this.nodes) {
              for (const c of chars) {
                if (c.resolved) continue;
                done = false;
                const d = parseFloat(c.span.dataset.delay || 0);
                const local = elapsed - d;
                if (local < 0) { if (Math.random() < 0.3) c.span.textContent = rnd(); continue; }
                const p = Math.min(1, (local / 400) * 0.12 + 0.02);
                if (Math.random() < p) {
                  c.span.textContent = c.real;
                  c.span.classList.add('diffuse-resolved');
                  c.resolved = true;
                } else { c.span.textContent = rnd(); }
              }
            }
            if (done) this._finish();
            else requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }

        _resolveAll() {
          for (const chars of this.nodes)
            for (const c of chars)
              if (!c.resolved) { c.span.textContent = c.real; c.span.classList.add('diffuse-resolved'); c.resolved = true; }
          this._finish();
        }

        _finish() {
          document.dispatchEvent(new CustomEvent('textDiffuseComplete'));
          setTimeout(() => {
            document.querySelectorAll('.diffuse-char').forEach(s => s.replaceWith(document.createTextNode(s.textContent)));
          }, 300);
        }
      }

      document.addEventListener('DOMContentLoaded', () => {
        const enabled = localStorage.getItem('stockholm-loading-screen') !== 'false';
        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (!enabled || reduced) return;
        const d = new TextDiffuser();
        d.wrap();
        requestAnimationFrame(() => d.start());
        window._textDiffuser = d;
      });
    })();
  </new_file>

  <css_spec file="static/css/modern.css">
    .diffuse-char {
      color: var(--glow-gold); opacity: 0.4;
      transition: color 80ms ease, opacity 80ms ease;
      font-family: inherit; font-size: inherit;
    }
    .diffuse-char.diffuse-resolved { color: var(--text-primary); opacity: 1; }
    [data-theme="light"] .diffuse-char { color: var(--glow-amber); opacity: 0.5; }
    [data-theme="light"] .diffuse-char.diffuse-resolved { color: var(--text-primary); opacity: 1; }
  </css_spec>

  <html_task file="templates/base.html">
    Add in head, BEFORE dashboard.js:
    &lt;script src="/static/js/loading-screen.js" defer&gt;&lt;/script&gt;
  </html_task>

  <constraints>
    <item>wrap() called synchronously in DOMContentLoaded, before requestAnimationFrame.</item>
    <item>Never diffuse: script, style, input, textarea, select, svg, pre, [aria-live] content.</item>
    <item>After completion: all .diffuse-char spans unwrapped — DOM must be clean.</item>
    <item>Digits receive +200ms extra delay so numbers crystallise after surrounding labels.</item>
    <item>If loading screen disabled in localStorage: TextDiffuser must not initialise at all.</item>
    <item>prefers-reduced-motion: skip entirely — all text shows real values immediately.</item>
  </constraints>

  <verification>
    <check>On load: text briefly shows noise characters resolving top-to-bottom into real content.</check>
    <check>Glass card shapes visible from frame zero — only text inside diffuses.</check>
    <check>KPI numbers resolve after their surrounding labels.</check>
    <check>Click during diffusion: all remaining text resolves instantly.</check>
    <check>After completion: no .diffuse-char spans in DOM (verify in DevTools Elements).</check>
    <check>Loading screen disabled in settings: normal page load, no diffusion.</check>
  </verification>

  <output>
    New file: static/js/loading-screen.js (TextDiffuser).
    Modified static/css/modern.css (diffuse-char and diffuse-resolved styles).
    Modified templates/base.html (script tag before dashboard.js).
  </output>

</prompt>
```

---

### Prompt D2 · ASCII Sword Centerpiece

```xml
<prompt id="D2" phase="D" title="ASCII Sword — In-Page Centerpiece During Diffusion">

  <objective>
    Inject the ASCII sword as a content element inside the dashboard hero area —
    not a fixed overlay. It uses the same diffusion algorithm with blade-axis priority
    (center column resolves first). It dissolves when textDiffuseComplete fires.
    Surrounded by diffusing page text, not sitting above it.
  </objective>

  <prerequisite>Prompt D1 complete — TextDiffuser running.</prerequisite>

  <read_first>
    <file>templates/dashboard.html</file>
    <file>static/js/loading-screen.js</file>
    <file>static/css/modern.css</file>
  </read_first>

  <sword_target_art>
    Final resolved state (JetBrains Mono, each char = one span, blade axis = col 20):

                        |
                       /|\
                      / | \
                     /  |  \
                    / ==+== \
                   /    |    \
                  /     |     \
                 /      |      \
                /       |       \
               /________|________\
                        |
                        |
                        |
                        |
                  ======+======
                        |
                  +-----+-----+
                  |     |     |
                  +-----------+
  </sword_target_art>

  <js_spec file="static/js/loading-screen.js">
    Add SwordDiffuser class and wire it alongside TextDiffuser:

    class SwordDiffuser {
      constructor() {
        this.chars = [];
        this.el = null;
      }
      inject(heroContainer) {
        const lines = [/* sword art lines as array of strings */];
        const wrap = document.createElement('div');
        wrap.className = 'loading-sword-container';
        const pre = document.createElement('pre');
        pre.id = 'loading-sword'; pre.setAttribute('aria-hidden','true');
        pre.className = 'loading-sword';
        const BLADE_COL = 24;
        lines.forEach(line => {
          for (let col = 0; col < line.length; col++) {
            const ch = line[col];
            if (ch === ' ') { pre.appendChild(document.createTextNode(' ')); continue; }
            const span = document.createElement('span');
            span.className = 'diffuse-char sword-char';
            const dist = Math.abs(col - BLADE_COL) / Math.max(1, line.length / 2);
            span.dataset.dist = dist;
            span.textContent = NOISE[Math.floor(Math.random() * NOISE.length)];
            this.chars.push({ span, real: ch, resolved: false, dist });
            pre.appendChild(span);
          }
          pre.appendChild(document.createTextNode('\n'));
        });
        wrap.appendChild(pre);
        heroContainer.prepend(wrap);
        this.el = wrap;
      }
      tick() {
        let done = true;
        for (const c of this.chars) {
          if (c.resolved) continue;
          done = false;
          const p = (1 - c.dist) * 0.10 + 0.02;
          if (Math.random() < p) {
            c.span.textContent = c.real;
            c.span.classList.add('diffuse-resolved');
            c.resolved = true;
          } else { c.span.textContent = NOISE[Math.floor(Math.random() * NOISE.length)]; }
        }
        return done;
      }
      dissolve() {
        if (!this.el) return;
        this.el.style.transition = 'opacity 400ms var(--ease-defuse)';
        this.el.style.opacity = '0';
        setTimeout(() => this.el?.remove(), 420);
      }
    }

    // In the DOMContentLoaded boot, alongside TextDiffuser:
    // 1. Construct SwordDiffuser, call .inject(heroElement)
    // 2. Run sword.tick() inside the same rAF loop as TextDiffuser
    // 3. On textDiffuseComplete: call sword.dissolve()
    document.addEventListener('textDiffuseComplete', () => window._sword?.dissolve());
  </js_spec>

  <css_spec file="static/css/modern.css">
    .loading-sword-container {
      display: flex; justify-content: center;
      padding: var(--space-8) 0; pointer-events: none;
      position: relative;
    }
    .loading-sword-container::before {
      content: ''; position: absolute; width: 200px; height: 200px;
      background: radial-gradient(circle, rgba(var(--glow-gold-rgb),0.1) 0%, transparent 70%);
      z-index: -1; pointer-events: none;
    }
    [data-theme="light"] .loading-sword-container::before { display: none; }

    .loading-sword {
      font-family: 'JetBrains Mono', monospace;
      font-size: clamp(0.5rem, 1.2vw, 0.75rem);
      line-height: 1.3; color: var(--text-primary);
      background: none; border: none; padding: 0; margin: 0; white-space: pre;
    }
    .sword-char { color: var(--glow-gold); opacity: 0.5; }
  </css_spec>

  <constraints>
    <item>Sword is position: relative inside hero — NOT position: fixed and NOT z-index: overlay.</item>
    <item>aria-hidden="true" — purely decorative, screen readers skip it.</item>
    <item>Sword dissolves (fade + DOM removal) when textDiffuseComplete event fires.</item>
    <item>If loading screen disabled, SwordDiffuser never runs and no element is injected.</item>
  </constraints>

  <verification>
    <check>Sword appears inside the hero section on load — surrounded by diffusing text.</check>
    <check>Blade column resolves visibly before guard and pommel.</check>
    <check>Sword fully resolved at ~600ms — stands clear while surrounding text still diffuses.</check>
    <check>When page text finishes, sword fades and is removed from DOM cleanly.</check>
  </verification>

  <output>
    Modified static/js/loading-screen.js (SwordDiffuser class, rAF integration).
    Modified static/css/modern.css (sword container and pre styles).
    Modified templates/dashboard.html (loading-sword-container div in hero).
  </output>

</prompt>
```

---

## Phase E — Settings

### Prompt E1 · Appearance, Effects, and Deep Sleep Settings

```xml
<prompt id="E1" phase="E" title="Settings — Appearance Controls and Deep Sleep Mode">

  <objective>
    Add the Appearance & Effects settings section (4 client-side toggles) and the
    Server & Performance section with Deep Sleep Mode system (3-level intensity,
    configurable sleep window, weekend extension, live status indicator).
  </objective>

  <read_first>
    <file>templates/settings.html</file>
    <file>static/js/dashboard.js</file>
    <file>core/config.py</file>
    <file>scheduler.py</file>
    <file>app.py</file>
  </read_first>

  <appearance_controls>

    <control id="loading-screen" storage="localStorage:stockholm-loading-screen" default="true">
      Label: "Intro Loading Screen"
      Description: "Text diffusion and ASCII sword animation on page load."
      Effect: If false at DOMContentLoaded, TextDiffuser and SwordDiffuser do not initialise.
    </control>

    <control id="depth-effects" storage="localStorage:stockholm-depth-effects" default="true">
      Label: "Glass Depth Effects"
      Description: "Backdrop-blur, glassmorphism, and glow. Disable on slow devices."
      Effect: Sets/removes data-depth="off" on document.documentElement.
    </control>

    <control id="glow-intensity" type="range" min="0" max="100" step="5" default="60"
             storage="localStorage:stockholm-glow-intensity">
      Label: "Glow Intensity"
      Description: "Brightness of light sources behind panels."
      Effect: Live — document.documentElement.style.setProperty('--glow-intensity', val/100)
    </control>

    <control id="parallax" storage="localStorage:stockholm-parallax"
             default="true-desktop / false-touch">
      Label: "Parallax Depth"
      Description: "Light sources shift with cursor movement for 3D depth."
      Effect: Calls window.parallaxManager.setEnabled(bool)
    </control>

  </appearance_controls>

  <deep_sleep_spec>

    <backend_config file="core/config.py">
      Add to DEFAULT_SETTINGS:
        'deep_sleep_enabled': False,
        'deep_sleep_intensity': 'deep',
        'deep_sleep_start': '22:00',
        'deep_sleep_end': '07:00',
        'deep_sleep_full_weekends': False,
        'deep_sleep_min_checks': 1,
    </backend_config>

    <backend_scheduler file="scheduler.py">
      Add helper: is_deep_sleep_active(settings) -> bool
        Handle overnight range: start > end means sleep crosses midnight.
        If deep_sleep_full_weekends and today is Sat/Sun: always return True.

      Add helper: get_sleep_multiplier(settings) -> int
        'light': 2  |  'deep': 6  |  'hibernate': 999

      In each NON-CRITICAL job trigger — guard pattern:
        if is_deep_sleep_active(settings):
            if job_ran_within(interval * get_sleep_multiplier(settings)):
                return  # skip

      NEVER guard these jobs (exempt from deep sleep at all intensity levels):
        - price breach alerts
        - stop-loss guards
        - portfolio hard limit monitors
    </backend_scheduler>

    <backend_endpoint file="app.py">
      GET /api/server/sleep-status
      Response: { "sleeping": bool, "intensity": str, "resumes_at": str | null }
      resumes_at: next wake time as "HH:MM" string, null if not sleeping.
    </backend_endpoint>

    <frontend file="templates/settings.html">
      Section: "Server and Performance"

      Controls:
      1. Deep Sleep toggle (saves to DB via PATCH /api/settings on change)
      2. Intensity radio group: Light / Deep (default) / Hibernate
         — visible and enabled only when toggle is ON
      3. "Sleep from" time input (default 22:00) — saves on blur
      4. "Wake at" time input (default 07:00) — saves on blur
      5. "Full weekends" checkbox — saves on change

      Status display in section header:
        Fetch GET /api/server/sleep-status on settings page load.
        Active:    "● Active — next scan in N min"  (--signal-positive dot)
        Sleeping:  "◌ Deep Sleep — resumes at HH:MM" (--text-tertiary dot)
        Hibernate: "○ Hibernating — alerts only"    (--text-tertiary dot, empty)
        Font: JetBrains Mono for the status line.
    </frontend>

  </deep_sleep_spec>

  <constraints>
    <item>Appearance controls are localStorage only — no server API calls.</item>
    <item>Deep Sleep controls PATCH to existing /api/settings endpoint.</item>
    <item>Critical alert jobs are NEVER affected by deep sleep at any intensity.</item>
    <item>Overnight sleep window (start > end) must be handled correctly in is_deep_sleep_active.</item>
    <item>All 4 appearance settings must be read from localStorage and applied before first paint.</item>
  </constraints>

  <verification>
    <check>Loading Screen OFF: reload — text appears instantly, no diffusion.</check>
    <check>Depth Effects OFF: solid bg cards, luminary hidden, no backdrop-filter.</check>
    <check>Glow slider: moving it live visibly changes card glow intensity.</check>
    <check>Parallax OFF: mouse movement does not shift orbs.</check>
    <check>Deep Sleep ON + intensity Deep: non-critical jobs skip during sleep window.</check>
    <check>Sleep window overnight (22:00–07:00): correctly active at 23:00 and 03:00.</check>
    <check>Status indicator shows correct state after page load.</check>
    <check>All settings persist across browser restart.</check>
  </verification>

  <output>
    Modified templates/settings.html (Appearance section + Server/Performance section).
    Modified static/js/dashboard.js (apply localStorage settings on load).
    Modified core/config.py (deep sleep defaults).
    Modified scheduler.py (sleep guard helpers + job exemptions).
    Modified app.py (GET /api/server/sleep-status endpoint).
  </output>

</prompt>
```

---

## Phase F — Verification

### Prompt F1 · Full System QA

```xml
<prompt id="F1" phase="F" title="Full System Visual QA and Final Polish">

  <objective>
    Systematic visual and functional verification of the complete Breathe system.
    Read the key files, identify any gaps against the checklist, and fix them.
    This is a read-and-fix pass only — no new features.
  </objective>

  <prerequisite>All phases A–E complete.</prerequisite>

  <read_first>
    <file>static/css/modern.css</file>
    <file>static/js/dashboard.js</file>
    <file>static/js/loading-screen.js</file>
    <file>templates/base.html</file>
    <file>templates/settings.html</file>
  </read_first>

  <checklist>

    <group name="Layer System">
      <item>Dark: 3 depth layers distinct — void, frosted shell, floating glass cards.</item>
      <item>Light: frosted morning frost — cards have white-smoke glass over warm cream.</item>
      <item>Luminary: warm gold orb top-right, cool blue orb bottom-left, both breathing.</item>
      <item>Card Bulb: positive cards glow green, negative red, neutral blue.</item>
      <item>Card hover: lifts 3px, blur clears, bulb brightens. Transition is smooth.</item>
    </group>

    <group name="Text Diffusion">
      <item>On load: text diffuses top-to-bottom from noise into real content.</item>
      <item>Glass card shapes and backgrounds visible from frame zero.</item>
      <item>KPI numbers crystallise after surrounding label text.</item>
      <item>Sword materialises inside hero — blade first, then guard, then pommel.</item>
      <item>Sword dissolves cleanly when page text resolves. No DOM remnants after.</item>
      <item>Click or keypress: everything resolves instantly.</item>
      <item>Loading screen disabled in settings: clean page load, zero diffusion.</item>
    </group>

    <group name="Dark Mode — The Deep">
      <item>Background has ultraviolet depth — not pure black, subtle blue undertone.</item>
      <item>Glass cards: dark smoke tint, content visible through them.</item>
      <item>Text: warm off-white, readable through smoke glass.</item>
      <item>Signal colours: restrained, not neon.</item>
    </group>

    <group name="Light Mode — The Breath">
      <item>Background: warm Nordic cream (#F5F0E8), not pure white.</item>
      <item>Cards: white-smoke frost — visibly frosted, between transparent and opaque.</item>
      <item>Signal colours: forest green and deep burgundy — no neon.</item>
      <item>Shadows: warm sepia-tinted, not cold black.</item>
    </group>

    <group name="Settings">
      <item>All 4 appearance settings survive page reload.</item>
      <item>Glow slider value restored and applied to --glow-intensity on load.</item>
      <item>Depth OFF: data-depth="off" set before first paint — no flash of blur.</item>
      <item>Deep Sleep ON: scheduler correctly skips non-critical jobs in window.</item>
      <item>Critical jobs (price alerts, stop-loss) run at full frequency regardless.</item>
    </group>

    <group name="Accessibility and Performance">
      <item>prefers-reduced-motion: no diffusion, no parallax, no orb movement, no glow pulse.</item>
      <item>Focus states visible on all interactive elements — sharp 2px outline.</item>
      <item>data-depth="off": fully usable, no layout breaks from removed backdrop-filter.</item>
      <item>Mobile (375px): no horizontal overflow, no jank, parallax off, particles hidden.</item>
    </group>

  </checklist>

  <output>
    CSS/JS fixes for any failing checklist items.
    Each fix includes a comment citing the checklist item it addresses.
  </output>

</prompt>
```

---

*9 prompts across 6 phases: A1 → B1 → B2 → B3 → C1 → D1 → D2 → E1 → F1*
*Each prompt is self-contained and can be fed directly to an LLM.*

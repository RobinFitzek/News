// Liquid Glass — SVG Displacement Filter
// Adapted from Shu Ding's liquid-glass (https://github.com/shuding/liquid-glass)
// Generates an SVG feDisplacementMap that creates edge refraction on glass surfaces.

(function () {
  'use strict';

  const RESOLUTION = 200;
  const CORNER_RADIUS = 0.01; // near-zero for boxy Swedish design
  const EDGE_SOFTNESS = 0.12;
  const REFRACTION_STRENGTH = 0.7;

  function smoothStep(a, b, t) {
    t = Math.max(0, Math.min(1, (t - a) / (b - a)));
    return t * t * (3 - 2 * t);
  }

  function roundedRectSDF(x, y, w, h, r) {
    const qx = Math.abs(x) - w + r;
    const qy = Math.abs(y) - h + r;
    return (
      Math.min(Math.max(qx, qy), 0) +
      Math.sqrt(Math.max(qx, 0) ** 2 + Math.max(qy, 0) ** 2) -
      r
    );
  }

  function generateDisplacementMap() {
    const canvas = document.createElement('canvas');
    canvas.width = RESOLUTION;
    canvas.height = RESOLUTION;
    const ctx = canvas.getContext('2d');
    const data = new Uint8ClampedArray(RESOLUTION * RESOLUTION * 4);

    const rawValues = [];
    let maxScale = 0;

    for (let y = 0; y < RESOLUTION; y++) {
      for (let x = 0; x < RESOLUTION; x++) {
        const ux = x / RESOLUTION;
        const uy = y / RESOLUTION;
        const ix = ux - 0.5;
        const iy = uy - 0.5;

        const dist = roundedRectSDF(ix, iy, 0.42, 0.42, CORNER_RADIUS);
        const displacement = smoothStep(REFRACTION_STRENGTH, 0, dist - EDGE_SOFTNESS);
        const scaled = smoothStep(0, 1, displacement);

        const newX = ix * scaled + 0.5;
        const newY = iy * scaled + 0.5;
        const dx = newX - ux;
        const dy = newY - uy;

        maxScale = Math.max(maxScale, Math.abs(dx), Math.abs(dy));
        rawValues.push(dx, dy);
      }
    }

    maxScale = maxScale || 1;

    let idx = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = rawValues[idx++] / maxScale + 0.5;
      const g = rawValues[idx++] / maxScale + 0.5;
      data[i] = (r * 255) | 0;
      data[i + 1] = (g * 255) | 0;
      data[i + 2] = 128;
      data[i + 3] = 255;
    }

    ctx.putImageData(new ImageData(data, RESOLUTION, RESOLUTION), 0, 0);
    return { dataUrl: canvas.toDataURL(), scale: maxScale * RESOLUTION * 0.6 };
  }

  class LiquidBulgeManager {
    constructor(svgDefs) {
      this.svgDefs = svgDefs;
      this.nextId = 0;
      this.filterPool = new Map();
      this.initObservers();
    }

    initObservers() {
      const attachAll = () => {
        document.querySelectorAll('.card, .glass, .glass-card').forEach(card => this.attachToCard(card));
      };
      attachAll();
      
      const observer = new MutationObserver(mutations => {
        let shouldAttach = false;
        mutations.forEach(m => {
          m.addedNodes.forEach(node => {
            if (node.nodeType === 1 && (node.classList.contains('card') || node.classList.contains('glass') || node.classList.contains('glass-card'))) {
              shouldAttach = true;
            } else if (node.nodeType === 1 && node.querySelector) {
              if (node.querySelector('.card, .glass, .glass-card')) shouldAttach = true;
            }
          });
        });
        if (shouldAttach) attachAll();
      });
      observer.observe(document.body, { childList: true, subtree: true });
    }

    attachToCard(card) {
      if (this.filterPool.has(card)) return;

      const id = `card-bulge-${this.nextId++}`;
      const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
      filter.setAttribute('id', id);
      filter.setAttribute('filterUnits', 'objectBoundingBox');
      filter.setAttribute('x', '-10%');
      filter.setAttribute('y', '-10%');
      filter.setAttribute('width', '120%');
      filter.setAttribute('height', '120%');
      
      filter.innerHTML = `
        <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="1" result="noise" />
        <feDisplacementMap id="${id}-disp" in="SourceGraphic" in2="noise" scale="0" xChannelSelector="R" yChannelSelector="G" />
      `;

      this.svgDefs.appendChild(filter);
      const dispMap = filter.querySelector(`#${id}-disp`);

      this.filterPool.set(card, { filterId: id, dispMap });
      card.style.filter = `url(#${id})`;

      card.addEventListener('mouseenter', () => this.triggerBulge(dispMap));
    }

    triggerBulge(dispMap) {
      let start = null;
      const duration = 800;
      const maxScale = 6;

      function tick(timestamp) {
        if (!start) start = timestamp;
        const progress = (timestamp - start) / duration;

        if (progress >= 1) {
          dispMap.setAttribute('scale', '0');
          return;
        }

        const currentScale = Math.sin(progress * Math.PI) * maxScale;
        dispMap.setAttribute('scale', currentScale.toFixed(2));
        requestAnimationFrame(tick);
      }

      requestAnimationFrame(tick);
    }
  }

  function init() {
    const { dataUrl, scale } = generateDisplacementMap();

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '0');
    svg.setAttribute('height', '0');
    svg.setAttribute('aria-hidden', 'true');
    svg.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:-1;';

    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

    defs.innerHTML = `
      <filter id="liquid-glass" filterUnits="objectBoundingBox"
              color-interpolation-filters="sRGB"
              x="-2%" y="-2%" width="104%" height="104%">
        <feImage href="${dataUrl}"
                 preserveAspectRatio="none"
                 x="-2%" y="-2%" width="104%" height="104%"
                 result="dispMap" />
        <feDisplacementMap in="SourceGraphic" in2="dispMap"
                           xChannelSelector="R" yChannelSelector="G"
                           scale="${scale}" />
      </filter>
      <filter id="liquid-glass-nav" filterUnits="objectBoundingBox"
              color-interpolation-filters="sRGB"
              x="0%" y="-5%" width="100%" height="110%">
        <feImage href="${dataUrl}"
                 preserveAspectRatio="none"
                 x="0%" y="-5%" width="100%" height="110%"
                 result="dispMap" />
        <feDisplacementMap in="SourceGraphic" in2="dispMap"
                           xChannelSelector="R" yChannelSelector="G"
                           scale="${scale * 0.4}" />
      </filter>
      
      <filter id="shell-refraction">
        <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" result="noise" />
        <feDisplacementMap in="SourceGraphic" in2="noise" scale="8" xChannelSelector="R" yChannelSelector="G" />
      </filter>
    `;

    svg.appendChild(defs);
    document.body.appendChild(svg);

    // Skip heavy animations dynamically if reduced-motion is preferred
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!prefersReducedMotion) {
      // Disabled per user request - stops UI "bending" effect
      // new LiquidBulgeManager(defs);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

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

  function init() {
    const { dataUrl, scale } = generateDisplacementMap();

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '0');
    svg.setAttribute('height', '0');
    svg.setAttribute('aria-hidden', 'true');
    svg.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:-1;';

    svg.innerHTML = `
      <defs>
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
      </defs>
    `;

    document.body.appendChild(svg);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

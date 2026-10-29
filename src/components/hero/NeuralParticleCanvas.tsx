import React, { useEffect, useRef, useState } from 'react';

export interface MorphStage {
  id: number;
  label: string;
  sublabel: string;
}

export const MORPH_STAGES: MorphStage[] = [
  {
    id: 0,
    label: 'Neural Cortex',
    sublabel: 'First-Principles ML & PyTorch',
  },
  {
    id: 1,
    label: 'Attention Lattice',
    sublabel: 'RAG, Transformers & Backends',
  },
  {
    id: 2,
    label: 'Live Products',
    sublabel: 'Web, iOS & Android in Production',
  },
];

interface Particle {
  // Current 3D position
  x: number;
  y: number;
  z: number;
  // Velocity for spring + scatter physics
  vx: number;
  vy: number;
  vz: number;
  // Target 3D coordinates for each of the 3 formations
  tx: [number, number, number];
  ty: [number, number, number];
  tz: [number, number, number];
  // Visual traits
  baseSize: number;
  shade: string;
  alpha: number;
  isSquare: boolean;
  isHollow: boolean;
  // Space-dust / ascending plume behavior (like the reference image)
  isPlume: boolean;
  plumeLife: number;
  plumeSpeed: number;
  plumeDriftFreq: number;
  plumePhase: number;
  anchorIndex: number;
}

interface HubNode {
  tx: [number, number, number];
  ty: [number, number, number];
  tz: [number, number, number];
  x: number;
  y: number;
  z: number;
  sx: number;
  sy: number;
  sz: number;
}

interface SynapticPulse {
  fromHub: number;
  toHub: number;
  progress: number;
  speed: number;
}

const INK_PALETTE = [
  '#0a0a0a',
  '#141414',
  '#1f1f1f',
  '#2e2e2e',
  '#404040',
  '#525252',
  '#6b6b6b',
  '#858585',
  '#a3a3a3',
  '#57534e', // subtle warm graphite speck
];

// Deterministic pseudo-random helper for consistent sculpture geometry
function seededRandom(seed: number): () => number {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

interface NeuralParticleCanvasProps {
  /** 0 = Hero right column, 1 = Docked in sticky left column */
  scrollProgress?: number;
  /** Optional chapter-driven stage hint when scrolling through sections */
  chapterStageHint?: number;
  /** Custom height class override */
  heightClass?: string;
}

export const NeuralParticleCanvas: React.FC<NeuralParticleCanvasProps> = ({
  scrollProgress = 0,
  chapterStageHint,
  heightClass = 'h-[400px] sm:h-[460px] lg:h-[500px]',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [activeStage, setActiveStage] = useState<number>(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState<boolean>(true);
  const stageRef = useRef<number>(0);
  const scrollProgressRef = useRef<number>(scrollProgress);
  const prevScrollProgressRef = useRef<number>(scrollProgress);

  useEffect(() => {
    scrollProgressRef.current = scrollProgress;
  }, [scrollProgress]);

  useEffect(() => {
    if (typeof chapterStageHint === 'number' && scrollProgress > 0.85) {
      const normalized = chapterStageHint % MORPH_STAGES.length;
      if (stageRef.current !== normalized) {
        stageRef.current = normalized;
        setActiveStage(normalized);
      }
    }
  }, [chapterStageHint, scrollProgress]);

  // Auto-cycle between formations unless user manually paused it recently
  useEffect(() => {
    if (!isAutoPlaying) {
      const resumeTimer = setTimeout(() => setIsAutoPlaying(true), 14000);
      return () => clearTimeout(resumeTimer);
    }

    const interval = setInterval(() => {
      const next = (stageRef.current + 1) % MORPH_STAGES.length;
      stageRef.current = next;
      setActiveStage(next);
    }, 6200);

    return () => clearInterval(interval);
  }, [isAutoPlaying]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    const rand = seededRandom(421337);

    const isMobile = window.innerWidth < 768;
    const CORE_COUNT = isMobile ? 650 : 1350;
    const PLUME_COUNT = isMobile ? 120 : 260;
    const TOTAL_PARTICLES = CORE_COUNT + PLUME_COUNT;
    const HUB_COUNT = isMobile ? 26 : 44;

    const generateNeuralTarget = (i: number, total: number): [number, number, number] => {
      const r1 = rand();
      const r2 = rand();
      const r3 = rand();
      const section = i / total;

      const randomDir = (): [number, number, number] => {
        const u = r1 * Math.PI * 2;
        const v = Math.acos(2 * r2 - 1);
        return [Math.sin(v) * Math.cos(u), Math.sin(v) * Math.sin(u), Math.cos(v)];
      };

      // Brainstem: tapered tube angled down and slightly backward
      if (section > 0.9) {
        const t = (section - 0.9) / 0.1;
        const angle = r1 * Math.PI * 2;
        const radius = 0.1 * (1 - t * 0.35) * (0.5 + 0.5 * Math.sqrt(r2));
        return [
          -0.06 - t * 0.1 + Math.cos(angle) * radius,
          0.2 + t * 0.55,
          Math.sin(angle) * radius,
        ];
      }

      // Cerebellum: small folded ellipsoid tucked under the posterior cerebrum
      if (section > 0.78) {
        const [nx, ny, nz] = randomDir();
        const shell = 0.8 + 0.2 * Math.pow(r3, 0.3);
        const folia = 1 + 0.07 * Math.sin(ny * 26 + nx * 3);
        return [
          -0.4 + nx * 0.25 * shell * folia,
          0.3 + ny * 0.15 * shell * folia,
          nz * 0.24 * shell * folia,
        ];
      }

      // Cerebrum
      const [nx, ny, nz] = randomDir();
      // Mostly surface points so the folded cortex reads clearly
      const shell = r3 < 0.82 ? 0.97 + 0.03 * r3 : 0.6 + 0.4 * (r3 - 0.82) / 0.18;

      // Winding gyri: ridged meandering stripes
      const meander = Math.sin(nx * 9 + Math.sin(ny * 7 + nz * 5) * 1.8) *
        Math.cos(ny * 8 - nz * 6 + Math.sin(nx * 5) * 1.4);
      const ridge = 1 - Math.abs(meander); // crests near 1, sulci near 0
      const gyri = 1 + 0.085 * ridge - 0.04 + 0.025 * Math.sin(nx * 22 + nz * 17);

      let x = nx * 0.64 * shell * gyri;
      let y = ny * (ny > 0 ? 0.36 : 0.44) * shell * gyri - 0.1;
      let z = nz * 0.4 * shell * gyri;

      // Temporal lobe bulge low and forward, frontal lobe tapering at top
      if (ny > 0.1 && nx > -0.2) {
        y += 0.07 * Math.sin(Math.min(1, (nx + 0.2) * 1.6) * Math.PI) * ny;
      }
      // Sylvian fissure: a groove separating temporal from frontal/parietal lobes
      const sylvian = Math.exp(-Math.pow((ny - 0.18 + nx * 0.1) * 9, 2)) * (nx > -0.3 ? 1 : 0);
      x *= 1 - 0.06 * sylvian;
      z *= 1 - 0.1 * sylvian;

      // Longitudinal fissure along the top midline
      if (ny < -0.5 && Math.abs(nz) < 0.12) {
        y += 0.03;
      }
      return [x, y, z];
    };

    const generateLatticeTarget = (i: number, total: number): [number, number, number] => {
      // 3D Multi-Head Attention / Distributed System Architecture Stack
      const r1 = rand();
      const r2 = rand();
      const r3 = rand();

      // 4 horizontal architectural layers + vertical data buses + central core
      const section = i / total;

      if (section < 0.68) {
        // 4 Stacked Tensor / Microservice Planes
        const layerIdx = i % 4;
        const layerY = -0.54 + layerIdx * 0.36; // 4 crisp levels
        const gridSize = 11;
        const cellIndex = Math.floor(i / 4) % (gridSize * gridSize);
        const gx = (cellIndex % gridSize) / (gridSize - 1) - 0.5;
        const gz = Math.floor(cellIndex / gridSize) / (gridSize - 1) - 0.5;

        // Slight jitter around grid nodes for organic ink-stippled feel
        const jitterX = (r1 - 0.5) * 0.045;
        const jitterY = (r2 - 0.5) * 0.025;
        const jitterZ = (r3 - 0.5) * 0.045;

        const scale = layerIdx === 1 || layerIdx === 2 ? 1.18 : 0.92;
        return [
          gx * 1.1 * scale + jitterX,
          layerY + jitterY,
          gz * 1.1 * scale + jitterZ,
        ];
      } else if (section < 0.86) {
        // Vertical residual / attention routing pillars connecting the 4 layers
        const pillarIdx = i % 16;
        const px = ((pillarIdx % 4) / 3 - 0.5) * 0.95;
        const pz = (Math.floor(pillarIdx / 4) / 3 - 0.5) * 0.95;
        const py = (r1 - 0.5) * 1.15;
        return [px + (r2 - 0.5) * 0.03, py, pz + (r3 - 0.5) * 0.03];
      } else {
        // Central high-density embedding / vector core sphere
        const u = r1 * Math.PI * 2;
        const v = Math.acos(2 * r2 - 1);
        const rad = 0.26 * Math.cbrt(r3);
        return [
           rad * Math.sin(v) * Math.cos(u),
          rad * Math.sin(v) * Math.sin(u),
          rad * Math.cos(v),
        ];
      }
    };

    const generateProductTarget = (i: number, total: number): [number, number, number] => {
      // 3D Isometric Shipped Product Ecosystem:
      // Central sleek mobile device + floating UI cards + orbital backend ring
      const r1 = rand();
      const r2 = rand();
      const r3 = rand();
      const section = i / total;

      if (section < 0.48) {
        // Central Mobile Device Frame & Screen UI
        const w = 0.46;
        const h = 0.86;
        const sub = i % 10;

        if (sub < 4) {
          // Crisp outer chassis bezel & rounded perimeter
          const perim = r1 * 2 * (w * 2 + h * 2);
          let px = 0;
          let py = 0;
          if (perim < w * 2) {
            px = -w + perim;
            py = -h;
          } else if (perim < w * 2 + h * 2) {
            px = w;
            py = -h + (perim - w * 2);
          } else if (perim < w * 4 + h * 2) {
            px = w - (perim - (w * 2 + h * 2));
            py = h;
          } else {
            px = -w;
            py = h - (perim - (w * 4 + h * 2));
          }
          const pz = (r2 - 0.5) * 0.07;
          return [px + (r3 - 0.5) * 0.02, py + (r3 - 0.5) * 0.02, pz];
        } else {
          // Interactive UI cards & components floating on the mobile screen
          const cardIdx = i % 3;
          const cardYCenter = -0.42 + cardIdx * 0.42;
          const px = (r1 - 0.5) * (w * 1.55);
          const py = cardYCenter + (r2 - 0.5) * 0.28;
          const pz = 0.06 + (cardIdx === 1 ? 0.06 : 0.02) + (r3 - 0.5) * 0.02;
          return [px, py, pz];
        }
      } else if (section < 0.78) {
        // Flanking Web / SaaS Dashboard Cards (Left & Right in 3D space)
        const side = i % 2 === 0 ? -1 : 1;
        const cardW = 0.44;
        const cardH = 0.34;
        const cx = side * 0.72 + (r1 - 0.5) * cardW;
        const cy = (side === -1 ? -0.16 : 0.14) + (r2 - 0.5) * cardH;
        const cz = -0.12 + side * (r1 - 0.5) * 0.15 + (r3 - 0.5) * 0.03;
        return [cx, cy, cz];
      } else {
        // Orbital Cloud / Production Deployment Ring & API Nodes
        const angle = r1 * Math.PI * 2;
        const ringRadius = 0.88 + (r2 - 0.5) * 0.08;
        const rx = Math.cos(angle) * ringRadius;
        const rz = Math.sin(angle) * ringRadius * 0.72;
        // Tilt the orbital ring diagonally
        const ry = Math.sin(angle) * 0.32 + (r3 - 0.5) * 0.04;
        return [rx, ry, rz];
      }
    };

    // Build Core & Plume Particles
    const particles: Particle[] = [];
    for (let i = 0; i < TOTAL_PARTICLES; i++) {
      const isPlume = i >= CORE_COUNT;
      const anchorIndex = isPlume ? Math.floor(rand() * (CORE_COUNT * 0.5)) : i;

      const [nx, ny, nz] = generateNeuralTarget(anchorIndex, CORE_COUNT);
      const [lx, ly, lz] = generateLatticeTarget(anchorIndex, CORE_COUNT);
      const [px, py, pz] = generateProductTarget(anchorIndex, CORE_COUNT);

      // Bias neural target slightly higher for plume anchors so they rise from the top half
      const plumeYOffset = isPlume ? -Math.abs(ny) * 0.45 - 0.2 : 0;

      const sizeRoll = rand();
      const baseSize =
        sizeRoll > 0.94
          ? 2.4 + rand() * 1.1
          : sizeRoll > 0.65
          ? 1.45 + rand() * 0.75
          : 0.85 + rand() * 0.55;

      const shade = INK_PALETTE[Math.floor(rand() * INK_PALETTE.length)];

      particles.push({
        x: nx + (rand() - 0.5) * 0.3,
        y: ny + plumeYOffset + (rand() - 0.5) * 0.3,
        z: nz + (rand() - 0.5) * 0.3,
        vx: 0,
        vy: 0,
        vz: 0,
        tx: [nx, lx, px],
        ty: [ny + plumeYOffset, ly + plumeYOffset, py + plumeYOffset],
        tz: [nz, lz, pz],
        baseSize,
        shade,
        alpha: isPlume ? 0.25 + rand() * 0.55 : 0.45 + rand() * 0.52,
        isSquare: rand() > 0.82, // 18% crisp square data bits
        isHollow: rand() > 0.92, // 8% hollow ring nodes
        isPlume,
        plumeLife: rand(),
        plumeSpeed: 0.0022 + rand() * 0.0038,
        plumeDriftFreq: 1.5 + rand() * 3.0,
        plumePhase: rand() * Math.PI * 2,
        anchorIndex,
      });
    }

    // Build 30 Structural Hub Nodes for Synaptic Filaments & Firing Pulses
    const hubs: HubNode[] = [];
    for (let h = 0; h < HUB_COUNT; h++) {
      const sampleIdx = Math.floor((h / HUB_COUNT) * CORE_COUNT * 0.9);
      const p = particles[sampleIdx];
      hubs.push({
        tx: [...p.tx] as [number, number, number],
        ty: [...p.ty] as [number, number, number],
        tz: [...p.tz] as [number, number, number],
        x: p.x,
        y: p.y,
        z: p.z,
        sx: 0,
        sy: 0,
        sz: 0,
      });
    }

    const pulses: SynapticPulse[] = [];
    const spawnPulse = () => {
      if (pulses.length > 14) return;
      const fromHub = Math.floor(Math.random() * HUB_COUNT);
      let toHub = (fromHub + 1 + Math.floor(Math.random() * 5)) % HUB_COUNT;
      if (toHub === fromHub) toHub = (fromHub + 1) % HUB_COUNT;
      pulses.push({
        fromHub,
        toHub,
        progress: 0,
        speed: 0.016 + Math.random() * 0.022,
      });
    };

    for (let k = 0; k < 8; k++) spawnPulse();

    // -------------------------------------------------------------------------
    // 2. Interaction State (Mouse Panning Parallax + Repulsion Scatter)
    // -------------------------------------------------------------------------
    const pointer = {
      normX: 0, // -1 to 1 across window
      normY: 0,
      canvasX: -9999,
      canvasY: -9999,
      isInsideCanvas: false,
      smoothRotY: -0.22,
      smoothRotX: 0.12,
    };

    const handleWindowMouseMove = (e: MouseEvent) => {
      pointer.normX = (e.clientX / window.innerWidth - 0.5) * 2;
      pointer.normY = (e.clientY / window.innerHeight - 0.5) * 2;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
        pointer.canvasX = x;
        pointer.canvasY = y;
        pointer.isInsideCanvas = true;
      } else {
        pointer.isInsideCanvas = false;
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 0) return;
      const touch = e.touches[0];
      pointer.normX = (touch.clientX / window.innerWidth - 0.5) * 2;
      pointer.normY = (touch.clientY / window.innerHeight - 0.5) * 2;
      const rect = canvas.getBoundingClientRect();
      pointer.canvasX = touch.clientX - rect.left;
      pointer.canvasY = touch.clientY - rect.top;
      pointer.isInsideCanvas = true;
    };

    const handleTouchEnd = () => {
      pointer.isInsideCanvas = false;
    };

    const handleCanvasClick = () => {
      // Radial impulse burst + cycle to next stage
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.vx += (Math.random() - 0.5) * 0.06;
        p.vy += (Math.random() - 0.5) * 0.06;
        p.vz += (Math.random() - 0.5) * 0.06;
      }
      const next = (stageRef.current + 1) % MORPH_STAGES.length;
      stageRef.current = next;
      setActiveStage(next);
      setIsAutoPlaying(false);
    };

    window.addEventListener('mousemove', handleWindowMouseMove, { passive: true });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: true });
    canvas.addEventListener('touchend', handleTouchEnd, { passive: true });
    canvas.addEventListener('click', handleCanvasClick);

    // -------------------------------------------------------------------------
    // 3. Resize Observer for Crisp Retina Rendering
    // -------------------------------------------------------------------------
    let width = 500;
    let height = 520;
    let dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 1.75);

    const updateSize = () => {
      if (!container) return;
      const rect = container.getBoundingClientRect();
      width = Math.max(300, rect.width);
      height = Math.max(360, rect.height);
      dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 1.75);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };

    updateSize();
    const resizeObserver = new ResizeObserver(updateSize);
    resizeObserver.observe(container);

    // -------------------------------------------------------------------------
    // 4. Main 60fps Animation Loop with Visibility & Intersection Optimization
    // -------------------------------------------------------------------------
    let animationFrameId = 0;
    let time = 0;
    let isVisible = true;

    const render = () => {
      if (!isVisible) {
        animationFrameId = 0;
        return;
      }
      time += 0.015;
      const stg = stageRef.current;
      const curScroll = scrollProgressRef.current;
      const shiftDelta = curScroll - prevScrollProgressRef.current;
      prevScrollProgressRef.current = curScroll;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      const centerX = width * 0.5;
      const centerY = height * 0.52;
      const scale = Math.min(width, height) * 0.37;

      // Smooth camera rotation combining idle orbit + scroll-transition spin + cursor panning parallax
      const targetRotY =
        -0.28 +
        curScroll * 0.52 +
        Math.sin(time * 0.45) * 0.16 +
        pointer.normX * 0.52;
      const targetRotX = 0.14 + Math.cos(time * 0.35) * 0.07 + pointer.normY * 0.34;
      pointer.smoothRotY += (targetRotY - pointer.smoothRotY) * 0.065;
      pointer.smoothRotX += (targetRotX - pointer.smoothRotX) * 0.065;

      const cosY = Math.cos(pointer.smoothRotY);
      const sinY = Math.sin(pointer.smoothRotY);
      const cosX = Math.cos(pointer.smoothRotX);
      const sinX = Math.sin(pointer.smoothRotX);

      // Subtle floating levitation bob
      const floatBobY = Math.sin(time * 1.4) * 0.032;

      // -----------------------------------------------------------------------
      // A. Grounded Soft Shadow Ellipse (Like the Reference Screenshot)
      // -----------------------------------------------------------------------
      const shadowY = height * 0.89;
      const shadowRadiusX = scale * (0.68 - Math.sin(time * 1.4) * 0.03);
      const shadowRadiusY = shadowRadiusX * 0.11;
      const shadowGrad = ctx.createRadialGradient(
        centerX + pointer.normX * 14,
        shadowY,
        shadowRadiusX * 0.05,
        centerX + pointer.normX * 14,
        shadowY,
        shadowRadiusX
      );
      shadowGrad.addColorStop(0, 'rgba(23, 23, 23, 0.16)');
      shadowGrad.addColorStop(0.55, 'rgba(23, 23, 23, 0.06)');
      shadowGrad.addColorStop(1, 'rgba(23, 23, 23, 0)');

      ctx.fillStyle = shadowGrad;
      ctx.beginPath();
      ctx.ellipse(
        centerX + pointer.normX * 14,
        shadowY,
        shadowRadiusX,
        shadowRadiusY,
        0,
        0,
        Math.PI * 2
      );
      ctx.fill();

      // -----------------------------------------------------------------------
      // B. Update & Project Structural Hub Nodes + Synaptic Axons
      // -----------------------------------------------------------------------
      for (let h = 0; h < HUB_COUNT; h++) {
        const hub = hubs[h];
        hub.x += (hub.tx[stg] - hub.x) * 0.05;
        hub.y += (hub.ty[stg] + floatBobY - hub.y) * 0.05;
        hub.z += (hub.tz[stg] - hub.z) * 0.05;

        // 3D rotation
        const rx = hub.x * cosY - hub.z * sinY;
        const rz1 = hub.x * sinY + hub.z * cosY;
        const ry = hub.y * cosX - rz1 * sinX;
        const rz = hub.y * sinX + rz1 * cosX;

        const fov = 2.8 / (2.8 + rz);
        hub.sx = centerX + rx * scale * fov;
        hub.sy = centerY + ry * scale * fov;
        hub.sz = rz;
      }

      // Draw ultra-fine monochrome synaptic filaments between proximate hubs
      ctx.lineWidth = 0.7;
      for (let i = 0; i < HUB_COUNT; i++) {
        const h1 = hubs[i];
        for (let j = i + 1; j < HUB_COUNT; j++) {
          const h2 = hubs[j];
          const dx = h1.sx - h2.sx;
          const dy = h1.sy - h2.sy;
          const distSq = dx * dx + dy * dy;
          const maxDist = scale * 0.8;
          if (distSq < maxDist * maxDist) {
            const dist = Math.sqrt(distSq);
            const lineAlpha = (1 - dist / maxDist) * 0.3;
            ctx.strokeStyle = `rgba(38, 38, 38, ${lineAlpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(h1.sx, h1.sy);
            ctx.lineTo(h2.sx, h2.sy);
            ctx.stroke();
          }
        }

        // If cursor is inside canvas, connect nearby hubs to cursor like an active probe
        if (pointer.isInsideCanvas) {
          const cdx = h1.sx - pointer.canvasX;
          const cdy = h1.sy - pointer.canvasY;
          const cDist = Math.sqrt(cdx * cdx + cdy * cdy);
          if (cDist < 135) {
            const cAlpha = (1 - cDist / 135) * 0.28;
            ctx.strokeStyle = `rgba(23, 23, 23, ${cAlpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(pointer.canvasX, pointer.canvasY);
            ctx.lineTo(h1.sx, h1.sy);
            ctx.stroke();
          }
        }
      }

      // Neuron node markers at each hub
      for (let h = 0; h < HUB_COUNT; h++) {
        const hub = hubs[h];
        const a = Math.max(0.25, Math.min(0.85, 0.85 - hub.sz * 0.4));
        ctx.fillStyle = `rgba(15, 15, 15, ${a})`;
        ctx.beginPath();
        ctx.arc(hub.sx, hub.sy, 2.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = `rgba(15, 15, 15, ${a * 0.45})`;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.arc(hub.sx, hub.sy, 5, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Update & Draw Firing Synaptic Pulses along Hub Filaments
      if (Math.random() < 0.14) spawnPulse();
      for (let pIdx = pulses.length - 1; pIdx >= 0; pIdx--) {
        const pulse = pulses[pIdx];
        pulse.progress += pulse.speed;
        if (pulse.progress >= 1) {
          pulses.splice(pIdx, 1);
          continue;
        }
        const h1 = hubs[pulse.fromHub];
        const h2 = hubs[pulse.toHub];
        const px = h1.sx + (h2.sx - h1.sx) * pulse.progress;
        const py = h1.sy + (h2.sy - h1.sy) * pulse.progress;

        // Crisp high-contrast action-potential pulse dot
        ctx.fillStyle = 'rgba(15, 15, 15, 0.78)';
        ctx.beginPath();
        ctx.arc(px, py, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      // -----------------------------------------------------------------------
      // C. Update & Render Stippled Particle Cloud + Ascending Space Dust
      // -----------------------------------------------------------------------
      const scatterRadius = 92;
      const scatterRadiusSq = scatterRadius * scatterRadius;

      for (let i = 0; i < TOTAL_PARTICLES; i++) {
        const p = particles[i];

        if (p.isPlume) {
          // Ascending "bits & pieces" space dust that detaches and drifts upward
          p.plumeLife += p.plumeSpeed;
          if (p.plumeLife >= 1) {
            p.plumeLife = 0;
            p.x = p.tx[stg] + (Math.random() - 0.5) * 0.25;
            p.y = p.ty[stg] * 0.5;
            p.z = p.tz[stg] + (Math.random() - 0.5) * 0.25;
            p.vx = 0;
            p.vy = 0;
            p.vz = 0;
          }

          // Drift upward into space with gentle lateral curl
          const riseHeight = p.plumeLife * 1.18;
          const curlX =
            Math.sin(time * p.plumeDriftFreq + p.plumePhase + p.plumeLife * 4) *
            (0.08 + p.plumeLife * 0.24);
          const curlZ =
            Math.cos(time * p.plumeDriftFreq + p.plumePhase) *
            (0.08 + p.plumeLife * 0.22);

          const desiredX = p.tx[stg] * (1 - p.plumeLife * 0.35) + curlX;
          const desiredY = p.ty[stg] - riseHeight + floatBobY;
          const desiredZ = p.tz[stg] * (1 - p.plumeLife * 0.35) + curlZ;

          p.vx += (desiredX - p.x) * 0.04;
          p.vy += (desiredY - p.y) * 0.04;
          p.vz += (desiredZ - p.z) * 0.04;
        } else {
          // Core morphing particle with subtle organic breathing micro-motion
          const breathe =
            Math.sin(time * 1.8 + i * 0.13) * 0.012;
          const targetX = p.tx[stg] * (1 + breathe);
          const targetY = p.ty[stg] * (1 + breathe) + floatBobY;
          const targetZ = p.tz[stg] * (1 + breathe);

          p.vx += (targetX - p.x) * 0.055;
          p.vy += (targetY - p.y) * 0.055;
          p.vz += (targetZ - p.z) * 0.055;
        }

        // Subtle kinetic trail when gliding across columns on scroll
        if (Math.abs(shiftDelta) > 0.0005) {
          p.vx += shiftDelta * (p.isPlume ? 0.22 : 0.11);
          p.vy -= Math.abs(shiftDelta) * 0.03;
        }

        // Apply damping
        p.vx *= 0.84;
        p.vy *= 0.84;
        p.vz *= 0.84;

        p.x += p.vx;
        p.y += p.vy;
        p.z += p.vz;

        // 3D rotation & perspective projection
        const rx = p.x * cosY - p.z * sinY;
        const rz1 = p.x * sinY + p.z * cosY;
        const ry = p.y * cosX - rz1 * sinX;
        const rz = p.y * sinX + rz1 * cosX;

        const fov = 2.8 / (2.8 + rz);
        const sx = centerX + rx * scale * fov;
        const sy = centerY + ry * scale * fov;

        // Direct Cursor Scatter / Repulsion Physics (swiping through breaks bits loose)
        if (pointer.isInsideCanvas) {
          const dx = sx - pointer.canvasX;
          const dy = sy - pointer.canvasY;
          const distSq = dx * dx + dy * dy;
          if (distSq < scatterRadiusSq && distSq > 0.01) {
            const dist = Math.sqrt(distSq);
            const force = (1 - dist / scatterRadius) * 0.036;
            const nx = dx / dist;
            const ny = dy / dist;
            // Convert 2D screen push back into 3D velocity impulse
            p.vx += (nx * cosY) * force;
            p.vy += ny * force;
            p.vz += (-nx * sinY) * force;
          }
        }

        // Depth-modulated size & opacity
        const depthScale = Math.max(0.45, Math.min(1.45, (1.35 - rz * 0.5) * fov));
        const radius = p.baseSize * depthScale;

        let alpha = p.alpha * Math.max(0.22, Math.min(1, 1.15 - rz * 0.45));
        if (p.isPlume) {
          // Fade in on detach, fade out gracefully at top of space plume
          const fadeIn = Math.min(1, p.plumeLife * 4.5);
          const fadeOut = Math.max(0, 1 - Math.pow(p.plumeLife, 1.65));
          alpha *= fadeIn * fadeOut;
        }

        if (alpha <= 0.02) continue;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = p.shade;

        if (p.isHollow) {
          ctx.strokeStyle = p.shade;
          ctx.lineWidth = 0.9;
          ctx.beginPath();
          ctx.arc(sx, sy, radius * 1.15, 0, Math.PI * 2);
          ctx.stroke();
        } else if (p.isSquare) {
          const side = radius * 1.75;
          ctx.fillRect(sx - side * 0.5, sy - side * 0.5, side, side);
        } else {
          ctx.beginPath();
          ctx.arc(sx, sy, radius, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      ctx.globalAlpha = 1;
      ctx.restore();

      animationFrameId = requestAnimationFrame(render);
    };

    const startLoop = () => {
      if (!animationFrameId && isVisible) {
        animationFrameId = requestAnimationFrame(render);
      }
    };

    const intersectionObserver = new IntersectionObserver(
      ([entry]) => {
        isVisible = entry.isIntersecting;
        if (isVisible) {
          startLoop();
        } else if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
          animationFrameId = 0;
        }
      },
      { threshold: 0.05 }
    );
    intersectionObserver.observe(container);

    const handleVisibilityChange = () => {
      if (document.hidden) {
        isVisible = false;
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
          animationFrameId = 0;
        }
      } else {
        isVisible = true;
        startLoop();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    startLoop();

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('mousemove', handleWindowMouseMove);
      canvas.removeEventListener('touchmove', handleTouchMove);
      canvas.removeEventListener('touchend', handleTouchEnd);
      canvas.removeEventListener('click', handleCanvasClick);
    };
  }, []);

  const currentStage = MORPH_STAGES[activeStage];

  return (
    <div className={`relative w-full ${heightClass} flex flex-col items-center justify-center select-none`}>
      {/* Interactive 3D Particle Canvas */}
      <div
        ref={containerRef}
        className="relative w-full h-full cursor-grab active:cursor-grabbing"
        title={`${currentStage.label} (${currentStage.sublabel}) · Move cursor to pan & scatter · Click to morph`}
      >
        <canvas ref={canvasRef} className="block w-full h-full" />
      </div>
    </div>
  );
};

export default NeuralParticleCanvas;

import React from 'react';

interface ArticleSchematicProps {
  slug: string;
  variant?: 'thumbnail' | 'featured' | 'banner';
  className?: string;
}

interface SchematicMeta {
  label: string;
  caption: string;
}

const schematicMetaBySlug: Record<string, SchematicMeta> = {
  'point-supervised-loveda-segmentation-pce': {
    label: 'Sparse Point Supervision & Tiled Inference',
    caption: '5% sparse point clicks over 256×256 satellite tiles versus dense polygon masks.'
  },
  'recursive-language-models-agent-context-scaling': {
    label: 'Recursive Context & Parallel Agent Fleet',
    caption: 'Externalizing long documents into Python REPL memory with parallel worker dispatch.'
  },
  'histopathology-breast-cancer-path-foundation-llm': {
    label: 'Whole-Slide Patch Pipeline & Clinical Copilot',
    caption: 'Biopsy slide tessellation, stain normalization, and subtype classification.'
  },
  'continuous-asl-fingerspelling-mediapipe-transformer': {
    label: 'Hand Kinematic Tracking to Sequence Decoder',
    caption: '3D hand landmark extraction and temporal compression for on-device translation.'
  },
  'rsna-lumbar-spine-mri-deep-learning': {
    label: 'Volumetric 3D MRI Multi-Level Grading',
    caption: '3D spatial slice alignment and localized pooling across L1/L2 through L5/S1.'
  },
  'the-illusion-of-replacement-product-engineers-ai-era': {
    label: 'Product & Systems Engineering Topology',
    caption: 'Bridging code generation with architecture, domain trade-offs, and user experience.'
  },
  'kv-cache-sliding-window-attention': {
    label: 'KV-Cache & Sliding Window Memory',
    caption: 'Reusing cached past states to keep long-context generation fast and linear.'
  },
  'understanding-transformer-architecture': {
    label: 'Multi-Head Causal Self-Attention',
    caption: 'Query, Key, and Value projections with causal masking and residual connections.'
  },
  'visualizing-positional-encodings': {
    label: 'Sinusoidal Waves & Rotary Embeddings',
    caption: 'How sequence order and relative token distances are encoded inside Transformers.'
  },
  'implementing-skip-gram': {
    label: 'Contextual Word Embedding Space',
    caption: 'Learning dense semantic vector representations from word co-occurrences.'
  },
  'tokenization-and-encoding-bpe': {
    label: 'Byte Pair Subword Merge Tree',
    caption: 'Iteratively merging frequent character pairs into a compact subword vocabulary.'
  }
};

export const getSeriesBadge = (slug: string): string | null => {
  const seriesOrder: Record<string, string> = {
    'tokenization-and-encoding-bpe': 'LLM Series · Part 1/5',
    'implementing-skip-gram': 'LLM Series · Part 2/5',
    'visualizing-positional-encodings': 'LLM Series · Part 3/5',
    'understanding-transformer-architecture': 'LLM Series · Part 4/5',
    'kv-cache-sliding-window-attention': 'LLM Series · Part 5/5'
  };
  return seriesOrder[slug] || null;
};

export const ArticleSchematic: React.FC<ArticleSchematicProps> = ({
  slug,
  variant = 'thumbnail',
  className = ''
}) => {
  const meta = schematicMetaBySlug[slug] || {
    label: 'System Architecture Overview',
    caption: 'Visual overview of the core architecture and system workflow.'
  };

  const renderGraphic = () => {
    switch (slug) {
      case 'point-supervised-loveda-segmentation-pce':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            <pattern id="grid-loveda" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-neutral-200" />
            </pattern>
            <rect width="240" height="130" fill="url(#grid-loveda)" />
            <path d="M 15 95 Q 70 65, 130 85 T 225 55 L 225 118 L 15 118 Z" className="fill-neutral-100 stroke-neutral-300" strokeWidth="1" />
            <path d="M 15 45 Q 95 25, 160 55 T 225 30" className="stroke-neutral-400" strokeWidth="1.2" strokeDasharray="3 3" />
            <rect x="78" y="22" width="84" height="84" rx="4" className="stroke-neutral-900 fill-white/60" strokeWidth="1.4" />
            <text x="86" y="35" className="fill-neutral-500 text-[7px] font-mono">256×256 TILE</text>
            {[
              { cx: 45, cy: 78, active: true },
              { cx: 102, cy: 52, active: true },
              { cx: 138, cy: 72, active: true },
              { cx: 115, cy: 88, active: true },
              { cx: 192, cy: 64, active: true },
              { cx: 92, cy: 68, active: false },
              { cx: 146, cy: 44, active: false }
            ].map((pt, i) => (
              <g key={i}>
                {pt.active && (
                  <circle cx={pt.cx} cy={pt.cy} r="6.5" className="stroke-neutral-400/80 animate-pulse" strokeWidth="0.9" />
                )}
                <circle
                  cx={pt.cx}
                  cy={pt.cy}
                  r={pt.active ? '3' : '2'}
                  className={pt.active ? 'fill-neutral-900' : 'fill-neutral-300'}
                />
              </g>
            ))}
          </svg>
        );

      case 'recursive-language-models-agent-context-scaling':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            <rect x="16" y="28" width="64" height="74" rx="5" className="fill-neutral-50 stroke-neutral-900" strokeWidth="1.3" />
            <text x="24" y="42" className="fill-neutral-900 text-[7px] font-mono font-bold">REPL MEM</text>
            <line x1="24" y1="52" x2="68" y2="52" className="stroke-neutral-300" strokeWidth="2" />
            <line x1="24" y1="62" x2="72" y2="62" className="stroke-neutral-900" strokeWidth="2" />
            <line x1="24" y1="72" x2="60" y2="72" className="stroke-neutral-300" strokeWidth="2" />
            <line x1="24" y1="82" x2="66" y2="82" className="stroke-neutral-300" strokeWidth="2" />
            <circle cx="126" cy="65" r="18" className="fill-neutral-900" />
            <text x="115" y="68" className="fill-white text-[7px] font-mono">SUPV</text>
            <path d="M 80 65 L 104 65" className="stroke-neutral-900" strokeWidth="1.2" strokeDasharray="2 2" />
            {[30, 65, 100].map((cy, idx) => (
              <g key={idx}>
                <path d={`M 144 65 Q 165 ${cy}, 182 ${cy}`} className="stroke-neutral-400" strokeWidth="1.1" />
                <rect x="182" y={cy - 11} width="42" height="22" rx="4" className="fill-white stroke-neutral-800" strokeWidth="1.1" />
                <text x="191" y={cy + 2} className="fill-neutral-800 text-[6.5px] font-mono">W_0{idx + 1}</text>
              </g>
            ))}
          </svg>
        );

      case 'rsna-lumbar-spine-mri-deep-learning':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            {[0, 1, 2, 3, 4].map((idx) => {
              const offset = idx * 24;
              const isFocus = idx === 2;
              return (
                <g key={idx} transform={`translate(${38 + offset}, ${18 + idx * 4})`}>
                  <polygon
                    points="0,20 38,4 38,68 0,84"
                    className={
                      isFocus
                        ? 'fill-neutral-900/10 stroke-neutral-900'
                        : 'fill-white/80 stroke-neutral-300'
                    }
                    strokeWidth={isFocus ? '1.5' : '1'}
                  />
                  {isFocus && (
                    <>
                      <circle cx="19" cy="44" r="7" className="stroke-neutral-900 fill-neutral-900/20" strokeWidth="1.2" />
                      <circle cx="19" cy="44" r="2.5" className="fill-neutral-900" />
                    </>
                  )}
                </g>
              );
            })}
            <text x="155" y="34" className="fill-neutral-500 text-[7px] font-mono">L1/L2 – L5/S1</text>
            <text x="155" y="46" className="fill-neutral-900 text-[7.5px] font-mono font-bold">3D MRI SERIES</text>
          </svg>
        );

      case 'histopathology-breast-cancer-path-foundation-llm':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            <g transform="translate(22, 22)">
              {[0, 1, 2].map((r) =>
                [0, 1, 2].map((c) => {
                  const highlight = (r === 1 && c === 1) || (r === 0 && c === 2);
                  return (
                    <rect
                      key={`${r}-${c}`}
                      x={c * 26}
                      y={r * 26}
                      width="22"
                      height="22"
                      rx="2.5"
                      className={
                        highlight
                          ? 'fill-neutral-900 stroke-neutral-900'
                          : 'fill-neutral-100 stroke-neutral-300'
                      }
                      strokeWidth="1"
                    />
                  );
                })
              )}
            </g>
            <path d="M 104 60 L 142 42 M 104 60 L 142 78" className="stroke-neutral-400" strokeWidth="1.1" strokeDasharray="2 2" />
            <rect x="142" y="26" width="76" height="76" rx="5" className="fill-white stroke-neutral-900" strokeWidth="1.2" />
            <text x="150" y="41" className="fill-neutral-900 text-[7px] font-mono font-bold">PATH-FOUND</text>
            <line x1="150" y1="50" x2="206" y2="50" className="stroke-neutral-900" strokeWidth="2" />
            <line x1="150" y1="60" x2="196" y2="60" className="stroke-neutral-300" strokeWidth="2" />
            <line x1="150" y1="70" x2="202" y2="70" className="stroke-neutral-300" strokeWidth="2" />
            <rect x="150" y="80" width="44" height="12" rx="2" className="fill-neutral-100 stroke-neutral-400" strokeWidth="0.8" />
            <text x="155" y="88" className="fill-neutral-700 text-[6px] font-mono">8 SUBTYPES</text>
          </svg>
        );

      case 'continuous-asl-fingerspelling-mediapipe-transformer':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            <g className="stroke-neutral-800" strokeWidth="1.2">
              <line x1="48" y1="98" x2="32" y2="68" />
              <line x1="48" y1="98" x2="46" y2="48" />
              <line x1="48" y1="98" x2="62" y2="44" />
              <line x1="48" y1="98" x2="76" y2="54" />
              <line x1="48" y1="98" x2="86" y2="74" />
            </g>
            {[
              [48, 98],
              [32, 68],
              [46, 48],
              [62, 44],
              [76, 54],
              [86, 74]
            ].map(([cx, cy], i) => (
              <circle key={i} cx={cx} cy={cy} r="3" className="fill-white stroke-neutral-900" strokeWidth="1.4" />
            ))}
            <path d="M 102 65 Q 116 35, 130 65 T 158 65 T 186 65" className="stroke-neutral-400" strokeWidth="1.2" />
            <rect x="192" y="46" width="32" height="38" rx="4" className="fill-neutral-900" />
            <text x="199" y="68" className="fill-white text-[8px] font-mono font-bold">A·S·L</text>
          </svg>
        );

      case 'understanding-transformer-architecture':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            {['Q', 'K', 'V'].map((label, i) => {
              const y = 26 + i * 32;
              return (
                <g key={label}>
                  <rect x="20" y={y} width="32" height="20" rx="3" className="fill-neutral-100 stroke-neutral-800" strokeWidth="1.1" />
                  <text x="32" y={y + 13} className="fill-neutral-900 text-[8px] font-mono font-bold">{label}</text>
                  <path d={`M 52 ${y + 10} C 78 ${y + 10}, 78 65, 102 65`} className="stroke-neutral-400" strokeWidth="1.1" />
                </g>
              );
            })}
            <rect x="102" y="38" width="52" height="52" rx="4" className="fill-white stroke-neutral-900" strokeWidth="1.3" />
            <polygon points="106,42 150,86 106,86" className="fill-neutral-900" />
            <polygon points="112,42 150,42 150,80" className="fill-neutral-200" />
            <path d="M 154 65 L 182 65" className="stroke-neutral-900" strokeWidth="1.3" />
            <rect x="182" y="49" width="38" height="32" rx="4" className="fill-neutral-900" />
            <text x="190" y="68" className="fill-white text-[7px] font-mono">MHA</text>
          </svg>
        );

      case 'kv-cache-sliding-window-attention':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            <g transform="translate(28, 22)">
              {[0, 1, 2, 3].map((row) => (
                <g key={row} transform={`translate(0, ${row * 21})`}>
                  <rect
                    x="0"
                    y="0"
                    width="115"
                    height="16"
                    rx="2.5"
                    className={
                      row < 3
                        ? 'fill-neutral-200/80 stroke-neutral-400'
                        : 'fill-neutral-900 stroke-neutral-900'
                    }
                    strokeWidth="1"
                  />
                  <text
                    x="8"
                    y="11"
                    className={`text-[6.5px] font-mono ${
                      row < 3 ? 'fill-neutral-600' : 'fill-white font-bold'
                    }`}
                  >
                    {row < 3 ? `CACHED STATE ${row + 1}` : 'NEW TOKEN STEP'}
                  </text>
                </g>
              ))}
            </g>
            <path d="M 162 96 Q 175 40, 212 24" className="stroke-neutral-300" strokeWidth="1.2" strokeDasharray="2 2" />
            <line x1="162" y1="86" x2="216" y2="86" className="stroke-neutral-900" strokeWidth="1.8" />
            <text x="166" y="34" className="fill-neutral-400 text-[7px] font-mono">RECOMPUTE</text>
            <text x="176" y="79" className="fill-neutral-900 text-[7px] font-mono font-bold">KV CACHE</text>
          </svg>
        );

      case 'visualizing-positional-encodings':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            <path d="M 18 48 Q 42 18, 66 48 T 114 48" className="stroke-neutral-900" strokeWidth="1.3" />
            <path d="M 18 82 Q 30 62, 42 82 T 66 82 T 90 82 T 114 82" className="stroke-neutral-400" strokeWidth="1.1" />
            <circle cx="176" cy="65" r="34" className="stroke-neutral-300 fill-neutral-50/50" strokeWidth="1" />
            <line x1="142" y1="65" x2="210" y2="65" className="stroke-neutral-200" strokeWidth="0.8" />
            <line x1="176" y1="31" x2="176" y2="99" className="stroke-neutral-200" strokeWidth="0.8" />
            <line x1="176" y1="65" x2="202" y2="44" className="stroke-neutral-900" strokeWidth="1.6" />
            <line x1="176" y1="65" x2="196" y2="84" className="stroke-neutral-400" strokeWidth="1.2" strokeDasharray="2 2" />
            <path d="M 192 52 A 22 22 0 0 1 190 78" className="stroke-neutral-700" strokeWidth="1.1" />
            <text x="195" y="67" className="fill-neutral-900 text-[7px] font-mono font-bold">RoPE</text>
          </svg>
        );

      case 'implementing-skip-gram':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            <circle cx="115" cy="65" r="14" className="fill-neutral-900" />
            <text x="104" y="68" className="fill-white text-[6.5px] font-mono">WORD</text>
            {[
              { x: 52, y: 36, label: 'CTX 1' },
              { x: 56, y: 94, label: 'CTX 2' }
            ].map((node, i) => (
              <g key={i}>
                <line x1="102" y1="65" x2={node.x + 12} y2={node.y} className="stroke-neutral-900" strokeWidth="1.3" />
                <circle cx={node.x} cy={node.y} r="12" className="fill-white stroke-neutral-900" strokeWidth="1.3" />
                <text x={node.x - 9} y={node.y + 2} className="fill-neutral-900 text-[5.5px] font-mono">{node.label}</text>
              </g>
            ))}
            {[
              { x: 188, y: 34 },
              { x: 196, y: 68 },
              { x: 184, y: 100 }
            ].map((neg, i) => (
              <g key={i}>
                <line x1="129" y1="65" x2={neg.x - 9} y2={neg.y} className="stroke-neutral-300" strokeWidth="1" strokeDasharray="3 3" />
                <circle cx={neg.x} cy={neg.y} r="7" className="fill-neutral-100 stroke-neutral-400" strokeWidth="1" />
              </g>
            ))}
          </svg>
        );

      case 'tokenization-and-encoding-bpe':
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            {[28, 66, 104, 142].map((x, i) => (
              <rect key={i} x={x} y="88" width="28" height="18" rx="3" className="fill-neutral-100 stroke-neutral-400" strokeWidth="1" />
            ))}
            <text x="35" y="100" className="fill-neutral-600 text-[7px] font-mono">&apos;l&apos;</text>
            <text x="73" y="100" className="fill-neutral-600 text-[7px] font-mono">&apos;o&apos;</text>
            <text x="111" y="100" className="fill-neutral-600 text-[7px] font-mono">&apos;w&apos;</text>
            <text x="147" y="100" className="fill-neutral-600 text-[7px] font-mono">&apos;er&apos;</text>
            <path d="M 42 88 L 61 64 M 80 88 L 61 64" className="stroke-neutral-700" strokeWidth="1.1" />
            <rect x="44" y="46" width="36" height="18" rx="3" className="fill-white stroke-neutral-800" strokeWidth="1.2" />
            <text x="52" y="58" className="fill-neutral-900 text-[7px] font-mono">&apos;lo&apos;</text>
            <path d="M 62 46 L 96 28 M 118 88 L 96 28" className="stroke-neutral-900" strokeWidth="1.3" />
            <rect x="74" y="14" width="46" height="20" rx="3.5" className="fill-neutral-900" />
            <text x="83" y="27" className="fill-white text-[7.5px] font-mono font-bold">&apos;low&apos;</text>
            <text x="172" y="58" className="fill-neutral-400 text-[7px] font-mono">BPE MERGE</text>
          </svg>
        );

      default:
        return (
          <svg viewBox="0 0 240 130" className="w-full h-full" fill="none">
            <rect x="24" y="28" width="52" height="74" rx="4" className="fill-neutral-100 stroke-neutral-400" strokeWidth="1" />
            <rect x="94" y="20" width="52" height="90" rx="4" className="fill-white stroke-neutral-900" strokeWidth="1.4" />
            <rect x="164" y="28" width="52" height="74" rx="4" className="fill-neutral-900" />
            <line x1="76" y1="65" x2="94" y2="65" className="stroke-neutral-800" strokeWidth="1.2" />
            <line x1="146" y1="65" x2="164" y2="65" className="stroke-neutral-800" strokeWidth="1.2" />
          </svg>
        );
    }
  };

  if (variant === 'thumbnail') {
    return (
      <div
        className={`relative overflow-hidden rounded-lg border border-neutral-200/80 bg-neutral-50/60 group-hover:border-neutral-400 group-hover:bg-white transition-all duration-300 ${className}`}
      >
        <div className="w-full h-full transform group-hover:scale-[1.04] transition-transform duration-300">
          {renderGraphic()}
        </div>
      </div>
    );
  }

  if (variant === 'featured') {
    return (
      <div
        className={`relative overflow-hidden rounded-xl border border-neutral-200/90 bg-neutral-50/60 group-hover:border-neutral-300 group-hover:bg-neutral-50/30 transition-all duration-300 flex flex-col justify-between ${className}`}
      >
        <div className="flex-1 min-h-[135px] flex items-center justify-center p-3 transform group-hover:scale-[1.02] transition-transform duration-300">
          {renderGraphic()}
        </div>
        <div className="px-3.5 py-2 bg-white/90 border-t border-neutral-200/60 text-[10.5px] font-mono text-neutral-500 truncate">
          {meta.label}
        </div>
      </div>
    );
  }

  // Banner variant inside BlogPost.tsx — clean visual diagram without any formula bar
  return (
    <figure className={`my-8 rounded-2xl border border-neutral-200/90 bg-neutral-50/40 overflow-hidden ${className}`}>
      <div className="h-44 sm:h-52 w-full max-w-[460px] mx-auto p-3 flex items-center justify-center">
        {renderGraphic()}
      </div>
      <figcaption className="px-4 py-2.5 bg-white border-t border-neutral-100 text-xs text-neutral-500 text-center">
        {meta.caption}
      </figcaption>
    </figure>
  );
};

export default ArticleSchematic;

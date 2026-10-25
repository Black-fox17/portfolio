import { ArticleItem, OpenSourceReproduction } from '../types/portfolio';

export const publishedArticles: ArticleItem[] = [
  {
    id: '11',
    slug: 'point-supervised-loveda-segmentation-pce',
    title: 'Point-Supervised Remote Sensing Segmentation: Empirical Scaling & Partial Cross-Entropy Dynamics on LoveDA',
    date: '2026-10-10',
    readTime: '10 min read',
    category: 'Computer Vision & Medical AI',
    excerpt:
      'An empirical deep learning study on point-supervised aerial semantic segmentation on the LoveDA Urban benchmark: mathematical formulation of Partial Cross-Entropy (pCE), U-Net architecture, and the discovery of non-monotonic scaling where 5% density achieves peak mIoU.'
  },
  {
    id: '10',
    slug: 'recursive-language-models-agent-context-scaling',
    title: 'Beyond Context Windows: Engineering Autonomous Agent Systems with Recursive Language Models (RLMs)',
    date: '2025-02-18',
    readTime: '11 min read',
    category: 'Agentic Systems & Engineering',
    excerpt:
      'Why 1M+ token prompt stuffing leads to context rot and latency spikes, how Recursive Language Models (RLMs) treat context as variable memory inside a Python REPL, and how we engineered the Asera Core supervisor orchestrator from scratch.'
  },
  {
    id: '8',
    slug: 'histopathology-breast-cancer-path-foundation-llm',
    title: 'Clinical Histopathology AI: From Pathology Foundation Models to Multimodal Diagnostic Copilots',
    date: '2025-01-15',
    readTime: '9 min read',
    category: 'Computer Vision & Medical AI',
    excerpt:
      'How we engineered an end-to-end breast cancer subtype classification and multimodal clinical copilot at IAMRAT (UCH Ibadan): leveraging Google Path Foundation models, focal loss optimization across 8 histological subtypes, and interpretable diagnostic reporting.'
  },
  {
    id: '7',
    slug: 'continuous-asl-fingerspelling-mediapipe-transformer',
    title: 'Continuous Sign Language Translation: From MediaPipe Kinematics to Edge Transformers',
    date: '2024-11-29',
    readTime: '9 min read',
    category: 'Computer Vision & Medical AI',
    excerpt:
      'A first-principles deep dive into translating continuous ASL fingerspelling from raw video to text: landmark spatial normalization, temporal downsampling convolutions, transformer encoder-decoder attention, and TFLite quantization for sub-30ms edge inference.'
  },
  {
    id: '9',
    slug: 'rsna-lumbar-spine-mri-deep-learning',
    title: 'Multi-Level Lumbar Spine Degenerative Disease Grading on Volumetric MRI',
    date: '2024-10-18',
    readTime: '9 min read',
    category: 'Computer Vision & Medical AI',
    excerpt:
      'End-to-end deep learning framework for grading 25 multi-level spinal degeneration targets: spatial slice coordinate projection, uniform 2.5D volumetric sampling, Generalized Mean (GeM) pooling, and severe-penalized weighted log loss optimization.'
  },
  {
    id: '1',
    slug: 'tokenization-and-encoding-bpe',
    title: 'Tokenization and Encoding with BytePairEncoder (BPE)',
    date: '2025-12-28',
    readTime: '8 min read',
    category: 'Language Models & Deep Learning',
    excerpt:
      'A practical walkthrough of byte pair encoding (BPE), showing how raw text is transformed into subword tokens and why tokenization—not raw characters—is what language models actually see.'
  },
  {
    id: '2',
    slug: 'implementing-skip-gram',
    title: 'From Distributional Hypothesis to Code: Implementing Skip-Gram',
    date: '2026-01-04',
    readTime: '6 min read',
    category: 'Language Models & Deep Learning',
    excerpt:
      'A first-principles implementation of Skip-Gram and negative sampling in PyTorch, exploring word vector spaces and semantic clustering.'
  },
  {
    id: '3',
    slug: 'visualizing-positional-encodings',
    title: 'Visualizing Positional Encodings: What Actually Changes Inside a Transformer',
    date: '2026-01-12',
    readTime: '7 min read',
    category: 'Language Models & Deep Learning',
    excerpt:
      'A mathematical and visual breakdown of absolute sinusoidal versus rotary positional embeddings (RoPE), and why sequence order awareness is vital for self-attention.'
  },
  {
    id: '4',
    slug: 'understanding-transformer-architecture',
    title: 'Attention is All You Need: Understanding the Transformer Architecture',
    date: '2026-01-20',
    readTime: '10 min read',
    category: 'Language Models & Deep Learning',
    excerpt:
      'Deconstructing multi-head scaled dot-product attention, query-key-value projections, feed-forward sublayers, and residual normalization from the ground up.'
  },
  {
    id: '5',
    slug: 'kv-cache-sliding-window-attention',
    title: 'Modern Inference Optimizations: KV Cache and Sliding Window Attention',
    date: '2026-02-05',
    readTime: '7 min read',
    category: 'Language Models & Deep Learning',
    excerpt:
      'Exploring how KV Caching and Flash/Sliding Window attention transform quadratic generation costs into linear time memory lookups for long context windows.'
  }
];

export const openSourceReproductions: OpenSourceReproduction[] = [
  {
    id: 'point-supervised-loveda-segmentation',
    title: 'Point-Supervised LoveDA Semantic Segmentation',
    description: 'Weakly-supervised aerial remote sensing segmentation on LoveDA Urban using Partial Cross-Entropy (pCE), U-Net architecture, and non-overlapping tiled sliding-window inference.',
    category: 'Computer Vision & Deep Learning',
    technologies: ['PyTorch', 'U-Net', 'Remote Sensing', 'LoveDA', 'pCE Loss'],
    githubUrl: 'https://github.com/Black-fox17/point-supervised-loveda-segmentation'
  },
  {
    id: 'asera-core-supervisor-agent',
    title: 'Asera Core: Supervisor Agent & RLM Context Engine',
    description: 'Modular multi-agent orchestration framework: dynamic routing schema synthesis, parallel asyncio.gather worker fleets, and unbounded RLM context reasoning via isolated Python REPLs.',
    category: 'Agentic AI & Distributed Systems',
    technologies: ['Python', 'Gemini 2.0', 'RLM', 'AsyncIO', 'PostgreSQL', 'Docker'],
    githubUrl: 'https://github.com/Black-fox17/asera-core'
  },
  {
    id: 'rsna-lumbar-spine-grading',
    title: 'RSNA Lumbar Spine Degenerative Disease Classification',
    description: 'Volumetric multi-level 25-target spine degeneration grading: DICOM VOI LUT normalization, GeM pooling backbones, and severe-weighted multi-class loss.',
    category: 'Medical AI & Vision',
    technologies: ['PyTorch', 'DICOM', 'GeM-Pooling', 'EfficientNetV2', 'FastAI'],
    githubUrl: 'https://github.com/Black-fox17/rsna-lumbar-spine-classification-'
  },
  {
    id: 'breast-cancer-histopathology-ai',
    title: 'Breast Cancer Histopathology & Diagnostic Copilot',
    description: 'Google Path-Foundation Vision Transformer embeddings, 8-class subtype classification, and multimodal Gemini diagnostic explanation generation.',
    category: 'Medical AI & Vision',
    technologies: ['Path-Foundation', 'TensorFlow', 'Gemini', 'FastAPI', 'Docker'],
    githubUrl: 'https://github.com/Black-fox17/BreastCancer.git'
  },
  {
    id: 'asl-fingerspelling-recognition',
    title: 'Continuous ASL Fingerspelling Engine',
    description: 'Real-time gesture-to-text translation: 543 MediaPipe holistic keypoint filtering, 1D conv temporal downsampling, seq2seq transformer, and sub-30ms TFLite model.',
    category: 'Vision & Multimodal',
    technologies: ['TensorFlow', 'MediaPipe', 'Transformers', 'TFLite', 'Python'],
    githubUrl: 'https://github.com/Black-fox17/ASL.git'
  },
  {
    id: 'gpt-transformer',
    title: 'GPT-Style Autoregressive Transformer',
    description: 'PyTorch implementation of a decoder-only generative transformer with causal masking and multi-head attention.',
    category: 'Transformers & NLP',
    technologies: ['PyTorch', 'Python', 'CUDA'],
    githubUrl: 'https://github.com/Black-fox17'
  },
  {
    id: 'bpe-from-scratch',
    title: 'Byte Pair Encoding (BPE) Tokenizer Engine',
    description: 'Zero-dependency BPE tokenizer with vocabulary training, merge table persistence, and subword segmentation.',
    category: 'Transformers & NLP',
    technologies: ['Python', 'Algorithms', 'Regex'],
    githubUrl: 'https://github.com/Black-fox17'
  },
  {
    id: 'skipgram-word2vec',
    title: 'Skip-Gram with Negative Sampling (Word2Vec)',
    description: 'Vector embedding training using PyTorch sparse embeddings and noise-contrastive estimation.',
    category: 'Transformers & NLP',
    technologies: ['PyTorch', 'Embeddings', 'NLP'],
    githubUrl: 'https://github.com/Black-fox17'
  },
  {
    id: 'kv-cache-optimization',
    title: 'KV Cache & Sliding Window Attention Simulator',
    description: 'Profiling memory allocations and cache hit rates in autoregressive text generation.',
    category: 'Optimization',
    technologies: ['PyTorch', 'Performance Benchmarking'],
    githubUrl: 'https://github.com/Black-fox17'
  },
  {
    id: 'yolo-retail-vision',
    title: 'YOLO-Based Retail Shelf Intelligence',
    description: 'Custom bounding box detection, SKU classification, and spatial planogram gap analysis.',
    category: 'Vision & Multimodal',
    technologies: ['YOLO', 'Computer Vision', 'PyTorch'],
    githubUrl: 'https://github.com/Black-fox17'
  }
];

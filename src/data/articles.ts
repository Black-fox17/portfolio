import { ArticleItem, OpenSourceReproduction } from '../types/portfolio';

export const publishedArticles: ArticleItem[] = [
  {
    id: '7',
    slug: 'continuous-asl-fingerspelling-mediapipe-transformer',
    title: 'Continuous Sign Language Translation: From MediaPipe Kinematics to Edge Transformers',
    date: '2024-11-29',
    readTime: '9 min read',
    category: 'Computer Vision & Deep Learning',
    excerpt:
      'A first-principles deep dive into translating continuous ASL fingerspelling from raw video to text: landmark spatial normalization, temporal downsampling convolutions, transformer encoder-decoder attention, and TFLite quantization for sub-30ms edge inference.'
  },
  {
    id: '1',
    slug: 'tokenization-and-encoding-bpe',
    title: 'Tokenization and Encoding with BytePairEncoder (BPE)',
    date: '2025-12-28',
    readTime: '8 min read',
    category: 'NLP & Tokenization',
    excerpt:
      'A practical walkthrough of byte pair encoding (BPE), showing how raw text is transformed into subword tokens and why tokenization—not raw characters—is what language models actually see.'
  },
  {
    id: '2',
    slug: 'implementing-skip-gram',
    title: 'From Distributional Hypothesis to Code: Implementing Skip-Gram',
    date: '2026-01-04',
    readTime: '6 min read',
    category: 'Representation Learning',
    excerpt:
      'A first-principles implementation of Skip-Gram and negative sampling in PyTorch, exploring word vector spaces and semantic clustering.'
  },
  {
    id: '3',
    slug: 'visualizing-positional-encodings',
    title: 'Visualizing Positional Encodings: What Actually Changes Inside a Transformer',
    date: '2026-01-12',
    readTime: '7 min read',
    category: 'Transformer Mechanics',
    excerpt:
      'A mathematical and visual breakdown of absolute sinusoidal versus rotary positional embeddings (RoPE), and why sequence order awareness is vital for self-attention.'
  },
  {
    id: '4',
    slug: 'understanding-transformer-architecture',
    title: 'Attention is All You Need: Understanding the Transformer Architecture',
    date: '2026-01-20',
    readTime: '10 min read',
    category: 'Deep Learning',
    excerpt:
      'Deconstructing multi-head scaled dot-product attention, query-key-value projections, feed-forward sublayers, and residual normalization from the ground up.'
  },
  {
    id: '5',
    slug: 'kv-cache-sliding-window-attention',
    title: 'Modern Inference Optimizations: KV Cache and Sliding Window Attention',
    date: '2026-02-05',
    readTime: '7 min read',
    category: 'Systems & Inference',
    excerpt:
      'Exploring how KV Caching and Flash/Sliding Window attention transform quadratic generation costs into linear time memory lookups for long context windows.'
  }
];

export const openSourceReproductions: OpenSourceReproduction[] = [
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

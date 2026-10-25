export interface BlogPost {
  id: string;
  slug: string;
  title: string;
  date: string;
  readTime: string;
  category: string;
  excerpt: string;
  file: () => Promise<{ default: string }>;
}

export const blogPosts: BlogPost[] = [
  {
    id: "1",
    slug: "tokenization-and-encoding-bpe",
    title: "Tokenization and Encoding with BytePairEncoder(BPE)",
    date: "2025-12-28",
    readTime: "8 min read",
    category: "Natural Language Processing",
    excerpt: "A practical walkthrough of byte pair encoding (BPE), showing how raw text is transformed into tokens and why tokenization not characters is what language models actually see.",
    file: () => import('./posts/bpe.md?raw')
  },
  {
    id: "2",
    slug: "implementing-skip-gram",
    title: "From Distributional Hypothesis to Code: Implementing Skip-Gram",
    date: "2026-01-04",
    readTime: "6 min read",
    category: "Natural Language Processing",
    excerpt: "A practical walkthrough of implementing Skip-Gram, a key component of Word2Vec, using PyTorch.",
    file: () => import('./posts/learned_embedding.md?raw')
  },
  {
    id: "3",
    slug: "visualizing-positional-encodings",
    title: "Visualizing Positional Encodings: What Actually Changes Inside a Transformer",
    date: "2026-01-12",
    readTime: "7 min read",
    category: "Natural Language Processing",
    excerpt: "A deep dive into the inner workings of positional encodings in transformers, exploring the impact of different encoding strategies on model performance.",
    file: () => import('./posts/position.md?raw')
  },
  {
    id: "4",
    slug: "understanding-transformer-architecture",
    title: "Attention is All You Need: Understanding the Transformer Architecture",
    date: "2026-01-20",
    readTime: "10 min read",
    category: "Natural Language Processing",
    excerpt: "A deep dive into the inner workings of transformers",
    file: () => import('./posts/transformer.md?raw')
  },
  {
    id: "5",
    slug: "kv-cache-sliding-window-attention",
    title: "Modern Inference Optimizations: KV Cache and Sliding Window Attention",
    date: "2026-02-05",
    readTime: "7 min read",
    category: "Natural Language Processing",
    excerpt: "Exploring the latest optimizations in LLM inference, including KV Cache and Sliding Window Attention.",
    file: () => import('./posts/kv_cache.md?raw')
  },
  {
    id: "6",
    slug: "the-illusion-of-replacement-product-engineers-ai-era",
    title: "The Illusion of Replacement: Why the AI Era Demands Product Engineers, Not Just Code Generators",
    date: "2026-09-19",
    readTime: "7 min read",
    category: "Engineering & AI",
    excerpt: "Why syntax was never the real bottleneck of software engineering, the crucial difference between code generation and product design, and how to thrive as a T-shaped specialist in the AI era.",
    file: () => import('./posts/impact_ai.md?raw')
  },
  {
    id: "7",
    slug: "continuous-asl-fingerspelling-mediapipe-transformer",
    title: "Continuous Sign Language Translation: From MediaPipe Kinematics to Edge Transformers",
    date: "2024-11-29",
    readTime: "9 min read",
    category: "Computer Vision & Deep Learning",
    excerpt: "A first-principles deep dive into translating continuous ASL fingerspelling from raw video to text: landmark spatial normalization, temporal downsampling convolutions, transformer encoder-decoder attention, and TFLite quantization for sub-30ms edge inference.",
    file: () => import('./posts/asl_fingerspelling.md?raw')
  },
  {
    id: "8",
    slug: "histopathology-breast-cancer-path-foundation-llm",
    title: "Clinical Histopathology AI: From Pathology Foundation Models to Multimodal Diagnostic Copilots",
    date: "2025-01-15",
    readTime: "9 min read",
    category: "Medical AI & Clinical Decision Support",
    excerpt: "How we engineered an end-to-end breast cancer subtype classification and multimodal clinical copilot at IAMRAT (UCH Ibadan): leveraging Google Path Foundation models, focal loss optimization across 8 histological subtypes, and interpretable diagnostic reporting.",
    file: () => import('./posts/breast_cancer_histopathology.md?raw')
  },
  {
    id: "9",
    slug: "rsna-lumbar-spine-mri-deep-learning",
    title: "Multi-Level Lumbar Spine Degenerative Disease Grading on Volumetric MRI",
    date: "2024-10-18",
    readTime: "9 min read",
    category: "Medical AI & Computer Vision",
    excerpt: "End-to-end deep learning framework for grading 25 multi-level spinal degeneration targets: spatial slice coordinate projection, uniform 2.5D volumetric sampling, Generalized Mean (GeM) pooling, and severe-penalized weighted log loss optimization.",
    file: () => import('./posts/rsna_lumbar_spine.md?raw')
  },
  {
    id: "10",
    slug: "recursive-language-models-agent-context-scaling",
    title: "Beyond Context Windows: Engineering Autonomous Agent Systems with Recursive Language Models (RLMs)",
    date: "2025-02-18",
    readTime: "11 min read",
    category: "Agentic AI & Systems Architecture",
    excerpt: "Why 1M+ token prompt stuffing leads to context rot and latency spikes, how Recursive Language Models (RLMs) treat context as variable memory inside a Python REPL, and how we engineered the Asera Core supervisor orchestrator from scratch.",
    file: () => import('./posts/recursive_language_models_agent_context.md?raw')
  },
];

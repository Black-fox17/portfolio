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
];

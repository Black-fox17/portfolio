export interface BlogPost {
    id: string;
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
        title: "Tokenization and Encoding with BytePairEncoder(BPE)",
        date: "2025-12-28",
        readTime: "8 min read",
        category: "Natural Language Processing",
        excerpt: "A practical walkthrough of byte pair encoding (BPE), showing how raw text is transformed into tokens and why tokenization not characters is what language models actually see.",
        file: () => import('./posts/bpe.md?raw')
  },
  {
    id: "2",
    title: "From Distributional Hypothesis to Code: Implementing Skip-Gram",
    date: "2026-01-04",
    readTime: "6 min read",
    category: "Natural Language Processing",
    excerpt: "A practical walkthrough of implementing Skip-Gram, a key component of Word2Vec, using PyTorch.",
    file: () => import('./posts/learned_embedding.md?raw')
  },
  {
    id: "3",
    title: "Visualizing Positional Encodings: What Actually Changes Inside a Transformer",
    date: "2026-01-12",
    readTime: "6 min read",
    category: "Natural Language Processing",
    excerpt: "A deep dive into the inner workings of positional encodings in transformers, exploring the impact of different encoding strategies on model performance.",
    file: () => import('./posts/position.md?raw')
  },
  {
    id: "4",
    title: "Attention is All You Need: Understanding the Transformer Architecture",
    date: "2026-01-20",
    readTime: "10 min read",
    category: "Natural Language Processing",
    excerpt: "A deep dive into the inner workings of transformers",
    file: () => import('./posts/transformer.md?raw')
  },
];

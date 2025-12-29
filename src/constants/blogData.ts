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
];

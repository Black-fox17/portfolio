import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { blogPosts } from '../constants/blogData';
import Navbar from '../components/layout/Navbar';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft, Search } from 'lucide-react';

export const BlogList: React.FC = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const categories = ['All', ...Array.from(new Set(blogPosts.map((p) => p.category)))];

  const filteredPosts = blogPosts.filter((post) => {
    const matchesSearch =
      post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.excerpt.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || post.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="min-h-screen bg-white text-neutral-900 selection:bg-neutral-900 selection:text-white flex flex-col"
    >
      <Helmet>
        <title>The Deep End | Abdulsalam Ayeleru</title>
        <meta
          name="description"
          content="In-depth technical notes on byte-pair encoding, transformers, positional encodings, and inference optimizations by Abdulsalam Oluwaseun Ayeleru."
        />
      </Helmet>

      <Navbar />

      <main className="flex-1 max-w-[680px] w-full mx-auto px-5 sm:px-6 pt-20 sm:pt-28 pb-20">
        {/* Back Link */}
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-900 transition-colors mb-10"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Home</span>
        </Link>

        {/* Publication Header */}
        <header className="space-y-3 mb-10">
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-neutral-900">
            The Deep End.
          </h1>
          <p className="text-sm sm:text-base text-neutral-600 leading-relaxed">
            Technical notes on artificial intelligence, systems architecture, and first-principles deep learning by Abdulsalam Oluwaseun Ayeleru.
          </p>
        </header>

        {/* Search & Categories */}
        <div className="space-y-4 mb-10">
          <div className="relative">
            <Search className="w-4 h-4 text-neutral-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search articles..."
              className="w-full pl-9 pr-4 py-2 rounded-md border border-neutral-200 bg-white text-sm text-neutral-900 placeholder:text-neutral-400 focus:outline-none focus:border-neutral-900 transition-colors"
            />
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
                  selectedCategory === cat
                    ? 'bg-neutral-900 text-white'
                    : 'text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Articles List */}
        {filteredPosts.length === 0 ? (
          <div className="py-12 text-center text-sm text-neutral-500">
            No articles match your search.
          </div>
        ) : (
          <div className="space-y-8 divide-y divide-neutral-100">
            {filteredPosts.map((post, idx) => (
              <article
                key={post.id}
                onClick={() => navigate(`/blog/${post.slug}`)}
                className={`group cursor-pointer ${idx > 0 ? 'pt-8' : ''}`}
              >
                <div className="space-y-1.5">
                  <div className="flex items-baseline justify-between gap-3">
                    <h2 className="text-base font-medium text-neutral-900 group-hover:underline underline-offset-4 decoration-neutral-300">
                      {post.title}
                    </h2>
                    <span className="text-xs text-neutral-400 font-mono shrink-0">
                      {post.readTime}
                    </span>
                  </div>

                  <p className="text-sm text-neutral-600 leading-relaxed">
                    {post.excerpt}
                  </p>

                  <div className="flex items-center gap-3 text-xs text-neutral-400 font-mono pt-1">
                    <span>{post.category}</span>
                    <span>·</span>
                    <time dateTime={post.date}>
                      {new Date(post.date).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric'
                      })}
                    </time>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}

        <footer className="pt-16 mt-16 border-t border-neutral-100 text-xs text-neutral-400 flex items-center justify-between">
          <span>© {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru</span>
          <button
            type="button"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="hover:text-neutral-900 transition-colors"
          >
            Back to top ↑
          </button>
        </footer>
      </main>
    </motion.div>
  );
};

export default BlogList;

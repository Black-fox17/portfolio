import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { blogPosts } from '../constants/blogData';
import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft, ArrowUpRight, Search, BookOpen, Clock, Calendar } from 'lucide-react';

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
    <div className="min-h-screen bg-paper text-ink flex flex-col selection:bg-ink selection:text-paper">
      <Helmet>
        <title>The Deep End — Technical Writing & AI Systems | Abdulsalam Ayeleru</title>
        <meta
          name="description"
          content="In-depth technical notes on byte-pair encoding, transformers, skip-gram, positional encodings, and inference optimizations by Abdulsalam Oluwaseun Ayeleru."
        />
        <link rel="canonical" href="https://salam-portfolio-three.vercel.app/blog" />
        <meta property="og:title" content="The Deep End — Technical Writing | Abdulsalam Ayeleru" />
        <meta
          property="og:description"
          content="In-depth technical notes on AI systems, transformers, tokenization, and deep learning architectures."
        />
      </Helmet>

      <Navbar />

      <main className="flex-1 pt-32 pb-20">
        <div className="max-w-4xl mx-auto px-5 sm:px-8">
          {/* Back to Home Link */}
          <Link
            to="/"
            className="inline-flex items-center gap-1.5 text-xs font-mono text-ink-muted hover:text-ink transition-colors mb-8"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Portfolio</span>
          </Link>

          {/* Publication Header */}
          <div className="border-b border-border pb-10 mb-10 space-y-4">
            <div className="section-heading">
              <span className="w-2 h-2 rounded-full bg-ink" />
              <span>Technical Publication</span>
            </div>
            <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-ink font-editorial">
              The Deep End
            </h1>
            <p className="text-sm sm:text-base text-ink-muted max-w-2xl leading-relaxed">
              Notes on artificial intelligence, distributed systems, and the mathematical machinery underneath modern software. Written from a first-principles engineering perspective by{' '}
              <span className="text-ink font-medium">Abdulsalam Oluwaseun Ayeleru</span>.
            </p>
          </div>

          {/* Search & Filter Controls */}
          <div className="mb-8 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
            {/* Search Input */}
            <div className="relative flex-1 max-w-md">
              <Search className="w-4 h-4 text-ink-faint absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search articles and topics..."
                className="w-full pl-9 pr-4 py-2 rounded-lg border border-border bg-white text-xs text-ink placeholder:text-ink-faint focus:outline-none focus:border-ink transition-colors shadow-subtle"
              />
            </div>

            {/* Category Filter Pills */}
            <div className="flex flex-wrap items-center gap-1.5">
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                    selectedCategory === cat
                      ? 'bg-ink text-paper font-semibold shadow-subtle'
                      : 'bg-white border border-border text-ink-muted hover:bg-paper-100 hover:text-ink'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>

          {/* Articles Feed */}
          {filteredPosts.length === 0 ? (
            <div className="bg-white border border-border rounded-xl p-12 text-center space-y-2">
              <p className="text-sm font-semibold text-ink">No articles matched your criteria</p>
              <p className="text-xs text-ink-muted">Try refining your search terms or category filters.</p>
            </div>
          ) : (
            <div className="divide-y divide-border border-y border-border bg-white rounded-xl overflow-hidden shadow-subtle">
              {filteredPosts.map((post) => (
                <article
                  key={post.id}
                  onClick={() => navigate(`/blog/${post.id}`)}
                  className="p-6 sm:p-8 block hover:bg-paper-50 transition-colors cursor-pointer group"
                >
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                    <div className="space-y-2.5 max-w-2xl">
                      <div className="flex items-center gap-2 text-xs font-mono text-ink-faint">
                        <span className="editorial-tag text-[10px] !py-0.5">
                          {post.category}
                        </span>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          <span>{post.readTime}</span>
                        </span>
                      </div>

                      <h2 className="text-lg sm:text-xl font-bold text-ink font-editorial group-hover:text-accent-blue transition-colors">
                        {post.title}
                      </h2>

                      <p className="text-xs sm:text-sm text-ink-muted leading-relaxed">
                        {post.excerpt}
                      </p>
                    </div>

                    <div className="text-xs font-mono text-ink-faint shrink-0 flex sm:flex-col items-center sm:items-end justify-between sm:justify-start gap-2">
                      <time dateTime={post.date} className="flex items-center gap-1">
                        <Calendar className="w-3 h-3 text-ink-faint" />
                        <span>
                          {new Date(post.date).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric'
                          })}
                        </span>
                      </time>
                      <span className="text-ink group-hover:text-accent-blue font-medium flex items-center gap-1 text-xs">
                        <span>Read article</span>
                        <ArrowUpRight className="w-3.5 h-3.5 transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                      </span>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default BlogList;

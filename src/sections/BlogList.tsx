import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { blogPosts, BlogPost } from '../constants/blogData';
import Navbar from '../components/layout/Navbar';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft, ArrowUpRight, Search } from 'lucide-react';
import ArticleSchematic, { getSeriesBadge } from '../components/writing/ArticleSchematic';

export const BlogList: React.FC = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const categories = ['All', ...Array.from(new Set(blogPosts.map((p) => p.category)))];

  const getCategoryCount = (cat: string) =>
    cat === 'All' ? blogPosts.length : blogPosts.filter((p) => p.category === cat).length;

  const filteredPosts = blogPosts.filter((post) => {
    const matchesSearch =
      post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.excerpt.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || post.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const showFeaturedCard = selectedCategory === 'All' && searchQuery.trim() === '' && filteredPosts.length > 0;
  const featuredPost = showFeaturedCard ? filteredPosts[0] : null;
  const archivePosts = showFeaturedCard ? filteredPosts.slice(1) : filteredPosts;

  const handlePostClick = (post: BlogPost) => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim().length > 0) return;
    navigate(`/blog/${post.slug}`);
  };

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
          content="In-depth technical publications on byte-pair encoding, transformers, positional encodings, LoveDA segmentation, RLMs, and ML systems by Abdulsalam Ayeleru."
        />
        <link rel="canonical" href="https://ayeleru.space/blog" />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://ayeleru.space/blog" />
        <meta property="og:title" content="The Deep End | Abdulsalam Ayeleru" />
        <meta property="og:description" content="In-depth technical publications on deep learning, transformers, and AI systems engineering." />
        <meta property="og:image" content="https://ayeleru.space/assets/preview.png" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:site" content="@ayeleruseun" />
        <meta name="twitter:creator" content="@ayeleruseun" />
        <meta name="twitter:url" content="https://ayeleru.space/blog" />
        <meta name="twitter:title" content="The Deep End | Abdulsalam Ayeleru" />
        <meta name="twitter:description" content="In-depth technical publications on deep learning, transformers, and AI systems engineering." />
        <meta name="twitter:image" content="https://ayeleru.space/assets/preview.png" />
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
        <header className="space-y-3 mb-9 pb-8 border-b border-neutral-100">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] font-mono uppercase tracking-wider text-neutral-400">
              Essays, Systems &amp; Technical Notes
            </span>
            <span className="text-[11px] font-mono text-neutral-400">
              {blogPosts.length} Publications
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-neutral-900">
            The Deep End.
          </h1>
          <p className="text-sm sm:text-base text-neutral-600 leading-relaxed">
            Notes on software engineering, product thinking, systems architecture, and artificial intelligence by Abdulsalam Oluwaseun Ayeleru.
          </p>
        </header>

        {/* Search & Spring-Animated Category Pills */}
        <div className="space-y-4 mb-10">
          <div className="relative">
            <Search className="w-4 h-4 text-neutral-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search articles, topics, or systems..."
              className="w-full pl-9 pr-4 py-2 rounded-lg border border-neutral-200 bg-white text-sm text-neutral-900 placeholder:text-neutral-400 focus:outline-none focus:border-neutral-900 transition-colors"
            />
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {categories.map((cat) => {
              const isActive = selectedCategory === cat;
              return (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`relative px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer select-none ${
                    isActive
                      ? 'text-white'
                      : 'text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100/80'
                  }`}
                >
                  {isActive && (
                    <motion.span
                      layoutId="activeBlogCategoryTab"
                      transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                      className="absolute inset-0 rounded-lg bg-neutral-900"
                    />
                  )}
                  <span className="relative z-10 inline-flex items-center gap-1.5">
                    <span>{cat}</span>
                    <span
                      className={`text-[10px] font-mono ${
                        isActive ? 'text-neutral-400' : 'text-neutral-400/80'
                      }`}
                    >
                      {getCategoryCount(cat)}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Featured Latest Monograph Card */}
        <AnimatePresence mode="popLayout">
          {featuredPost && (
            <motion.section
              key={`featured-${featuredPost.id}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="mb-12"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-[11px] font-mono uppercase tracking-wider text-neutral-500 font-medium">
                  Latest Research Monograph
                </span>
                <span className="text-xs font-mono text-neutral-400">
                  {featuredPost.readTime}
                </span>
              </div>

              <article
                onClick={() => handlePostClick(featuredPost)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handlePostClick(featuredPost);
                  }
                }}
                tabIndex={0}
                role="link"
                aria-label={`Read featured article: ${featuredPost.title}`}
                className="group cursor-pointer bg-white border border-neutral-200/90 hover:border-neutral-400 rounded-2xl p-5 sm:p-6 shadow-subtle hover:shadow-sm transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-neutral-900"
              >
                <div className="grid grid-cols-1 sm:grid-cols-12 gap-5 items-center">
                  <div className="sm:col-span-7 space-y-3">
                    <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-neutral-400">
                      <span>{featuredPost.category}</span>
                      <span>·</span>
                      <time dateTime={featuredPost.date}>
                        {new Date(featuredPost.date).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric'
                        })}
                      </time>
                    </div>

                    <h2 className="text-lg sm:text-xl font-bold tracking-tight text-neutral-900 group-hover:underline underline-offset-4 decoration-neutral-300 leading-snug">
                      {featuredPost.title}
                    </h2>

                    <p className="text-xs sm:text-sm text-neutral-600 leading-relaxed line-clamp-3">
                      {featuredPost.excerpt}
                    </p>

                    <div className="pt-1 flex items-center gap-2">
                      <span className="inline-flex items-center gap-1 text-xs font-mono text-neutral-800 bg-neutral-100 group-hover:bg-neutral-900 group-hover:text-white px-2.5 py-1 rounded transition-colors">
                        <span>Read Monograph</span>
                        <ArrowUpRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                      </span>
                    </div>
                  </div>

                  <div className="sm:col-span-5">
                    <ArticleSchematic slug={featuredPost.slug} variant="featured" />
                  </div>
                </div>
              </article>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Archive List */}
        {filteredPosts.length === 0 ? (
          <div className="py-12 text-center text-sm text-neutral-500">
            No articles match your search.
          </div>
        ) : (
          <div className="space-y-3">
            {showFeaturedCard && archivePosts.length > 0 && (
              <div className="flex items-center justify-between pb-2 border-b border-neutral-100">
                <span className="text-[11px] font-mono uppercase tracking-wider text-neutral-400">
                  Publication Archive
                </span>
                <span className="text-[11px] font-mono text-neutral-400">
                  Schematic Index
                </span>
              </div>
            )}

            <motion.div layout className="divide-y divide-neutral-100">
              <AnimatePresence mode="popLayout">
                {archivePosts.map((post, idx) => {
                  const seriesBadge = getSeriesBadge(post.slug);
                  return (
                    <motion.article
                      layout
                      key={post.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.2, delay: Math.min(idx * 0.03, 0.18) }}
                      onClick={() => handlePostClick(post)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          handlePostClick(post);
                        }
                      }}
                      tabIndex={0}
                      role="link"
                      aria-label={`Read article: ${post.title}`}
                      className="group cursor-pointer py-5 sm:py-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-neutral-50/50 -mx-3 px-3 rounded-xl transition-colors outline-none focus-visible:ring-2 focus-visible:ring-neutral-900"
                    >
                      <div className="space-y-2 flex-1 min-w-0">
                        <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-neutral-400">
                          <span>{post.category}</span>
                          <span>·</span>
                          <time dateTime={post.date}>
                            {new Date(post.date).toLocaleDateString('en-US', {
                              year: 'numeric',
                              month: 'short',
                              day: 'numeric'
                            })}
                          </time>
                          <span>·</span>
                          <span>{post.readTime}</span>
                          {seriesBadge && (
                            <span className="px-1.5 py-0.5 rounded bg-neutral-100 text-neutral-600 text-[10px] font-mono">
                              {seriesBadge}
                            </span>
                          )}
                        </div>

                        <h2 className="text-base font-semibold text-neutral-900 group-hover:underline underline-offset-4 decoration-neutral-300 flex items-center gap-1.5 leading-snug">
                          <span>{post.title}</span>
                          <ArrowUpRight className="w-3.5 h-3.5 text-neutral-400 opacity-50 group-hover:opacity-100 group-hover:text-neutral-900 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all shrink-0" />
                        </h2>

                        <p className="text-xs sm:text-sm text-neutral-600 leading-relaxed line-clamp-2">
                          {post.excerpt}
                        </p>
                      </div>

                      {/* Bespoke Monochrome Technical Schematic Preview */}
                      <ArticleSchematic
                        slug={post.slug}
                        variant="thumbnail"
                        className="w-full sm:w-32 h-20 sm:h-20 shrink-0"
                      />
                    </motion.article>
                  );
                })}
              </AnimatePresence>
            </motion.div>
          </div>
        )}

        <footer className="pt-16 mt-16 border-t border-neutral-100 text-xs text-neutral-400 flex items-center justify-between">
          <span>© {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru</span>
          <button
            type="button"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="hover:text-neutral-900 transition-colors cursor-pointer"
          >
            Back to top ↑
          </button>
        </footer>
      </main>
    </motion.div>
  );
};

export default BlogList;

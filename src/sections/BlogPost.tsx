import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { blogPosts } from '../constants/blogData';
import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft, Clock, Calendar, Share2, Check, BookOpen } from 'lucide-react';
import './BlogStyles.css';

export const BlogPost: React.FC = () => {
  const { slug } = useParams<{ slug: string }>();
  const [content, setContent] = useState('');
  const [copied, setCopied] = useState(false);
  const navigate = useNavigate();

  const post = blogPosts.find((p) => p.slug === slug || p.id === slug);

  useEffect(() => {
    if (post) {
      post.file().then((res) => setContent(res.default));
    }
  }, [post]);

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: post?.title,
        url: window.location.href,
      }).catch(() => {});
    } else {
      navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!post) {
    return (
      <div className="min-h-screen bg-white text-neutral-900 flex flex-col">
        <Navbar />
        <main className="flex-1 max-w-[680px] mx-auto px-6 py-36 text-center space-y-4">
          <h1 className="text-2xl font-bold text-neutral-900">Article not found</h1>
          <p className="text-sm text-neutral-600">The requested technical publication does not exist.</p>
          <Link to="/blog" className="text-sm font-medium underline underline-offset-4 decoration-neutral-300 hover:decoration-neutral-900">
            ← Return to The Deep End
          </Link>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="min-h-screen bg-white text-neutral-900 flex flex-col selection:bg-neutral-900 selection:text-white"
    >
      <Helmet>
        <title>{post.title} | The Deep End | Abdulsalam Ayeleru</title>
        <meta name="description" content={post.excerpt} />
        <link rel="canonical" href={`https://ayeleru.space/blog/${post.slug}`} />
        <meta property="og:type" content="article" />
        <meta property="og:title" content={`${post.title} | Abdulsalam Ayeleru`} />
        <meta property="og:description" content={post.excerpt} />
      </Helmet>

      <Navbar />

      <main className="flex-1 pt-20 sm:pt-28 pb-24 w-full max-w-full overflow-x-hidden">
        <article className="max-w-[680px] w-full mx-auto px-5 sm:px-6 min-w-0">
          {/* Back Link */}
          <Link
            to="/blog"
            className="inline-flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-900 transition-colors mb-8"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to The Deep End</span>
          </Link>

          {/* Article Header */}
          <header className="border-b border-neutral-100 pb-8 mb-10 space-y-4">
            <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-500 font-mono">
              <span>{post.category}</span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5" />
                <span>{post.readTime}</span>
              </span>
            </div>

            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-neutral-900 leading-tight">
              {post.title}
            </h1>

            <div className="flex flex-wrap items-center justify-between gap-4 pt-3 text-xs text-neutral-500 border-t border-neutral-100">
              <div className="flex items-center gap-2">
                <span>By</span>
                <span className="text-neutral-900 font-medium">Abdulsalam Oluwaseun Ayeleru</span>
                <span>•</span>
                <time dateTime={post.date}>
                  {new Date(post.date).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </time>
              </div>

              <button
                onClick={handleShare}
                className="inline-flex items-center gap-1.5 text-xs text-neutral-600 hover:text-neutral-900 transition-colors cursor-pointer"
                title="Share article link"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-neutral-800" /> : <Share2 className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Share'}</span>
              </button>
            </div>
          </header>

          {/* Article Markdown Body */}
          <div className="markdown-editorial font-sans text-ink-light leading-relaxed max-w-full min-w-0">
            <ReactMarkdown
              remarkPlugins={[remarkMath, remarkGfm]}
              rehypePlugins={[rehypeKatex, rehypeRaw]}
              components={{
                table({ children, ...props }: any) {
                  return (
                    <div className="w-full max-w-full my-6 overflow-x-auto rounded-lg border border-neutral-200 scrollbar-none">
                      <table className="w-full min-w-[520px] border-collapse" {...props}>
                        {children}
                      </table>
                    </div>
                  );
                },
                pre({ children }: any) {
                  return <div className="w-full max-w-full my-6">{children}</div>;
                },
                img({ src, alt }) {
                  if (!src) return null;
                  const isVideo = src.endsWith('.mp4') || src.endsWith('.webm') || src.endsWith('.ogg');
                  if (isVideo) {
                    return (
                      <figure className="my-8 rounded-xl overflow-hidden border border-border bg-paper-100">
                        <video src={src} controls autoPlay loop muted playsInline className="w-full h-auto" />
                        {alt && <figcaption className="p-2.5 text-center text-xs font-mono text-ink-faint">{alt}</figcaption>}
                      </figure>
                    );
                  }
                  return (
                    <figure className="my-8 rounded-xl overflow-hidden border border-border bg-paper-100">
                      <img src={src} alt={alt || ''} className="w-full h-auto object-cover" />
                      {alt && <figcaption className="p-2.5 text-center text-xs font-mono text-ink-faint">{alt}</figcaption>}
                    </figure>
                  );
                },
                code({ inline, className, children, ...props }: any) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline ? (
                    <div className="w-full max-w-full min-w-0 rounded-xl overflow-hidden border border-neutral-200 bg-neutral-900 text-neutral-100 shadow-sm">
                      {match && (
                        <div className="px-4 py-1.5 bg-neutral-800 border-b border-neutral-700/60 text-[11px] font-mono text-neutral-400 uppercase tracking-wider flex justify-between items-center">
                          <span>{match[1]}</span>
                        </div>
                      )}
                      <pre className="p-4 overflow-x-auto text-xs font-mono leading-relaxed max-w-full scrollbar-none">
                        <code className={className} {...props}>
                          {children}
                        </code>
                      </pre>
                    </div>
                  ) : (
                    <code className="px-1.5 py-0.5 rounded bg-neutral-100 border border-neutral-200 text-[12px] font-mono text-neutral-900 font-medium break-words" {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {content}
            </ReactMarkdown>
          </div>

          {/* Article Footer Navigation */}
          <div className="mt-16 pt-8 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
            <Link to="/blog" className="editorial-btn-secondary text-xs">
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Explore More Publications</span>
            </Link>
            <a
              href="https://github.com/Black-fox17"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-ink-muted hover:text-ink transition-colors"
            >
              Find me on GitHub →
            </a>
          </div>
        </article>
      </main>

      <Footer />
    </motion.div>
  );
};

export default BlogPost;
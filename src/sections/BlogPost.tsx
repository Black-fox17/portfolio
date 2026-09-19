import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
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
      <div className="min-h-screen bg-paper text-ink flex flex-col">
        <Navbar />
        <main className="flex-1 max-w-3xl mx-auto px-6 py-36 text-center space-y-4">
          <h1 className="text-2xl font-bold font-editorial text-ink">Article not found</h1>
          <p className="text-sm text-ink-muted">The requested technical publication does not exist.</p>
          <Link to="/blog" className="editorial-btn-primary inline-flex">
            ← Return to The Deep End
          </Link>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-paper text-ink flex flex-col selection:bg-ink selection:text-paper">
      <Helmet>
        <title>{post.title} — The Deep End | Abdulsalam Ayeleru</title>
        <meta name="description" content={post.excerpt} />
        <link rel="canonical" href={`https://ayeleru.space/blog/${post.slug}`} />
        <meta property="og:type" content="article" />
        <meta property="og:title" content={`${post.title} | Abdulsalam Ayeleru`} />
        <meta property="og:description" content={post.excerpt} />
      </Helmet>

      <Navbar />

      <main className="flex-1 pt-32 pb-24">
        <article className="max-w-3xl mx-auto px-5 sm:px-8">
          {/* Back Link */}
          <Link
            to="/blog"
            className="inline-flex items-center gap-1.5 text-xs font-mono text-ink-muted hover:text-ink transition-colors mb-8"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to The Deep End</span>
          </Link>

          {/* Article Header */}
          <header className="border-b border-border pb-8 mb-10 space-y-4">
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
              <span className="editorial-tag text-[11px] !py-0.5">
                {post.category}
              </span>
              <span className="text-ink-faint">•</span>
              <span className="text-ink-muted flex items-center gap-1">
                <Clock className="w-3.5 h-3.5" />
                <span>{post.readTime}</span>
              </span>
            </div>

            <h1 className="text-2xl sm:text-4xl font-bold tracking-tight text-ink font-editorial leading-tight">
              {post.title}
            </h1>

            <div className="flex flex-wrap items-center justify-between gap-4 pt-2 text-xs font-mono text-ink-muted border-t border-border-subtle">
              <div className="flex items-center gap-2">
                <span>By</span>
                <span className="text-ink font-semibold">Abdulsalam Oluwaseun Ayeleru</span>
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
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-white border border-border text-ink hover:bg-paper-100 transition-colors"
                title="Share article link"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Share2 className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Share'}</span>
              </button>
            </div>
          </header>

          {/* Article Markdown Body */}
          <div className="markdown-editorial font-sans text-ink-light leading-relaxed">
            <ReactMarkdown
              remarkPlugins={[remarkMath, remarkGfm]}
              rehypePlugins={[rehypeKatex, rehypeRaw]}
              components={{
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
                    <div className="my-6 rounded-xl overflow-hidden border border-ink/20 bg-ink text-paper-50 shadow-md">
                      {match && (
                        <div className="px-4 py-1.5 bg-ink-light border-b border-white/10 text-[11px] font-mono text-paper-300 uppercase tracking-wider flex justify-between items-center">
                          <span>{match[1]}</span>
                        </div>
                      )}
                      <pre className="p-4 overflow-x-auto text-xs font-mono leading-relaxed">
                        <code className={className} {...props}>
                          {children}
                        </code>
                      </pre>
                    </div>
                  ) : (
                    <code className="px-1.5 py-0.5 rounded bg-paper-200 border border-border text-[12px] font-mono text-ink font-medium" {...props}>
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
              First-principles code on GitHub →
            </a>
          </div>
        </article>
      </main>

      <Footer />
    </div>
  );
};

export default BlogPost;
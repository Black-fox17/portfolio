import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, BookOpen, GitBranch, Terminal } from 'lucide-react';
import { publishedArticles, openSourceReproductions } from '../../data/articles';

export const TheDeepEnd: React.FC = () => {
  return (
    <section id="the-deep-end" className="border-b border-border/60 bg-paper">
      <div className="section-container">
        {/* Section Header */}
        <div className="mb-12">
          <div className="section-heading">
            <span className="w-2 h-2 rounded-full bg-ink" />
            <span>Chapter 06 — First Principles & Technical Writing</span>
          </div>
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
            <div>
              <h2 className="section-title">
                The Deep End: Notes on AI Systems & Machinery
              </h2>
              <p className="section-desc">
                Investigating modern software and machine learning from first principles. Exploring what happens underneath abstractions—from byte-pair encoding to transformer attention matrices and inference memory caches.
              </p>
            </div>
            <Link
              to="/blog"
              className="editorial-btn-secondary shrink-0 text-xs self-start sm:self-auto"
            >
              <span>View All Articles</span>
              <ArrowUpRight className="w-4 h-4" />
            </Link>
          </div>
        </div>

        {/* Featured Technical Articles List */}
        <div className="space-y-4 mb-16">
          <div className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-ink-muted" />
            <span>Published Technical Publications</span>
          </div>

          <div className="divide-y divide-border border-y border-border bg-white rounded-xl overflow-hidden shadow-subtle">
            {publishedArticles.map((article) => (
              <Link
                key={article.id}
                to={`/blog/${article.slug}`}
                className="p-6 sm:p-7 block hover:bg-paper-50 transition-colors group"
              >
                <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                  <div className="space-y-2 max-w-2xl">
                    <div className="flex items-center gap-2 text-xs font-mono">
                      <span className="editorial-tag text-[10px] !py-0.5">
                        {article.category}
                      </span>
                      <span className="text-ink-faint">•</span>
                      <span className="text-ink-faint">{article.readTime}</span>
                    </div>

                    <h3 className="text-base sm:text-lg font-bold text-ink font-editorial group-hover:text-accent-blue transition-colors">
                      {article.title}
                    </h3>

                    <p className="text-xs sm:text-sm text-ink-muted leading-relaxed">
                      {article.excerpt}
                    </p>
                  </div>

                  <div className="text-xs font-mono text-ink-faint shrink-0 flex sm:flex-col items-center sm:items-end justify-between sm:justify-start gap-2">
                    <time dateTime={article.date}>
                      {new Date(article.date).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric'
                      })}
                    </time>
                    <span className="text-ink group-hover:text-accent-blue font-medium flex items-center gap-1 text-xs">
                      <span>Read paper</span>
                      <ArrowUpRight className="w-3.5 h-3.5 transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                    </span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* First-Principles Open Source Reproductions */}
        <div className="space-y-4">
          <div className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5 text-ink-muted" />
            <span>First-Principles Paper Reproductions & Open Source</span>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {openSourceReproductions.map((repo) => (
              <a
                key={repo.id}
                href={repo.githubUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="editorial-card p-5 flex flex-col justify-between space-y-4 bg-white group hover:border-ink/50"
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-ink-faint font-semibold px-2 py-0.5 rounded bg-paper-100">
                      {repo.category}
                    </span>
                    <ArrowUpRight className="w-3.5 h-3.5 text-ink-faint group-hover:text-ink transition-colors" />
                  </div>

                  <h4 className="text-sm font-bold text-ink font-editorial group-hover:text-ink-light">
                    {repo.title}
                  </h4>

                  <p className="text-xs text-ink-muted leading-relaxed">
                    {repo.description}
                  </p>
                </div>

                <div className="pt-2 border-t border-border-subtle flex flex-wrap gap-1">
                  {repo.technologies.map((t) => (
                    <span key={t} className="text-[10px] font-mono text-ink-faint">
                      #{t}
                    </span>
                  ))}
                </div>
              </a>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
export default TheDeepEnd;

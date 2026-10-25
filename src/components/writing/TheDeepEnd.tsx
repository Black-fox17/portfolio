import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import { publishedArticles, openSourceReproductions } from '../../data/articles';

export const TheDeepEnd: React.FC = () => {
  const [activeView, setActiveView] = useState<'essays' | 'reproductions'>('essays');

  return (
    <section id="writing" className="scroll-mt-24">
      {/* Section Header */}
      <div className="space-y-3 mb-8">
        <div className="section-kicker">
          <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
          <span>04 / First Principles & Writing</span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div>
            <h2 className="text-2xl sm:text-4xl font-bold tracking-[-0.025em] text-ink font-editorial leading-[1.15]">
              The Deep End: First-Principles AI & Systems Research
            </h2>
            <p className="text-[15.5px] sm:text-base text-ink-muted leading-relaxed max-w-2xl mt-2">
              To truly understand how systems work, I implement them from scratch and explain the core mechanics, covering tokenization, neural network layers, attention mechanisms, and computer vision models.
            </p>
          </div>
          <Link
            to="/blog"
            className="editorial-btn-secondary !py-2 !px-3.5 shrink-0 self-start sm:self-auto"
          >
            <span>Open All Publications</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* Interactive Toggle: Papers vs. Code Reproductions (Dots instead of Icons) */}
      <div className="flex flex-wrap items-center gap-2 mb-6 p-1.5 rounded-lg bg-paper-100 border border-border w-fit">
        <button
          type="button"
          onClick={() => setActiveView('essays')}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-md text-xs font-mono transition-all cursor-pointer ${
            activeView === 'essays'
              ? 'bg-ink text-paper-50 font-semibold shadow-subtle'
              : 'text-ink-muted hover:text-ink'
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              activeView === 'essays' ? 'bg-accent-emerald' : 'bg-ink-faint'
            }`}
          />
          <span>Research Articles ({publishedArticles.length})</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveView('reproductions')}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-md text-xs font-mono transition-all cursor-pointer ${
            activeView === 'reproductions'
              ? 'bg-ink text-paper-50 font-semibold shadow-subtle'
              : 'text-ink-muted hover:text-ink'
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              activeView === 'reproductions' ? 'bg-accent-emerald' : 'bg-ink-faint'
            }`}
          />
          <span>From-Scratch PyTorch Code ({openSourceReproductions.length})</span>
        </button>
      </div>

      {/* View 1: Published Technical Essays */}
      {activeView === 'essays' ? (
        <div className="bg-white border border-border rounded-xl divide-y divide-border overflow-hidden shadow-subtle">
          {publishedArticles.map((article, idx) => (
            <Link
              key={article.id}
              to={`/blog/${article.slug}`}
              className="p-5 sm:p-6 block hover:bg-paper-100/50 transition-colors group"
            >
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                <div className="space-y-1.5 max-w-xl">
                  <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                    <span className="text-ink-muted font-bold">
                      0{idx + 1}
                    </span>
                    <span className="editorial-tag !py-0.5 text-[11px]">
                      {article.category}
                    </span>
                    <span className="text-ink-faint">· {article.readTime}</span>
                  </div>

                  <h3 className="text-lg sm:text-xl font-bold tracking-[-0.015em] text-ink font-editorial group-hover:text-accent-cobalt transition-colors pt-0.5">
                    {article.title}
                  </h3>

                  <p className="text-[14px] sm:text-[14.5px] text-ink-light leading-relaxed">
                    {article.excerpt}
                  </p>
                </div>

                <div className="text-xs font-mono text-ink-faint shrink-0 flex sm:flex-col items-center sm:items-end justify-between gap-2 pt-1">
                  <time dateTime={article.date}>
                    {new Date(article.date).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric'
                    })}
                  </time>
                  <span className="text-ink group-hover:text-accent-cobalt font-semibold inline-flex items-center gap-1">
                    <span>Read Paper</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            </Link>
          ))}
        </div>
      ) : (
        /* View 2: From-Scratch PyTorch Paper Reproductions */
        <div className="grid sm:grid-cols-2 gap-4">
          {openSourceReproductions.map((repo) => (
            <a
              key={repo.id}
              href={repo.githubUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="editorial-card p-5 flex flex-col justify-between space-y-4 bg-white group"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-accent-cobalt font-semibold flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
                    <span>{repo.category}</span>
                  </span>
                  <ArrowUpRight className="w-4 h-4 text-ink-faint group-hover:text-ink transition-colors" />
                </div>

                <h4 className="text-base font-bold text-ink font-editorial group-hover:text-accent-cobalt transition-colors">
                  {repo.title}
                </h4>

                <p className="text-[13.5px] text-ink-light leading-relaxed">
                  {repo.description}
                </p>
              </div>

              <div className="pt-3 border-t border-border-subtle flex flex-wrap gap-1.5">
                {repo.technologies.map((t) => (
                  <span
                    key={t}
                    className="text-[11px] font-mono px-2 py-0.5 rounded bg-paper-100 text-ink-muted"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </a>
          ))}
        </div>
      )}
    </section>
  );
};

export default TheDeepEnd;

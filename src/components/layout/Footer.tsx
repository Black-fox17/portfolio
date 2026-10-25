import React from 'react';
import { ArrowUp, ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export const Footer: React.FC = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="border-t border-border bg-white mt-16">
      <div className="max-w-4xl mx-auto px-5 sm:px-8 py-12">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6 pb-8 border-b border-border-subtle">
          <div className="space-y-1">
            <Link to="/" className="font-bold text-lg text-ink font-editorial hover:text-accent-cobalt transition-colors">
              Abdulsalam Oluwaseun Ayeleru
            </Link>
            <p className="text-xs text-ink-muted">
              AI Systems & Full-Stack Product Engineer · Lagos / Remote
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-ink-muted">
            <Link to="/" className="hover:text-ink transition-colors">Portfolio</Link>
            <Link to="/blog" className="hover:text-ink transition-colors">The Deep End</Link>
            <Link to="/research" className="hover:text-ink transition-colors">Research</Link>
            <a
              href="https://github.com/Black-fox17"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 hover:text-ink transition-colors"
            >
              <span>GitHub</span>
              <ArrowUpRight className="w-3 h-3" />
            </a>
            <a
              href="/assets/ayeleru_cv.pdf"
              target="_blank"
              rel="noopener noreferrer"
              download="ayeleru_cv.pdf"
              className="inline-flex items-center gap-1 hover:text-ink transition-colors"
            >
              <span>CV</span>
              <ArrowUpRight className="w-3 h-3" />
            </a>
          </div>
        </div>

        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-ink-faint font-mono">
          <p>© {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru.</p>
          <button
            onClick={scrollToTop}
            className="inline-flex items-center gap-1.5 hover:text-ink transition-colors cursor-pointer"
          >
            <span>Back to top</span>
            <ArrowUp className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </footer>
  );
};
export default Footer;

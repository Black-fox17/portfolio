import React from 'react';
import { Github, Mail, ArrowUp, FileText, ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export const Footer: React.FC = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="border-t border-border bg-white mt-20">
      <div className="max-w-5xl mx-auto px-5 sm:px-8 py-12 sm:py-16">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-8 pb-10 border-b border-border-subtle">
          {/* Identity & Scope */}
          <div className="space-y-2.5 max-w-sm">
            <h3 className="font-semibold text-base text-ink font-editorial">
              Abdulsalam Oluwaseun Ayeleru
            </h3>
            <p className="text-xs text-ink-muted leading-relaxed">
              AI Systems Engineer · Full-Stack Software Engineer · Product Engineer. Building production AI, distributed backends, and reliable multi-platform products.
            </p>
            <div className="flex items-center gap-2 pt-1 text-xs text-ink-faint font-mono">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-subtle" />
              <span>Available for engineering roles & systems contracts</span>
            </div>
          </div>

          {/* Quick Links Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-6 sm:gap-10 text-xs">
            <div className="space-y-2.5">
              <span className="font-mono uppercase tracking-wider text-ink-faint font-semibold text-[10px]">
                Navigation
              </span>
              <ul className="space-y-2 text-ink-muted">
                <li><a href="#work" className="hover:text-ink transition-colors">Selected Work</a></li>
                <li><a href="#experience" className="hover:text-ink transition-colors">Experience</a></li>
                <li><a href="#philosophy" className="hover:text-ink transition-colors">How I Build</a></li>
                <li><a href="#skills" className="hover:text-ink transition-colors">Technical Depth</a></li>
              </ul>
            </div>

            <div className="space-y-2.5">
              <span className="font-mono uppercase tracking-wider text-ink-faint font-semibold text-[10px]">
                Writing & Research
              </span>
              <ul className="space-y-2 text-ink-muted">
                <li><Link to="/blog" className="hover:text-ink transition-colors">The Deep End</Link></li>
                <li><a href="#the-deep-end" className="hover:text-ink transition-colors">Open Source Papers</a></li>
                <li><a href="#leadership" className="hover:text-ink transition-colors">Mentorship & Community</a></li>
              </ul>
            </div>

            <div className="space-y-2.5">
              <span className="font-mono uppercase tracking-wider text-ink-faint font-semibold text-[10px]">
                Connect
              </span>
              <ul className="space-y-2 text-ink-muted">
                <li>
                  <a
                    href="https://github.com/Black-fox17"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 hover:text-ink transition-colors"
                  >
                    <span>GitHub</span>
                    <ArrowUpRight className="w-3 h-3 text-ink-faint" />
                  </a>
                </li>
                <li>
                  <a
                    href="https://x.com/Ayeleru_Salam"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 hover:text-ink transition-colors"
                  >
                    <span>X / Twitter</span>
                    <ArrowUpRight className="w-3 h-3 text-ink-faint" />
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:ayeleru1234@gmail.com"
                    className="inline-flex items-center gap-1 hover:text-ink transition-colors"
                  >
                    <span>Email</span>
                    <ArrowUpRight className="w-3 h-3 text-ink-faint" />
                  </a>
                </li>
                <li>
                  <a
                    href="/assets/ayeleru_cv.pdf"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 hover:text-ink transition-colors"
                    download="ayeleru_cv.pdf"
                  >
                    <span>Download CV</span>
                    <FileText className="w-3 h-3 text-ink-faint" />
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-ink-faint font-mono">
          <p>© {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru. Engineered with React, TypeScript & Vite.</p>
          <button
            onClick={scrollToTop}
            className="inline-flex items-center gap-1.5 hover:text-ink transition-colors focus:outline-none"
            aria-label="Back to top of page"
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

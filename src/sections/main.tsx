import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowUpRight,
  Menu,
  X,
  ArrowUp
} from 'lucide-react';
import AboutSection from '../components/about/AboutSection';
import SelectedWork from '../components/work/SelectedWork';
import ExperienceTimeline from '../components/experience/ExperienceTimeline';
import TheDeepEnd from '../components/writing/TheDeepEnd';
import ContactSection from '../components/contact/ContactSection';

interface ChapterItem {
  id: string;
  number: string;
  label: string;
  subtitle: string;
}

const chapters: ChapterItem[] = [
  {
    id: 'story',
    number: '01',
    label: 'The Story & Craft',
    subtitle: 'Beyond the single-page resume'
  },
  {
    id: 'work',
    number: '02',
    label: 'Systems in Production',
    subtitle: 'Selected work & architecture'
  },
  {
    id: 'experience',
    number: '03',
    label: 'Career & Leadership',
    subtitle: 'Track record & mentorship'
  },
  {
    id: 'writing',
    number: '04',
    label: 'The Deep End',
    subtitle: 'First-principles ML & PyTorch'
  },
  {
    id: 'contact',
    number: '05',
    label: 'Initiate Contact',
    subtitle: 'Direct coordinates & message'
  }
];

export const Main: React.FC = () => {
  const [activeChapter, setActiveChapter] = useState<string>('story');
  const [copiedEmail, setCopiedEmail] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + window.innerHeight * 0.32;

      for (let i = chapters.length - 1; i >= 0; i--) {
        const el = document.getElementById(chapters[i].id);
        if (el && el.offsetTop <= scrollPosition) {
          setActiveChapter(chapters[i].id);
          break;
        }
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToChapter = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    setMobileMenuOpen(false);
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
      setActiveChapter(id);
    }
  };

  const handleCopyEmail = () => {
    navigator.clipboard.writeText('ayeleru1234@gmail.com');
    setCopiedEmail(true);
    setTimeout(() => setCopiedEmail(false), 2000);
  };

  const currentChapterObj = chapters.find((c) => c.id === activeChapter) || chapters[0];

  return (
    <div className="min-h-screen bg-paper text-ink selection:bg-ink selection:text-paper">
      {/* Mobile / Tablet Sticky Top Navigation Bar (< lg) */}
      <header className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-paper/95 backdrop-blur-md border-b border-border">
        <div className="px-5 sm:px-8 h-14 flex items-center justify-between gap-3">
          <a
            href="#story"
            onClick={scrollToChapter('story')}
            className="flex items-center gap-2.5 min-w-0"
          >
            <span className="font-editorial font-bold text-base text-ink truncate">
              Abdulsalam Ayeleru
            </span>
            <span className="text-border-strong">/</span>
            <span className="text-[11px] font-mono font-semibold text-ink-muted truncate">
              {currentChapterObj.number} · {currentChapterObj.label}
            </span>
          </a>

          <div className="flex items-center gap-2 shrink-0">
            <a
              href="/assets/ayeleru_cv.pdf"
              target="_blank"
              rel="noopener noreferrer"
              download="ayeleru_cv.pdf"
              className="px-2.5 py-1 rounded bg-ink text-paper-50 text-xs font-mono font-medium"
            >
              CV
            </a>
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-1.5 rounded bg-white border border-border text-ink"
              aria-label="Toggle chapter menu"
            >
              {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Mobile Chapter Jumper Drawer */}
        {mobileMenuOpen && (
          <div className="bg-white border-b border-border px-5 py-5 space-y-4 shadow-elevated">
            <div className="text-[11px] font-mono uppercase tracking-widest text-ink-faint font-semibold">
              Jump to Section
            </div>
            <nav className="space-y-1">
              {chapters.map((chap) => {
                const isActive = activeChapter === chap.id;
                return (
                  <a
                    key={chap.id}
                    href={`#${chap.id}`}
                    onClick={scrollToChapter(chap.id)}
                    className={`flex items-center justify-between p-2.5 rounded-lg text-xs transition-colors ${
                      isActive
                        ? 'bg-paper-100 text-ink font-bold'
                        : 'text-ink-muted hover:bg-paper-50 hover:text-ink'
                    }`}
                  >
                    <span className="flex items-center gap-2.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-ink" />
                      <span className="font-mono text-ink-faint">{chap.number}</span>
                      <span>{chap.label}</span>
                    </span>
                    <span className="text-[11px] text-ink-faint">{chap.subtitle}</span>
                  </a>
                );
              })}
            </nav>
            <div className="pt-3 border-t border-border-subtle flex flex-wrap gap-2">
              <Link
                to="/blog"
                onClick={() => setMobileMenuOpen(false)}
                className="editorial-btn-secondary flex-1 !py-2 !text-xs justify-center"
              >
                <span>The Deep End Blog</span>
                <ArrowUpRight className="w-3.5 h-3.5" />
              </Link>
              <Link
                to="/research"
                onClick={() => setMobileMenuOpen(false)}
                className="editorial-btn-secondary flex-1 !py-2 !text-xs justify-center"
              >
                <span>Research</span>
                <ArrowUpRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        )}
      </header>

      {/* Main Split-Screen Container */}
      <div className="max-w-[1380px] mx-auto px-5 sm:px-8 lg:px-12">
        <div className="lg:grid lg:grid-cols-12 lg:gap-10 xl:gap-14">
          {/* LEFT COLUMN: Sticky Profile, Identity & Story Navigation Rail */}
          <aside className="lg:col-span-5 lg:sticky lg:top-0 lg:h-screen lg:overflow-y-auto lg:py-6 xl:py-10 lg:flex lg:flex-col lg:justify-between pt-24 pb-10 border-b lg:border-b-0 border-border scrollbar-none">
            {/* Top Identity Block */}
            <div className="space-y-3.5 xl:space-y-4">
              {/* Availability Pill */}
              <div className="inline-flex flex-wrap sm:flex-nowrap items-center gap-2 sm:gap-2.5 px-3 py-1.5 rounded-md bg-paper-100 border border-border text-[11px] font-mono text-ink-light font-medium">
                <span className="w-2 h-2 rounded-full bg-accent-emerald animate-pulse-subtle shrink-0" />
                <span className="whitespace-nowrap">Available for Engineering Roles</span>
                <span className="text-border-strong hidden sm:inline">·</span>
                <span className="text-ink-muted whitespace-nowrap">Remote · Relocation Open</span>
              </div>

              {/* Name & Role */}
              <div className="space-y-1.5">
                <h1 className="text-3xl sm:text-4xl lg:text-[2.25rem] xl:text-[2.75rem] font-bold tracking-[-0.03em] text-ink font-editorial leading-[1.06]">
                  Abdulsalam Ayeleru
                </h1>
                <p className="text-xs sm:text-sm font-mono uppercase tracking-wider font-semibold text-accent-cobalt">
                  AI Systems & Full-Stack Product Engineer
                </p>
              </div>

              {/* Concise Story Hook */}
              <p className="text-[14px] sm:text-[14.5px] text-ink-muted leading-[1.6] max-w-md font-normal">
                I engineer production AI, resilient financial backends, and multi-platform software—from mathematical first principles to the Apple App Store and Google Play.
              </p>

              {/* Light Editorial Metrics Strip (Designer Recommendation: warm surface, fine borders, clean typography) */}
              <div className="bg-paper-100/60 border border-border rounded-xl p-3.5 sm:p-4 w-full max-w-lg">
                <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-ink-muted/80 font-semibold block mb-2.5">
                  Engineering at a Glance
                </span>
                <div className="grid grid-cols-3 divide-x divide-border">
                  <div className="pr-2.5 sm:pr-3">
                    <div className="text-2xl sm:text-[1.75rem] font-bold font-editorial tracking-tight text-ink leading-none">
                      3
                    </div>
                    <div className="text-[11.5px] sm:text-xs font-semibold text-ink mt-1.5 leading-snug whitespace-nowrap">
                      Apps shipped
                    </div>
                    <div className="text-[10px] sm:text-[11px] font-mono text-ink-muted mt-0.5 leading-snug whitespace-nowrap">
                      iOS & Android
                    </div>
                  </div>
                  <div className="px-2.5 sm:px-3">
                    <div className="text-2xl sm:text-[1.75rem] font-bold font-editorial tracking-tight text-ink leading-none">
                      AI
                    </div>
                    <div className="text-[11.5px] sm:text-xs font-semibold text-ink mt-1.5 leading-snug whitespace-nowrap">
                      In production
                    </div>
                    <div className="text-[10px] sm:text-[11px] font-mono text-ink-muted mt-0.5 leading-snug whitespace-nowrap">
                      Vision & Edge
                    </div>
                  </div>
                  <div className="pl-2.5 sm:pl-3">
                    <div className="text-2xl sm:text-[1.75rem] font-bold font-editorial tracking-tight text-ink leading-none">
                      8
                    </div>
                    <div className="text-[11.5px] sm:text-xs font-semibold text-ink mt-1.5 leading-snug whitespace-nowrap">
                      Research articles
                    </div>
                    <div className="text-[10px] sm:text-[11px] font-mono text-ink-muted mt-0.5 leading-snug whitespace-nowrap">
                      First-principles
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Middle Block: Interactive Chapter Spy Navigation (Desktop) */}
            <nav
              className="hidden lg:block my-3 xl:my-4 space-y-0.5"
              aria-label="Portfolio Story Chapters"
            >
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-ink-faint font-semibold mb-2">
                Index
              </div>
              {chapters.map((chapter) => {
                const isActive = activeChapter === chapter.id;
                return (
                  <a
                    key={chapter.id}
                    href={`#${chapter.id}`}
                    onClick={scrollToChapter(chapter.id)}
                    className={`group flex items-center gap-3 py-1.5 transition-all ${
                      isActive ? 'text-ink' : 'text-ink-faint hover:text-ink-light'
                    }`}
                  >
                    <span
                      className={`w-2 h-2 rounded-full transition-all duration-200 ${
                        isActive
                          ? 'bg-accent-cobalt scale-110'
                          : 'bg-border-strong group-hover:bg-ink-muted'
                      }`}
                    />
                    <span className="text-xs font-mono font-semibold text-ink-muted">
                      {chapter.number}
                    </span>
                    <span
                      className={`text-sm tracking-tight transition-colors ${
                        isActive ? 'font-bold text-ink' : 'font-medium'
                      }`}
                    >
                      {chapter.label}
                    </span>
                    {isActive && (
                      <span className="text-[10.5px] font-mono text-ink-faint hidden 2xl:inline-block ml-auto whitespace-nowrap">
                        {chapter.subtitle}
                      </span>
                    )}
                  </a>
                );
              })}
            </nav>

            {/* Bottom Coordinates & Direct Links */}
            <div className="space-y-2.5 pt-4 lg:pt-0">
              {/* Primary Action Buttons */}
              <div className="flex flex-wrap items-center gap-2">
                <a
                  href="/assets/ayeleru_cv.pdf"
                  target="_blank"
                  rel="noopener noreferrer"
                  download="ayeleru_cv.pdf"
                  className="editorial-btn-primary !py-2 !px-3.5"
                >
                  <span>Download CV</span>
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </a>
                <Link
                  to="/blog"
                  className="editorial-btn-secondary !py-2 !px-3.5"
                >
                  <span>The Deep End</span>
                  <ArrowUpRight className="w-3 h-3 text-ink-faint" />
                </Link>
                <Link
                  to="/research"
                  className="editorial-btn-secondary !py-2 !px-3"
                >
                  <span>Research</span>
                </Link>
              </div>

              {/* Direct Email Copy + Socials Row */}
              <div className="flex flex-wrap items-center gap-2.5 pt-1 text-xs font-mono text-ink-muted">
                <button
                  type="button"
                  onClick={handleCopyEmail}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-paper-100 border border-border hover:border-ink/40 text-ink transition-colors cursor-pointer"
                  title="Click to copy email address"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
                  <span>ayeleru1234@gmail.com</span>
                  <span className="text-[10.5px] px-1.5 py-0.5 rounded bg-white border border-border text-ink-muted">
                    {copiedEmail ? 'Copied' : 'Copy'}
                  </span>
                </button>

                <a
                  href="https://github.com/Black-fox17"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-paper-100 border border-border hover:border-ink/40 text-ink transition-colors"
                >
                  <span>GitHub</span>
                  <ArrowUpRight className="w-3 h-3 text-ink-faint" />
                </a>

                <a
                  href="https://x.com/Ayeleru_Salam"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-paper-100 border border-border hover:border-ink/40 text-ink transition-colors"
                >
                  <span>X</span>
                  <ArrowUpRight className="w-3 h-3 text-ink-faint" />
                </a>
              </div>
            </div>
          </aside>

          {/* RIGHT COLUMN: Scrollable Narrative & Case Studies Feed */}
          <main className="lg:col-span-7 py-12 sm:py-16 lg:py-16 lg:pl-10 lg:border-l lg:border-border space-y-24 sm:space-y-28">
            <AboutSection />
            <div className="h-px bg-border" />
            <SelectedWork />
            <div className="h-px bg-border" />
            <ExperienceTimeline />
            <div className="h-px bg-border" />
            <TheDeepEnd />
            <div className="h-px bg-border" />
            <ContactSection />

            {/* Clean Colophon Footer */}
            <footer className="pt-10 border-t border-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs font-mono text-ink-faint">
              <div>
                © {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru · Built with React & TypeScript.
              </div>
              <button
                type="button"
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                className="inline-flex items-center gap-1.5 text-ink-muted hover:text-ink transition-colors cursor-pointer"
              >
                <span>Back to top</span>
                <ArrowUp className="w-3.5 h-3.5" />
              </button>
            </footer>
          </main>
        </div>
      </div>
    </div>
  );
};

export default Main;

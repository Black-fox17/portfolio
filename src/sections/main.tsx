import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
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
    subtitle: 'First principles & architectural approach'
  },
  {
    id: 'work',
    number: '02',
    label: 'Systems in Production',
    subtitle: 'Shipped apps, backends & ML'
  },
  {
    id: 'experience',
    number: '03',
    label: 'Career & Leadership',
    subtitle: 'Track record & engineering impact'
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
    subtitle: 'Direct coordinates & inquiry'
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
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="min-h-screen bg-white text-neutral-900 selection:bg-neutral-900 selection:text-white"
    >
      {/* Mobile / Tablet Top Navigation Header (< lg) */}
      <header className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-md border-b border-neutral-100">
        <div className="px-5 sm:px-8 h-14 flex items-center justify-between gap-3">
          <a
            href="#story"
            onClick={scrollToChapter('story')}
            className="flex items-center gap-2 min-w-0"
          >
            <span className="font-bold text-base text-neutral-900 tracking-tight">
              Abdulsalam.
            </span>
            <span className="text-neutral-300">/</span>
            <span className="text-xs font-mono text-neutral-500 truncate">
              {currentChapterObj.number} · {currentChapterObj.label}
            </span>
          </a>

          <div className="flex items-center gap-2 shrink-0">
            <a
              href="/assets/ayeleru_cv.pdf"
              target="_blank"
              rel="noopener noreferrer"
              download="ayeleru_cv.pdf"
              className="px-2.5 py-1 rounded bg-neutral-900 text-white text-xs font-mono font-medium"
            >
              CV
            </a>
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-1.5 rounded bg-white border border-neutral-200 text-neutral-700 hover:text-neutral-900"
              aria-label="Toggle chapter menu"
            >
              {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Mobile Chapter Drawer */}
        {mobileMenuOpen && (
          <div className="bg-white border-b border-neutral-200 px-5 py-5 space-y-4 shadow-elevated">
            <div className="text-[11px] font-mono uppercase tracking-widest text-neutral-400 font-semibold">
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
                        ? 'bg-neutral-100 text-neutral-900 font-semibold'
                        : 'text-neutral-600 hover:bg-neutral-50 hover:text-neutral-900'
                    }`}
                  >
                    <span className="flex items-center gap-2.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-neutral-900' : 'bg-neutral-300'}`} />
                      <span className="font-mono text-neutral-400">{chap.number}</span>
                      <span>{chap.label}</span>
                    </span>
                    <span className="text-[11px] text-neutral-400">{chap.subtitle}</span>
                  </a>
                );
              })}
            </nav>
            <div className="pt-3 border-t border-neutral-100 flex flex-wrap gap-2">
              <Link
                to="/blog"
                onClick={() => setMobileMenuOpen(false)}
                className="editorial-btn-secondary flex-1 !py-2 !text-xs justify-center"
              >
                <span>The Deep End</span>
                <ArrowUpRight className="w-3.5 h-3.5 opacity-60" />
              </Link>
            </div>
          </div>
        )}
      </header>

      {/* Main Dual-Page Container */}
      <div className="max-w-[1380px] mx-auto px-5 sm:px-8 lg:px-12">
        <div className="lg:grid lg:grid-cols-12 lg:gap-10 xl:gap-14">
          {/* LEFT COLUMN: Sticky Identity & Chapter Rail */}
          <aside className="lg:col-span-5 lg:sticky lg:top-0 lg:h-screen lg:overflow-y-auto lg:py-8 xl:py-12 lg:flex lg:flex-col lg:justify-between pt-20 sm:pt-24 pb-10 border-b lg:border-b-0 border-neutral-100 scrollbar-none">
            {/* Top Identity Block */}
            <div className="space-y-4 xl:space-y-5">
              {/* Bold Name Header */}
              <div className="space-y-1.5">
                <h1 className="text-3xl sm:text-4xl lg:text-[2.6rem] font-bold tracking-tight text-neutral-900 leading-tight">
                  Abdulsalam.
                </h1>
                <p className="text-xs sm:text-sm font-mono text-neutral-500 uppercase tracking-wider font-medium">
                  AI Systems &amp; Full-Stack Product Engineer
                </p>
              </div>

              {/* Concise Story Hook in user's words */}
              <p className="text-sm sm:text-[15px] text-neutral-600 leading-relaxed max-w-md font-normal">
                I build practical AI systems, dependable backends, and mobile apps, taking products from initial design all the way to the Apple App Store and Google Play.
              </p>
            </div>

            {/* Middle: Interactive Chapter Navigation Tracker */}
            <nav className="hidden lg:block space-y-1.5 my-6 xl:my-8" aria-label="Section Navigation">
              <div className="text-[10px] font-mono uppercase tracking-widest text-neutral-400 font-semibold mb-3">
                Index &amp; Navigation
              </div>
              {chapters.map((chapter) => {
                const isActive = activeChapter === chapter.id;
                return (
                  <a
                    key={chapter.id}
                    href={`#${chapter.id}`}
                    onClick={scrollToChapter(chapter.id)}
                    className={`group flex items-center gap-3.5 py-2 px-3 rounded-lg text-xs font-mono transition-colors ${
                      isActive
                        ? 'bg-neutral-100 text-neutral-900 font-semibold'
                        : 'text-neutral-500 hover:text-neutral-900 hover:bg-neutral-50'
                    }`}
                  >
                    <span
                      className={`w-1.5 h-1.5 rounded-full transition-colors ${
                        isActive ? 'bg-neutral-900' : 'bg-neutral-300 group-hover:bg-neutral-500'
                      }`}
                    />
                    <span className="font-mono text-neutral-400">
                      {chapter.number}
                    </span>
                    <span className="text-sm tracking-tight">
                      {chapter.label}
                    </span>
                    {isActive && (
                      <span className="text-[10.5px] font-mono text-neutral-400 hidden xl:inline-block ml-auto whitespace-nowrap">
                        {chapter.subtitle}
                      </span>
                    )}
                  </a>
                );
              })}
            </nav>

            {/* Bottom: Action Coordinates & Links */}
            <div className="space-y-3 pt-4 lg:pt-0">
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
                  <ArrowUpRight className="w-3.5 h-3.5 opacity-70" />
                </a>
                <Link
                  to="/blog"
                  className="editorial-btn-secondary !py-2 !px-3.5"
                >
                  <span>The Deep End</span>
                  <ArrowUpRight className="w-3 h-3 text-neutral-400" />
                </Link>
              </div>

              {/* Direct Email Copy + Socials Row */}
              <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-neutral-600 pt-1">
                <button
                  type="button"
                  onClick={handleCopyEmail}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-neutral-100 border border-neutral-200/80 hover:border-neutral-400 text-neutral-900 transition-colors cursor-pointer"
                  title="Click to copy email address"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-neutral-900" />
                  <span>ayeleru1234@gmail.com</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-white border border-neutral-200 text-neutral-500">
                    {copiedEmail ? 'Copied' : 'Copy'}
                  </span>
                </button>

                <a
                  href="https://github.com/Black-fox17"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-neutral-100 border border-neutral-200/80 hover:border-neutral-400 text-neutral-900 transition-colors"
                >
                  <span>GitHub</span>
                  <ArrowUpRight className="w-3 h-3 opacity-60" />
                </a>

                <a
                  href="https://x.com/Ayeleru_Salam"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-neutral-100 border border-neutral-200/80 hover:border-neutral-400 text-neutral-900 transition-colors"
                >
                  <span>Twitter</span>
                  <ArrowUpRight className="w-3 h-3 opacity-60" />
                </a>

                <a
                  href="https://www.linkedin.com/in/abdulsalam-ayeleru-384976219/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-neutral-100 border border-neutral-200/80 hover:border-neutral-400 text-neutral-900 transition-colors"
                >
                  <span>LinkedIn</span>
                  <ArrowUpRight className="w-3 h-3 opacity-60" />
                </a>
              </div>
            </div>
          </aside>

          {/* RIGHT COLUMN: Narrative & Projects Feed */}
          <main className="lg:col-span-7 py-10 sm:py-14 lg:py-12 lg:pl-10 lg:border-l lg:border-neutral-100 space-y-12 sm:space-y-14">
            <div className="pb-12 sm:pb-14 border-b border-neutral-100">
              <AboutSection />
            </div>
            <div className="pb-12 sm:pb-14 border-b border-neutral-100">
              <SelectedWork />
            </div>
            <div className="pb-12 sm:pb-14 border-b border-neutral-100">
              <ExperienceTimeline />
            </div>
            <div className="pb-12 sm:pb-14 border-b border-neutral-100">
              <TheDeepEnd />
            </div>
            <div>
              <ContactSection />
            </div>

            {/* Clean Colophon Footer */}
            <footer className="pt-6 border-t border-neutral-100 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs font-mono text-neutral-400">
              <div>
                © {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru.
              </div>
              <button
                type="button"
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                className="inline-flex items-center gap-1.5 text-neutral-500 hover:text-neutral-900 transition-colors cursor-pointer"
              >
                <span>Back to top</span>
                <ArrowUp className="w-3.5 h-3.5" />
              </button>
            </footer>
          </main>
        </div>
      </div>
    </motion.div>
  );
};

export default Main;

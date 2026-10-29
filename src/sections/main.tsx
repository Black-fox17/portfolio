import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowUpRight,
  ArrowDown,
  Menu,
  X,
  ArrowUp
} from 'lucide-react';
import NeuralParticleCanvas from '../components/hero/NeuralParticleCanvas';
import AboutSection from '../components/about/AboutSection';
import SelectedWork from '../components/work/SelectedWork';
import ExperienceTimeline from '../components/experience/ExperienceTimeline';
import TheDeepEnd from '../components/writing/TheDeepEnd';
import ContactSection from '../components/contact/ContactSection';

interface ChapterItem {
  id: string;
  label: string;
  subtitle: string;
  morphStage: number;
}

const chapters: ChapterItem[] = [
  {
    id: 'story',
    label: 'The Story & Craft',
    subtitle: 'First principles & architectural approach',
    morphStage: 0
  },
  {
    id: 'work',
    label: 'Systems in Production',
    subtitle: 'Shipped apps, backends & ML',
    morphStage: 2
  },
  {
    id: 'experience',
    label: 'Career & Leadership',
    subtitle: 'Track record & engineering impact',
    morphStage: 1
  },
  {
    id: 'writing',
    label: 'The Deep End',
    subtitle: 'First-principles ML & PyTorch',
    morphStage: 0
  },
  {
    id: 'contact',
    label: 'Initiate Contact',
    subtitle: 'Direct coordinates & inquiry',
    morphStage: 2
  }
];

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export const Main: React.FC = () => {
  const [activeChapter, setActiveChapter] = useState<string>('story');
  const [copiedEmail, setCopiedEmail] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  // 0 = Hero right side, 1 = Docked in sticky left column
  const [scrollShift, setScrollShift] = useState<number>(0);
  const [isDesktop, setIsDesktop] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth >= 1024 : true
  );

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px)');
    const onChange = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  useEffect(() => {
    let ticking = false;
    let sectionTops: { id: string; top: number }[] = [];

    const updateOffsets = () => {
      sectionTops = chapters.map((c) => {
        const el = document.getElementById(c.id);
        if (!el) return { id: c.id, top: 0 };
        const box = el.getBoundingClientRect();
        return { id: c.id, top: box.top + window.scrollY };
      });
    };

    updateOffsets();
    window.addEventListener('resize', updateOffsets, { passive: true });

    const handleScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          const scrollY = window.scrollY;
          const vh = window.innerHeight || 800;

          if (isDesktop) {
            const transitionRange = vh * 0.68;
            const raw = Math.min(1, Math.max(0, scrollY / transitionRange));
            const eased = easeInOutCubic(raw);
            setScrollShift((prev) => (Math.abs(prev - eased) > 0.005 ? eased : prev));
          }

          const scrollPosition = scrollY + vh * 0.35;
          for (let i = sectionTops.length - 1; i >= 0; i--) {
            if (sectionTops[i].top <= scrollPosition) {
              const targetId = sectionTops[i].id;
              setActiveChapter((prev) => (prev !== targetId ? targetId : prev));
              break;
            }
          }
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', updateOffsets);
    };
  }, [isDesktop]);

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

  // Horizontal translation percentage of the 5/12-width sticky visual container:
  // At scrollShift = 0 (Hero): sits on the Right side (x = 134% of its own width)
  // At scrollShift = 1 (Chapters): docks into the Left Column (x = 0%)
  const translateXPercent = (1 - scrollShift) * 134;
  const scaleValue = 1 + (1 - scrollShift) * 0.04;
  const dockedUiOpacity = Math.max(0, Math.min(1, (scrollShift - 0.55) / 0.45));

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
              {currentChapterObj.label}
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

      {/* Unified Container with Scroll-Choreographed Particle Stage */}
      <div className="max-w-[1380px] mx-auto px-5 sm:px-8 lg:px-12 relative">
        {/*
          DESKTOP STICKY VISUAL STAGE (lg+):
          Starts on the Right Column in the Hero, then smoothly glides across to dock
          into the Sticky Left Column while the user scrolls through the Right Column sections.
        */}
        <div className="hidden lg:flex lg:sticky lg:top-0 lg:h-screen lg:w-full lg:items-center pointer-events-none z-20">
          <div
            style={{
              transform: `translate3d(${translateXPercent.toFixed(2)}%, 0, 0) scale(${scaleValue.toFixed(3)})`,
              willChange: 'transform'
            }}
            className="w-[42%] flex flex-col justify-between py-8 xl:py-10 pointer-events-auto"
          >
            {/* Compact Top Header (Fades in once docked on the Left Column) */}
            <div
              style={{
                opacity: dockedUiOpacity,
                transform: `translateY(${((1 - dockedUiOpacity) * -8).toFixed(1)}px)`,
                pointerEvents: dockedUiOpacity > 0.4 ? 'auto' : 'none'
              }}
              className="flex items-center justify-between px-2 pb-1 transition-opacity duration-150"
            >
              <button
                type="button"
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                className="flex items-center gap-2 text-left group cursor-pointer"
              >
                <span className="font-bold text-base text-neutral-900 tracking-tight group-hover:text-neutral-600 transition-colors">
                  Abdulsalam.
                </span>
                <span className="text-neutral-300">/</span>
                <span className="text-xs font-mono text-neutral-500">
                  {currentChapterObj.label}
                </span>
              </button>
            </div>

            {/* The Living Neural-to-Product Particle Cloud (Desktop Only) */}
            {isDesktop && (
              <NeuralParticleCanvas
                scrollProgress={scrollShift}
                chapterStageHint={currentChapterObj.morphStage}
                heightClass="h-[460px] xl:h-[510px]"
              />
            )}

            {/* Compact Chapter Navigation Bar (Fades in below the animation when docked on the Left) */}
            <div
              style={{
                opacity: dockedUiOpacity,
                transform: `translateY(${((1 - dockedUiOpacity) * 6).toFixed(1)}px)`,
                pointerEvents: dockedUiOpacity > 0.4 ? 'auto' : 'none'
              }}
              className="mt-2 flex flex-col items-center gap-2 transition-opacity duration-150"
            >
              <nav
                className="inline-flex flex-wrap items-center justify-center gap-1 p-1 rounded-xl bg-neutral-50/90 border border-neutral-200/80"
                aria-label="Chapter Navigation"
              >
                {chapters.map((chapter) => {
                  const isActive = activeChapter === chapter.id;
                  return (
                    <a
                      key={chapter.id}
                      href={`#${chapter.id}`}
                      onClick={scrollToChapter(chapter.id)}
                      className={`flex items-center px-2.5 py-1.5 rounded-lg text-[11px] font-mono transition-all ${
                        isActive
                          ? 'bg-neutral-900 text-white font-medium shadow-subtle'
                          : 'text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100'
                      }`}
                    >
                      <span>{chapter.label}</span>
                    </a>
                  );
                })}
              </nav>
            </div>
          </div>
        </div>

        {/* PAGE FLOW CONTENT (Pulled up by -100vh on lg+ to align with the sticky layer) */}
        <div className="lg:-mt-[100vh] relative z-10">
          {/* =================================================================
              SECTION 0: SPLIT HERO STAGE
              Left: Signature Identity & Copy | Right: Particle Stage (at scroll=0)
             ================================================================= */}
          <section className="pt-20 pb-12 sm:pt-24 sm:pb-16 lg:py-0 lg:min-h-screen flex items-center border-b border-neutral-100">
            <div className="w-full lg:grid lg:grid-cols-12 lg:gap-10 xl:gap-14 items-center">
              {/* Left Column: Exact Signature Identity & Bio */}
              <div className="lg:col-span-5 space-y-6">
                <div className="space-y-2">
                  <h1 className="text-4xl sm:text-5xl xl:text-[3.6rem] font-bold tracking-tight text-neutral-900 leading-[1.08]">
                    Abdulsalam.
                  </h1>
                  <p className="text-xs sm:text-sm font-mono text-neutral-500 uppercase tracking-wider font-medium">
                    AI Systems &amp; Full-Stack Product Engineer
                  </p>
                </div>

                <p className="text-base sm:text-[17px] text-neutral-600 leading-relaxed max-w-md font-normal">
                  I build practical AI systems, dependable backends, and mobile apps, taking products from design all the way to production servers, web apps, the Apple App Store, and Google Play.
                </p>

                {/* Primary Action Buttons */}
                <div className="pt-1 flex flex-wrap items-center gap-2.5">
                  <a
                    href="#work"
                    onClick={scrollToChapter('work')}
                    className="editorial-btn-primary !py-2.5 !px-4"
                  >
                    <span>See my work</span>
                    <ArrowDown className="w-3.5 h-3.5 opacity-80" />
                  </a>

                  <a
                    href="/assets/ayeleru_cv.pdf"
                    target="_blank"
                    rel="noopener noreferrer"
                    download="ayeleru_cv.pdf"
                    className="editorial-btn-secondary !py-2.5 !px-4"
                  >
                    <span>Download CV</span>
                    <ArrowUpRight className="w-3.5 h-3.5 text-neutral-400" />
                  </a>

                  <Link
                    to="/blog"
                    className="editorial-btn-secondary !py-2.5 !px-4"
                  >
                    <span>The Deep End</span>
                    <ArrowUpRight className="w-3.5 h-3.5 text-neutral-400" />
                  </Link>
                </div>

                {/* Direct Email Copy + Socials Row */}
                <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-neutral-600 pt-2">
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

              {/* Right Column on Mobile/Tablet (< lg): Inline Particle Canvas */}
              {!isDesktop && (
                <div className="mt-8 lg:hidden">
                  <NeuralParticleCanvas heightClass="h-[390px] sm:h-[450px]" />
                </div>
              )}

              {/* Right Column on Desktop (lg+): Reserved stage where the Sticky Canvas sits at scroll=0 */}
              <div className="hidden lg:block lg:col-span-7 pointer-events-none" aria-hidden="true" />
            </div>
          </section>

          {/* =================================================================
              SECTIONS 01–05: DUAL-COLUMN FEED
              Left Column: Reserved for the Sticky Particle Stage (glided from right)
              Right Column: Scrollable Narrative, Projects, Experience, Writing, Contact
             ================================================================= */}
          <div className="lg:grid lg:grid-cols-12 lg:gap-10 xl:gap-14">
            {/* Left Column Spacer on Desktop (The Sticky Visual Stage occupies this space) */}
            <aside className="hidden lg:block lg:col-span-5 pointer-events-none" aria-hidden="true" />

            {/* Right Column: Scrollable Narrative & Projects Feed */}
            <main className="lg:col-span-7 py-12 sm:py-16 lg:py-20 lg:pl-10 lg:border-l lg:border-neutral-100 space-y-12 sm:space-y-14">
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
      </div>
    </motion.div>
  );
};

export default Main;

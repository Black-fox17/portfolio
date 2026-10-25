import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, ArrowUpRight, X, ChevronRight } from 'lucide-react';
import { Project } from '../../types/portfolio';
import { projects } from '../../data/projects';
import { ProjectLinkBadge } from '../common/ProjectLinkBadge';

interface CaseStudyModalProps {
  project: Project | null;
  onClose: () => void;
  onSelectProject?: (project: Project) => void;
}

export const CaseStudyModal: React.FC<CaseStudyModalProps> = ({
  project,
  onClose,
  onSelectProject,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (project) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.body.style.overflow = 'unset';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [project, onClose]);

  const currentIndex = project
    ? projects.findIndex((p) => p.id === project.id)
    : -1;
  const nextProject =
    currentIndex !== -1 && currentIndex < projects.length - 1
      ? projects[currentIndex + 1]
      : projects[0];

  return (
    <AnimatePresence>
      {project && (
        <div className="fixed inset-0 z-50 flex justify-end">
          {/* Smooth Dimmed Backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.28, ease: 'easeInOut' }}
            className="fixed inset-0 bg-ink/40 backdrop-blur-[2px]"
            onClick={onClose}
            aria-hidden="true"
          />

          {/* Smooth Spring-Loaded Slide-Over Panel */}
          <motion.aside
            key="case-study-drawer"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{
              type: 'spring',
              damping: 30,
              stiffness: 300,
              mass: 0.85,
            }}
            className="relative w-full lg:w-[64%] xl:w-[60%] h-full bg-paper border-l-2 border-ink shadow-2xl flex flex-col z-50 overflow-hidden"
            role="dialog"
            aria-modal="true"
            aria-labelledby="case-study-title"
          >
            {/* Sticky Top Navigation Bar */}
            <header className="px-5 sm:px-8 py-4 border-b border-border bg-paper/95 backdrop-blur-md sticky top-0 z-20 flex items-center justify-between gap-4">
              <button
                type="button"
                onClick={onClose}
                className="inline-flex items-center gap-2 text-xs font-mono font-semibold text-ink hover:text-accent-cobalt transition-colors cursor-pointer py-1"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>Back to Selected Work</span>
              </button>

              <div className="flex items-center gap-3">
                <span className="text-[11px] font-mono text-ink-faint hidden sm:inline-block">
                  Case Study 0{currentIndex + 1} / 0{projects.length}
                </span>
                <button
                  type="button"
                  onClick={onClose}
                  className="p-1.5 rounded-lg bg-paper-100 hover:bg-paper-200 border border-border text-ink transition-colors cursor-pointer"
                  aria-label="Close Case Study"
                  title="Close (Esc)"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </header>

            {/* Content Container with Smooth Cross-Fade on Project Change */}
            <motion.div
              key={project.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.28, ease: 'easeOut' }}
              className="flex-1 overflow-y-auto px-5 sm:px-8 lg:px-10 py-8 sm:py-10 space-y-12"
            >
              {/* 1. Monograph Hero Header */}
              <div className="space-y-4 border-b border-border pb-8">
                <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                  <span className="editorial-tag !py-0.5">
                    {project.category}
                  </span>
                  <span className="text-border-strong">·</span>
                  <span className="text-ink font-medium inline-flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald animate-pulse-subtle" />
                    <span>{project.status}</span>
                  </span>
                  <span className="text-border-strong">·</span>
                  <span className="text-ink-faint">{project.period}</span>
                </div>

                <div className="space-y-2">
                  <h1
                    id="case-study-title"
                    className="text-3xl sm:text-4xl lg:text-[2.6rem] font-bold tracking-[-0.03em] text-ink font-editorial leading-[1.08]"
                  >
                    {project.title}
                  </h1>
                  <p className="text-base sm:text-lg font-semibold text-accent-cobalt">
                    {project.subtitle}
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-x-6 gap-y-2 pt-1 text-xs font-mono text-ink-muted">
                  <div>
                    <span className="text-ink-faint">Role: </span>
                    <span className="text-ink font-semibold">{project.role}</span>
                  </div>
                  <div>
                    <span className="text-ink-faint">Execution: </span>
                    <span className="text-ink font-semibold">End-to-End Delivery</span>
                  </div>
                </div>

                {/* Live Releases & Repository Access Bar */}
                {project.links && project.links.length > 0 && (
                  <div className="pt-4 flex flex-wrap items-center gap-2">
                    <span className="text-[11px] font-mono uppercase tracking-wider text-ink-faint font-semibold mr-1 block sm:inline">
                      Releases:
                    </span>
                    {project.links.map((link) => (
                      <ProjectLinkBadge key={link.label} link={link} variant="dark" />
                    ))}
                  </div>
                )}
              </div>

              {/* 2. Executive Overview (Structured Callout Box) */}
              <section className="space-y-3">
                <div className="section-kicker">
                  <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
                  <span>01 / System Overview & Scope</span>
                </div>
                <div className="bg-white border-2 border-ink rounded-xl p-6 sm:p-7 shadow-subtle space-y-3">
                  <p className="text-base sm:text-[17px] text-ink leading-[1.75] font-normal">
                    {project.caseStudy.overview}
                  </p>
                  <div className="pt-3 border-t border-border-subtle flex flex-wrap gap-2 text-xs font-mono text-ink-muted">
                    <span className="font-semibold text-ink">Core Impact:</span>
                    <span>{project.summary}</span>
                  </div>
                </div>
              </section>

              {/* 3. The Operational Challenge vs Solution (Visual Comparison Grid) */}
              <section className="space-y-4">
                <div className="section-kicker">
                  <span className="w-2 h-2 rounded-full bg-ink" />
                  <span>02 / Problem vs. Engineered Solution</span>
                </div>

                <div className="grid sm:grid-cols-2 gap-4">
                  {/* Problem Side */}
                  <div className="bg-paper-100/90 border border-border rounded-xl p-5 sm:p-6 space-y-2.5">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-ink-muted" />
                      <h3 className="text-xs font-mono uppercase tracking-wider text-ink-muted font-bold">
                        The Failure Mode / Challenge
                      </h3>
                    </div>
                    <p className="text-[14.5px] text-ink-light leading-[1.75]">
                      {project.caseStudy.problem}
                    </p>
                  </div>

                  {/* Solution Side */}
                  <div className="bg-white border border-border-strong rounded-xl p-5 sm:p-6 space-y-2.5 shadow-subtle">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
                      <h3 className="text-xs font-mono uppercase tracking-wider text-accent-cobalt font-bold">
                        The Architectural Remedy
                      </h3>
                    </div>
                    <p className="text-[14.5px] text-ink-light leading-[1.75]">
                      {project.caseStudy.complexityAndFailures}
                    </p>
                  </div>
                </div>
              </section>

              {/* 4. System Topology & Layer Breakdown (Architectural Diagram) */}
              {project.architectureDiagram && (
                <section className="space-y-4">
                  <div className="section-kicker">
                    <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
                    <span>03 / System Topology & Boundary Stack</span>
                  </div>

                  <div className="bg-white border border-border rounded-xl p-6 shadow-subtle space-y-4">
                    <div className="grid sm:grid-cols-2 gap-4">
                      {project.architectureDiagram.layers.map((layer, idx) => (
                        <div
                          key={layer.name}
                          className="p-4 rounded-lg bg-paper/80 border border-border space-y-2.5"
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-mono font-bold text-ink flex items-center gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
                              <span>{layer.name}</span>
                            </span>
                            <span className="text-[10px] font-mono text-ink-faint">
                              Layer 0{idx + 1}
                            </span>
                          </div>
                          <ul className="space-y-1.5">
                            {layer.items.map((item) => (
                              <li
                                key={item}
                                className="text-xs font-mono text-ink-muted pl-3 border-l-2 border-border-strong"
                              >
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              )}

              {/* 5. Architectural Specifications (Clean Numbered Rows) */}
              <section className="space-y-4">
                <div className="section-kicker">
                  <span className="w-2 h-2 rounded-full bg-ink" />
                  <span>04 / Architectural Specifications</span>
                </div>

                <div className="bg-white border border-border rounded-xl divide-y divide-border overflow-hidden shadow-subtle">
                  {project.caseStudy.architecture.map((arch, idx) => (
                    <div key={idx} className="p-5 flex items-start gap-4">
                      <span className="text-xs font-mono font-bold text-accent-cobalt mt-0.5">
                        0{idx + 1}
                      </span>
                      <p className="text-[14.5px] text-ink-light leading-[1.75]">
                        {arch}
                      </p>
                    </div>
                  ))}
                </div>
              </section>

              {/* 6. Key Engineering Highlights (Numbered Spec Cards) */}
              <section className="space-y-4">
                <div className="section-kicker">
                  <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
                  <span>05 / What I Personally Engineered</span>
                </div>

                <div className="grid sm:grid-cols-2 gap-3.5">
                  {project.caseStudy.engineeringHighlights.map((hl, idx) => (
                    <div
                      key={idx}
                      className="bg-white border border-border rounded-xl p-5 flex flex-col justify-between space-y-3 shadow-subtle"
                    >
                      <div className="space-y-2">
                        <span className="text-xs font-mono font-bold text-ink-faint">
                          Spec 0{idx + 1}
                        </span>
                        <p className="text-[14px] text-ink-light leading-[1.7]">
                          {hl}
                        </p>
                      </div>
                      <div className="pt-2 border-t border-border-subtle flex items-center gap-1.5 text-[11px] font-mono text-accent-emerald">
                        <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald" />
                        <span>Implemented & Verified</span>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* 7. Product & User Experience */}
              <section className="space-y-3">
                <div className="section-kicker">
                  <span className="w-2 h-2 rounded-full bg-ink" />
                  <span>06 / User Experience & Interfaces</span>
                </div>
                <div className="bg-paper-100/70 border border-border rounded-xl p-5 sm:p-6 text-[14.5px] text-ink-light leading-[1.75]">
                  {project.caseStudy.productExperience}
                </div>
              </section>

              {/* 8. Deliverables vs Measurable Outcomes (High-Contrast Scorecard) */}
              <section className="space-y-4">
                <div className="section-kicker">
                  <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
                  <span>07 / Deliverables & Measured Impact</span>
                </div>

                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="bg-white border border-border rounded-xl p-5 space-y-3 shadow-subtle">
                    <span className="text-xs font-mono font-bold uppercase tracking-wider text-ink block">
                      Shipped Production Artifacts
                    </span>
                    <ul className="space-y-2">
                      {project.caseStudy.shippedArtifacts.map((art) => (
                        <li
                          key={art}
                          className="text-[13.5px] text-ink-light flex items-start gap-2.5"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald mt-2 shrink-0" />
                          <span>{art}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="bg-paper-100/60 border border-border rounded-xl p-5 space-y-3">
                    <span className="text-xs font-mono font-bold uppercase tracking-wider text-ink block">
                      Measurable Outcomes
                    </span>
                    <ul className="space-y-2">
                      {project.caseStudy.outcomes.map((outcome) => (
                        <li
                          key={outcome}
                          className="text-[13.5px] text-ink-light flex items-start gap-2.5"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt mt-2 shrink-0" />
                          <span>{outcome}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </section>

              {/* 9. Technologies Used */}
              <section className="space-y-3 pt-2">
                <span className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold block">
                  Core Technologies & Libraries
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {project.tags.map((tag) => (
                    <span key={tag} className="editorial-tag text-xs">
                      {tag}
                    </span>
                  ))}
                </div>
              </section>

              {/* 10. Next Project Transition Footer */}
              <footer className="pt-8 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
                <button
                  type="button"
                  onClick={onClose}
                  className="editorial-btn-secondary !py-2.5 !px-4 text-xs w-full sm:w-auto justify-center"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>Close Case Study</span>
                </button>

                {nextProject && onSelectProject && (
                  <button
                    type="button"
                    onClick={() => onSelectProject(nextProject)}
                    className="editorial-btn-primary !py-2.5 !px-5 text-xs w-full sm:w-auto justify-center"
                  >
                    <span>Next Case Study: {nextProject.title}</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                )}
              </footer>
            </motion.div>
          </motion.aside>
        </div>
      )}
    </AnimatePresence>
  );
};

export default CaseStudyModal;

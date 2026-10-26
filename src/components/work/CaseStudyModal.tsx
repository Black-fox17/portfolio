import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, ArrowUpRight, X } from 'lucide-react';
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
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="fixed inset-0 bg-neutral-900/30 backdrop-blur-[2px]"
            onClick={onClose}
            aria-hidden="true"
          />

          {/* Slide-Over Panel */}
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
            className="relative w-full lg:w-[62%] xl:w-[56%] h-full bg-white border-l border-neutral-200 shadow-xl flex flex-col z-50 overflow-hidden"
            role="dialog"
            aria-modal="true"
            aria-labelledby="case-study-title"
          >
            {/* Top Navigation Bar */}
            <header className="px-6 py-4 border-b border-neutral-100 bg-white/95 backdrop-blur-md sticky top-0 z-20 flex items-center justify-between gap-4">
              <button
                type="button"
                onClick={onClose}
                className="inline-flex items-center gap-2 text-xs font-medium text-neutral-600 hover:text-neutral-900 transition-colors cursor-pointer py-1"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>Back to Overview</span>
              </button>

              <div className="flex items-center gap-3">
                <span className="text-xs text-neutral-400 font-mono hidden sm:inline-block">
                  Case Study 0{currentIndex + 1} / 0{projects.length}
                </span>
                <button
                  type="button"
                  onClick={onClose}
                  className="p-1.5 rounded text-neutral-400 hover:text-neutral-900 transition-colors cursor-pointer"
                  aria-label="Close Case Study"
                  title="Close (Esc)"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </header>

            {/* Content Container */}
            <motion.div
              key={project.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              className="flex-1 overflow-y-auto px-6 sm:px-10 py-8 sm:py-10 space-y-10"
            >
              {/* Header */}
              <div className="space-y-3 pb-6 border-b border-neutral-100">
                <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-400 font-mono">
                  <span>{project.category}</span>
                  <span>·</span>
                  <span>{project.status}</span>
                  <span>·</span>
                  <span>{project.period}</span>
                </div>

                <div className="space-y-1">
                  <h1
                    id="case-study-title"
                    className="text-2xl sm:text-3xl font-bold tracking-tight text-neutral-900"
                  >
                    {project.title}
                  </h1>
                  <p className="text-sm sm:text-base text-neutral-600">
                    {project.subtitle}
                  </p>
                </div>

                <div className="text-xs text-neutral-500 pt-1">
                  <span>Role: </span>
                  <span className="text-neutral-900 font-medium">{project.role}</span>
                </div>

                {/* Releases */}
                {project.links && project.links.length > 0 && (
                  <div className="pt-2 flex flex-wrap items-center gap-2">
                    {project.links.map((link) => (
                      <ProjectLinkBadge key={link.label} link={link} variant="dark" />
                    ))}
                  </div>
                )}
              </div>

              {/* Overview */}
              <section className="space-y-2">
                <h2 className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-medium">
                  System Overview
                </h2>
                <p className="text-sm sm:text-base text-neutral-700 leading-relaxed">
                  {project.caseStudy.overview}
                </p>
              </section>

              {/* Challenge vs Solution */}
              <section className="space-y-3">
                <h2 className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-medium">
                  The Engineering Challenge &amp; Solution
                </h2>

                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="bg-neutral-50 border border-neutral-200/60 rounded-lg p-5 space-y-2">
                    <h3 className="text-xs font-medium text-neutral-800">
                      The Challenge
                    </h3>
                    <p className="text-xs sm:text-sm text-neutral-600 leading-relaxed">
                      {project.caseStudy.problem}
                    </p>
                  </div>

                  <div className="bg-white border border-neutral-200 rounded-lg p-5 space-y-2">
                    <h3 className="text-xs font-medium text-neutral-900">
                      Architectural Solution
                    </h3>
                    <p className="text-xs sm:text-sm text-neutral-600 leading-relaxed">
                      {project.caseStudy.complexityAndFailures}
                    </p>
                  </div>
                </div>
              </section>

              {/* Architecture Layers */}
              {project.architectureDiagram && (
                <section className="space-y-3">
                  <h2 className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-medium">
                    Architecture Layers
                  </h2>

                  <div className="grid sm:grid-cols-2 gap-3">
                    {project.architectureDiagram.layers.map((layer) => (
                      <div
                        key={layer.name}
                        className="p-4 rounded-lg border border-neutral-200/80 space-y-2"
                      >
                        <span className="text-xs font-medium text-neutral-900 block">
                          {layer.name}
                        </span>
                        <ul className="space-y-1">
                          {layer.items.map((item) => (
                            <li
                              key={item}
                              className="text-xs text-neutral-600 pl-2.5 border-l border-neutral-300"
                            >
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Specifications */}
              <section className="space-y-3">
                <h2 className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-medium">
                  Technical Specifications
                </h2>
                <div className="divide-y divide-neutral-100">
                  {project.caseStudy.architecture.map((arch, idx) => (
                    <div key={idx} className="py-2.5 flex items-start gap-3">
                      <span className="text-xs font-mono text-neutral-400 shrink-0 mt-0.5">
                        0{idx + 1}
                      </span>
                      <p className="text-xs sm:text-sm text-neutral-600 leading-relaxed">
                        {arch}
                      </p>
                    </div>
                  ))}
                </div>
              </section>

              {/* Next Project Footer */}
              <div className="pt-8 border-t border-neutral-100 flex items-center justify-between">
                <button
                  type="button"
                  onClick={onClose}
                  className="text-xs text-neutral-500 hover:text-neutral-900 transition-colors cursor-pointer"
                >
                  Close Case Study
                </button>
                {onSelectProject && nextProject && (
                  <button
                    type="button"
                    onClick={() => onSelectProject(nextProject)}
                    className="inline-flex items-center gap-1.5 text-xs text-neutral-900 font-medium hover:underline underline-offset-4 decoration-neutral-300 cursor-pointer"
                  >
                    <span>Next: {nextProject.title}</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </motion.div>
          </motion.aside>
        </div>
      )}
    </AnimatePresence>
  );
};

export default CaseStudyModal;

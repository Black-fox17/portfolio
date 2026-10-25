import React, { useEffect } from 'react';
import { X, ArrowUpRight } from 'lucide-react';
import { Project } from '../../types/portfolio';

interface CaseStudyModalProps {
  project: Project | null;
  onClose: () => void;
}

export const CaseStudyModal: React.FC<CaseStudyModalProps> = ({ project, onClose }) => {
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

  if (!project) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 md:p-10 bg-ink/70 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="case-study-title"
    >
      <div
        className="bg-paper border-2 border-ink rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-modal overflow-hidden animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="p-6 sm:p-8 border-b border-border flex items-start justify-between gap-4 bg-white sticky top-0 z-10">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="editorial-tag text-[11px] font-semibold text-ink">
                {project.category}
              </span>
              <span className="text-xs font-mono text-ink-faint">·</span>
              <span className="text-xs font-mono text-ink font-medium inline-flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald" />
                <span>{project.status}</span>
              </span>
              <span className="text-xs font-mono text-ink-faint">·</span>
              <span className="text-xs font-mono text-ink-faint">
                {project.period}
              </span>
            </div>
            <h2 id="case-study-title" className="text-2xl sm:text-3xl font-bold tracking-[-0.02em] text-ink font-editorial">
              {project.title} — Architectural Case Study
            </h2>
            <p className="text-sm text-ink-muted">
              Role: <span className="font-semibold text-ink">{project.role}</span>
            </p>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg text-ink-muted hover:text-ink hover:bg-paper-100 transition-colors focus:outline-none cursor-pointer"
            aria-label="Close Case Study"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content Body */}
        <div className="p-6 sm:p-8 space-y-8 overflow-y-auto">
          {/* Executive Overview */}
          <div className="space-y-2.5">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
              <span>01 / System Overview</span>
            </h3>
            <p className="text-[15.5px] sm:text-base text-ink leading-[1.75]">
              {project.caseStudy.overview}
            </p>
          </div>

          {/* Problem Statement */}
          <div className="space-y-2.5 bg-white border border-border p-5 sm:p-6 rounded-xl shadow-subtle">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-ink" />
              <span>02 / The Problem & Operational Friction</span>
            </h3>
            <p className="text-[15px] text-ink-light leading-[1.75]">
              {project.caseStudy.problem}
            </p>
          </div>

          {/* System Architecture Diagram / Layer Breakdown */}
          {project.architectureDiagram && (
            <div className="space-y-3">
              <h3 className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
                <span>03 / System Topology & Boundaries</span>
              </h3>
              <div className="grid sm:grid-cols-2 gap-3.5">
                {project.architectureDiagram.layers.map((layer, idx) => (
                  <div key={idx} className="bg-white border border-border p-4 rounded-xl space-y-2 shadow-subtle">
                    <span className="text-xs font-mono font-bold text-ink flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
                      {layer.name}
                    </span>
                    <ul className="space-y-1.5">
                      {layer.items.map((item, i) => (
                        <li key={i} className="text-xs text-ink-muted font-mono pl-3 border-l border-border-strong">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Architecture Details */}
          <div className="space-y-3">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
              <span>04 / Architectural Specifications</span>
            </h3>
            <ul className="space-y-2.5">
              {project.caseStudy.architecture.map((arch, index) => (
                <li key={index} className="text-[14.5px] sm:text-[15px] text-ink-light flex items-start gap-3">
                  <span className="w-1.5 h-1.5 rounded-full bg-ink mt-2.5 shrink-0" />
                  <span className="leading-[1.75]">{arch}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Engineering Highlights */}
          <div className="space-y-3">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
              <span>05 / Key Engineering Contributions</span>
            </h3>
            <div className="space-y-2.5">
              {project.caseStudy.engineeringHighlights.map((hl, index) => (
                <div key={index} className="bg-white border border-border p-4 rounded-xl text-[14.5px] text-ink flex items-start gap-3 shadow-subtle">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt mt-2.5 shrink-0" />
                  <span className="leading-[1.7]">{hl}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Complexity & Failure Modes */}
          <div className="space-y-2 border-l-2 border-ink pl-5 py-1">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink font-bold">
              06 / Failure Modes & Resilience Engineering
            </h3>
            <p className="text-[14.5px] sm:text-[15px] text-ink-light leading-[1.75]">
              {project.caseStudy.complexityAndFailures}
            </p>
          </div>

          {/* Product Experience */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold">
              07 / Product & User Experience
            </h3>
            <p className="text-[14.5px] sm:text-[15px] text-ink-light leading-[1.75]">
              {project.caseStudy.productExperience}
            </p>
          </div>

          {/* Shipped Deliverables & Outcomes Grid (Dots instead of Checkmarks) */}
          <div className="grid sm:grid-cols-2 gap-4 pt-2">
            <div className="bg-white border border-border p-5 rounded-xl space-y-2.5 shadow-subtle">
              <span className="text-xs font-mono font-bold uppercase text-ink block">
                Shipped Production Artifacts
              </span>
              <ul className="space-y-2">
                {project.caseStudy.shippedArtifacts.map((art, idx) => (
                  <li key={idx} className="text-[13.5px] text-ink-light flex items-start gap-2.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald mt-2 shrink-0" />
                    <span>{art}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-white border border-border p-5 rounded-xl space-y-2.5 shadow-subtle">
              <span className="text-xs font-mono font-bold uppercase text-ink block">
                Measurable Outcomes
              </span>
              <ul className="space-y-2">
                {project.caseStudy.outcomes.map((outcome, idx) => (
                  <li key={idx} className="text-[13.5px] text-ink-light flex items-start gap-2.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt mt-2 shrink-0" />
                    <span>{outcome}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Tech Stack Chips */}
          <div className="pt-2">
            <span className="text-xs font-mono uppercase tracking-wider text-ink-faint font-semibold block mb-2.5">
              Technologies & Infrastructure
            </span>
            <div className="flex flex-wrap gap-1.5">
              {project.tags.map((tag) => (
                <span key={tag} className="editorial-tag">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 sm:p-6 border-t border-border bg-white flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            {project.links.map((link) => (
              <a
                key={link.label}
                href={link.url}
                target={link.url.startsWith('http') ? '_blank' : '_self'}
                rel={link.url.startsWith('http') ? 'noopener noreferrer' : ''}
                className="editorial-btn-secondary !py-2 !px-3.5 !text-xs"
              >
                <span>{link.label}</span>
                <ArrowUpRight className="w-3.5 h-3.5" />
              </a>
            ))}
          </div>

          <button
            onClick={onClose}
            className="editorial-btn-primary !py-2 !px-4 !text-xs ml-auto cursor-pointer"
          >
            Close Case Study
          </button>
        </div>
      </div>
    </div>
  );
};
export default CaseStudyModal;

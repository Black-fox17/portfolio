import React, { useEffect } from 'react';
import { X, ExternalLink, CheckCircle2, AlertTriangle, Layers, Terminal, Sparkles, Cpu, ArrowUpRight } from 'lucide-react';
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
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 md:p-10 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="case-study-title"
    >
      <div
        className="bg-white border border-border rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-modal overflow-hidden animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="p-6 sm:p-8 border-b border-border flex items-start justify-between gap-4 bg-paper-50 sticky top-0 z-10">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="editorial-tag text-[11px] font-semibold text-ink bg-paper-200">
                {project.category}
              </span>
              <span className="text-xs font-mono text-ink-faint">•</span>
              <span className="text-xs font-mono text-emerald-600 font-medium">
                {project.status}
              </span>
              <span className="text-xs font-mono text-ink-faint">•</span>
              <span className="text-xs font-mono text-ink-faint">
                {project.period}
              </span>
            </div>
            <h2 id="case-study-title" className="text-xl sm:text-2xl font-bold text-ink font-editorial">
              {project.title} — Technical Deep Dive
            </h2>
            <p className="text-xs sm:text-sm text-ink-muted">
              Role: <span className="font-semibold text-ink">{project.role}</span>
            </p>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg text-ink-muted hover:text-ink hover:bg-paper-100 transition-colors focus:outline-none"
            aria-label="Close Case Study"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content Body */}
        <div className="p-6 sm:p-8 space-y-8 overflow-y-auto">
          {/* Executive Overview */}
          <div className="space-y-2.5">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-accent-blue" />
              <span>01 / System Overview</span>
            </h3>
            <p className="text-sm sm:text-base text-ink leading-relaxed">
              {project.caseStudy.overview}
            </p>
          </div>

          {/* Problem Statement */}
          <div className="space-y-2.5 bg-paper-100 border border-border-subtle p-5 rounded-xl">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
              <span>02 / The Technical Problem & Context</span>
            </h3>
            <p className="text-sm text-ink-muted leading-relaxed">
              {project.caseStudy.problem}
            </p>
          </div>

          {/* System Architecture Diagram / Layer Breakdown */}
          {project.architectureDiagram && (
            <div className="space-y-3">
              <h3 className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-ink-muted" />
                <span>03 / System Topology & Boundaries</span>
              </h3>
              <div className="grid sm:grid-cols-2 gap-3 bg-paper-100 border border-border-subtle p-4 rounded-xl">
                {project.architectureDiagram.layers.map((layer, idx) => (
                  <div key={idx} className="bg-white border border-border p-3.5 rounded-lg space-y-1.5">
                    <span className="text-xs font-mono font-semibold text-ink flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-ink-muted" />
                      {layer.name}
                    </span>
                    <ul className="space-y-1">
                      {layer.items.map((item, i) => (
                        <li key={i} className="text-xs text-ink-muted font-mono pl-3 border-l border-border">
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
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-ink-muted" />
              <span>04 / Architectural Specifications</span>
            </h3>
            <ul className="space-y-2">
              {project.caseStudy.architecture.map((arch, index) => (
                <li key={index} className="text-xs sm:text-sm text-ink-muted flex items-start gap-2.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-ink-muted mt-2 shrink-0" />
                  <span className="leading-relaxed">{arch}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Engineering Highlights */}
          <div className="space-y-3">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-ink-muted" />
              <span>05 / What I Personally Engineered</span>
            </h3>
            <div className="space-y-2">
              {project.caseStudy.engineeringHighlights.map((hl, index) => (
                <div key={index} className="bg-paper-50 border border-border-subtle p-3.5 rounded-lg text-xs sm:text-sm text-ink flex items-start gap-2.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span className="leading-relaxed">{hl}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Complexity & Failure Modes */}
          <div className="space-y-2.5 border-l-2 border-ink pl-4 py-1">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink font-semibold">
              06 / Technical Complexity & Failure Resilience
            </h3>
            <p className="text-xs sm:text-sm text-ink-muted leading-relaxed">
              {project.caseStudy.complexityAndFailures}
            </p>
          </div>

          {/* Product Experience */}
          <div className="space-y-2.5">
            <h3 className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold">
              07 / Product & User Experience
            </h3>
            <p className="text-xs sm:text-sm text-ink-muted leading-relaxed">
              {project.caseStudy.productExperience}
            </p>
          </div>

          {/* Shipped Deliverables & Outcomes Grid */}
          <div className="grid sm:grid-cols-2 gap-4 pt-2">
            <div className="bg-white border border-border p-4 rounded-xl space-y-2">
              <span className="text-xs font-mono font-semibold uppercase text-ink">
                Shipped Production Artifacts
              </span>
              <ul className="space-y-1.5">
                {project.caseStudy.shippedArtifacts.map((art, idx) => (
                  <li key={idx} className="text-xs text-ink-muted flex items-start gap-2">
                    <span className="text-emerald-600 font-bold">✓</span>
                    <span>{art}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-white border border-border p-4 rounded-xl space-y-2">
              <span className="text-xs font-mono font-semibold uppercase text-ink">
                Measurable Outcomes
              </span>
              <ul className="space-y-1.5">
                {project.caseStudy.outcomes.map((outcome, idx) => (
                  <li key={idx} className="text-xs text-ink-muted flex items-start gap-2">
                    <span className="text-accent-blue font-bold">→</span>
                    <span>{outcome}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Tech Stack Chips */}
          <div className="pt-2">
            <span className="text-xs font-mono uppercase tracking-wider text-ink-faint font-semibold block mb-2">
              Technologies & Infrastructure
            </span>
            <div className="flex flex-wrap gap-1.5">
              {project.tags.map((tag) => (
                <span key={tag} className="editorial-tag text-xs">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 sm:p-6 border-t border-border bg-paper-100 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            {project.links.map((link) => (
              <a
                key={link.label}
                href={link.url}
                target={link.url.startsWith('http') ? '_blank' : '_self'}
                rel={link.url.startsWith('http') ? 'noopener noreferrer' : ''}
                className="editorial-btn-secondary !py-2 !px-3 !text-xs"
              >
                <span>{link.label}</span>
                <ArrowUpRight className="w-3.5 h-3.5" />
              </a>
            ))}
          </div>

          <button
            onClick={onClose}
            className="editorial-btn-primary !py-2 !px-4 !text-xs ml-auto"
          >
            Close Deep Dive
          </button>
        </div>
      </div>
    </div>
  );
};
export default CaseStudyModal;

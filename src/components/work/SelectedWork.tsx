import React, { useState } from 'react';
import { ArrowUpRight, ChevronRight } from 'lucide-react';
import { projects } from '../../data/projects';
import { Project } from '../../types/portfolio';
import { CaseStudyModal } from './CaseStudyModal';

export const SelectedWork: React.FC = () => {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);

  const flagshipProject = projects[0]; // Upward Ecosystem
  const secondaryProjects = projects.slice(1);

  return (
    <section id="work" className="scroll-mt-24">
      {/* Section Header */}
      <div className="space-y-3 mb-10">
        <div className="section-kicker">
          <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
          <span>02 / Systems in Production</span>
        </div>
        <h2 className="text-2xl sm:text-4xl font-bold tracking-[-0.025em] text-ink font-editorial leading-[1.15]">
          Selected Work & Architectural Case Studies
        </h2>
        <p className="text-[15.5px] sm:text-base text-ink-muted leading-relaxed max-w-2xl">
          Every project below solves a concrete operational problem—from multi-gateway financial ledgers and mobile app rollouts to asynchronous scholarly NLP pipelines. Click any system to inspect its full architecture.
        </p>
      </div>

      {/* Dominant Flagship Monograph: Upward Ecosystem */}
      <div className="mb-10">
        <div className="bg-white border-2 border-ink rounded-2xl p-6 sm:p-8 shadow-elevated relative overflow-hidden transition-all">
          {/* Status Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-5 border-b border-border">
            <div className="flex flex-wrap items-center gap-2">
              <span className="px-2.5 py-1 rounded bg-ink text-paper-50 text-[11px] font-mono font-semibold tracking-wider uppercase">
                Flagship Production System
              </span>
              <span className="text-xs font-mono text-ink font-medium inline-flex items-center gap-2 bg-paper-100 border border-border px-2.5 py-1 rounded">
                <span className="w-2 h-2 rounded-full bg-accent-emerald animate-pulse-subtle" />
                <span>3 Live Apps on App Store & Google Play</span>
              </span>
            </div>
            <span className="text-xs font-mono text-ink-faint">
              {flagshipProject.period}
            </span>
          </div>

          {/* Identity & Narrative */}
          <div className="mt-6 space-y-4">
            <div>
              <h3 className="text-2xl sm:text-3xl font-bold tracking-[-0.02em] text-ink font-editorial">
                {flagshipProject.title}
              </h3>
              <p className="text-sm sm:text-base font-semibold text-accent-cobalt mt-1">
                {flagshipProject.subtitle}
              </p>
            </div>

            <p className="text-[15px] sm:text-[16px] text-ink-light leading-[1.75]">
              {flagshipProject.summary}
            </p>

            {/* The Story Behind the Build: Why & How */}
            <div className="grid sm:grid-cols-2 gap-4 pt-2">
              <div className="p-4 rounded-xl bg-paper-100/70 border border-border space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-ink-muted" />
                  <span className="text-[11px] font-mono uppercase tracking-wider text-ink-muted font-semibold">
                    The Engineering Challenge
                  </span>
                </div>
                <p className="text-[13.5px] text-ink-light leading-relaxed">
                  Managing multi-party financial flows across tenants, landlords, and property managers while coordinating 3 native mobile apps and maintaining sub-10ms transactional state without race conditions or ledger divergence.
                </p>
              </div>
              <div className="p-4 rounded-xl bg-paper-100/70 border border-border space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
                  <span className="text-[11px] font-mono uppercase tracking-wider text-accent-cobalt font-semibold">
                    How I Solved It
                  </span>
                </div>
                <p className="text-[13.5px] text-ink-light leading-relaxed">
                  Engineered an ACID-compliant double-entry ledger in PostgreSQL backed by distributed Redis idempotency locks, unified by a 200+ endpoint NestJS micro-monolith powering native iOS (Apple App Store) and Android clients.
                </p>
              </div>
            </div>
          </div>

          {/* Architectural Telemetry Strip (Light editorial aesthetic & clean non-wrapping hierarchy) */}
          <div className="mt-6 bg-paper-100/60 border border-border rounded-xl grid grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-border overflow-hidden">
            <div className="p-3.5 sm:p-4">
              <div className="text-sm sm:text-[15px] font-bold font-editorial text-ink whitespace-nowrap">3 Mobile Apps</div>
              <div className="text-[11px] font-mono text-ink-muted mt-1 leading-snug">iOS & Android Stores</div>
            </div>
            <div className="p-3.5 sm:p-4">
              <div className="text-sm sm:text-[15px] font-bold font-editorial text-ink whitespace-nowrap">Distributed Core</div>
              <div className="text-[11px] font-mono text-ink-muted mt-1 leading-snug">NestJS & 200+ REST APIs</div>
            </div>
            <div className="p-3.5 sm:p-4">
              <div className="text-sm sm:text-[15px] font-bold font-editorial text-ink whitespace-nowrap">ACID Financial Ledger</div>
              <div className="text-[11px] font-mono text-ink-muted mt-1 leading-snug">Double-Entry Engine</div>
            </div>
            <div className="p-3.5 sm:p-4">
              <div className="text-sm sm:text-[15px] font-bold font-editorial text-ink whitespace-nowrap">Zero Downtime</div>
              <div className="text-[11px] font-mono text-ink-muted mt-1 leading-snug">Docker CI/CD Engine</div>
            </div>
          </div>

          {/* Live Releases & Case Study Trigger */}
          <div className="mt-6 pt-5 border-t border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-2">
              {flagshipProject.links.map((link) => (
                <a
                  key={link.label}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-paper-100 hover:bg-paper-200 text-ink text-xs font-mono font-medium border border-border transition-colors"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald" />
                  <span>{link.label}</span>
                  <ArrowUpRight className="w-3 h-3 text-ink-faint" />
                </a>
              ))}
            </div>

            <button
              type="button"
              onClick={() => setSelectedProject(flagshipProject)}
              className="editorial-btn-primary !py-2.5 !px-4 shrink-0 cursor-pointer"
            >
              <span>Inspect Architecture Case Study</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Secondary Production & Research Systems */}
      <div className="space-y-5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-ink" />
            <span>Autonomous Agents, Advanced Computer Vision & Medical Systems</span>
          </span>
          <span className="text-xs font-mono text-ink-faint">Click any card for deep dive</span>
        </div>

        <div className="space-y-4">
          {secondaryProjects.map((project, index) => (
            <div
              key={project.id}
              onClick={() => setSelectedProject(project)}
              className="editorial-card p-6 sm:p-7 cursor-pointer group"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2.5">
                  <span className="text-xs font-mono font-bold text-ink-muted">
                    0{index + 2}
                  </span>
                  <span className="editorial-tag">
                    {project.category}
                  </span>
                  <span className="text-xs font-mono text-ink-light font-medium inline-flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald" />
                    <span>{project.status}</span>
                  </span>
                </div>
                <span className="text-xs font-mono text-ink-faint">
                  {project.period}
                </span>
              </div>

              <div className="space-y-2">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="text-xl sm:text-2xl font-bold tracking-[-0.02em] text-ink font-editorial group-hover:text-accent-cobalt transition-colors">
                      {project.title}
                    </h3>
                    <p className="text-xs sm:text-sm font-medium text-ink-muted mt-0.5">
                      {project.subtitle}
                    </p>
                  </div>
                  <span className="p-2 rounded-lg bg-paper-100 text-ink-muted group-hover:bg-ink group-hover:text-paper-50 transition-colors shrink-0">
                    <ArrowUpRight className="w-4 h-4" />
                  </span>
                </div>

                <p className="text-[14.5px] sm:text-[15.5px] text-ink-light leading-[1.75] pt-1">
                  {project.summary}
                </p>

                {/* Highlighted Engineering Takeaway */}
                <div className="pt-2">
                  <div className="p-3.5 rounded-lg bg-paper-100/70 border border-border text-xs sm:text-[13px] text-ink-light leading-relaxed flex items-start gap-2.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt mt-2 shrink-0" />
                    <div>
                      <span className="font-mono text-[11px] uppercase tracking-wider text-ink font-bold block mb-0.5">
                        Key Engineering Highlight
                      </span>
                      {project.caseStudy.engineeringHighlights[0]}
                    </div>
                  </div>
                </div>
              </div>

              {/* Footer: Stack & Actions */}
              <div className="mt-5 pt-4 border-t border-border-subtle flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap gap-1.5">
                  {project.tags.slice(0, 6).map((tag) => (
                    <span
                      key={tag}
                      className="text-[11px] font-mono px-2 py-0.5 rounded bg-paper-100 text-ink-muted border border-border-subtle"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="flex items-center gap-4 text-xs font-mono">
                  {project.links && project.links.length > 0 && (
                    <a
                      href={project.links[0].url}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="text-ink-muted hover:text-ink inline-flex items-center gap-1 underline decoration-border-strong underline-offset-4"
                    >
                      <span>{project.links[0].label}</span>
                      <ArrowUpRight className="w-3 h-3" />
                    </a>
                  )}
                  <span className="font-semibold text-ink group-hover:text-accent-cobalt inline-flex items-center gap-1">
                    <span>Case Study →</span>
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Case Study Deep-Dive Slide-Over */}
      <CaseStudyModal
        project={selectedProject}
        onClose={() => setSelectedProject(null)}
        onSelectProject={(proj) => setSelectedProject(proj)}
      />
    </section>
  );
};

export default SelectedWork;

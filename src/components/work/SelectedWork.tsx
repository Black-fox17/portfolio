import React, { useState } from 'react';
import { ArrowUpRight, ChevronRight, Smartphone, Database, Zap, ShieldCheck, ExternalLink, BookOpen } from 'lucide-react';
import { projects } from '../../data/projects';
import { Project } from '../../types/portfolio';
import { CaseStudyModal } from './CaseStudyModal';

export const SelectedWork: React.FC = () => {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);

  const flagshipProject = projects[0]; // Upward Ecosystem
  const secondaryProjects = projects.slice(1);

  return (
    <section id="work" className="border-b border-border/60 bg-paper">
      <div className="section-container">
        {/* Section Header */}
        <div className="mb-10 sm:mb-12">
          <div className="section-heading">
            <span className="w-2 h-2 rounded-full bg-ink" />
            <span>Chapter 02 — Shipped Production Systems</span>
          </div>
          <h2 className="section-title">
            Selected Work & Production Case Studies
          </h2>
          <p className="section-desc">
            Production ecosystems, document intelligence platforms, and edge machine learning systems shipped and operated across web, mobile, and cloud environments.
          </p>
        </div>

        {/* Dominant Flagship: Upward Ecosystem */}
        <div className="mb-12 sm:mb-14">
          <div className="bg-white border-2 border-ink/80 rounded-2xl p-5 sm:p-8 md:p-10 shadow-elevated relative overflow-hidden">
            {/* Top Status Header - Responsive and Clean on Mobile */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 sm:pb-6 border-b border-border">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2.5 py-1 rounded-full bg-ink text-paper text-[10px] sm:text-xs font-mono font-semibold tracking-wide">
                  FLAGSHIP PRODUCTION SYSTEM
                </span>
                <span className="text-[11px] sm:text-xs font-mono text-emerald-800 font-medium flex items-center gap-1.5 bg-emerald-50 border border-emerald-200/80 px-2.5 py-0.5 rounded-full">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-subtle" />
                  Live on Google Play & Web
                </span>
              </div>
              <span className="text-[11px] sm:text-xs font-mono text-ink-faint">
                March 2026 – Present
              </span>
            </div>

            {/* Main Project Identity */}
            <div className="mt-6 sm:mt-8 space-y-3 sm:space-y-4">
              <div className="space-y-1">
                <h3 className="text-2xl sm:text-4xl font-bold tracking-tight text-ink font-editorial">
                  {flagshipProject.title}
                </h3>
                <p className="text-sm sm:text-lg font-medium text-ink-muted">
                  {flagshipProject.subtitle}
                </p>
              </div>

              <p className="text-xs sm:text-base text-ink-light leading-relaxed max-w-3xl">
                {flagshipProject.summary}
              </p>
            </div>

            {/* Key Engineering Metrics Grid */}
            <div className="mt-6 sm:mt-8 grid sm:grid-cols-2 gap-3 sm:gap-3.5">
              <div className="bg-paper-100 border border-border-subtle p-3.5 sm:p-4 rounded-xl flex items-start gap-3">
                <Smartphone className="w-5 h-5 text-ink shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-mono font-semibold uppercase text-ink">
                    2 Production Android Apps
                  </h4>
                  <p className="text-xs text-ink-muted mt-0.5">
                    Shipped Upward Pay & Upward PM on Google Play Store via Next.js + Capacitor.
                  </p>
                </div>
              </div>

              <div className="bg-paper-100 border border-border-subtle p-3.5 sm:p-4 rounded-xl flex items-start gap-3">
                <Database className="w-5 h-5 text-ink shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-mono font-semibold uppercase text-ink">
                    200+ Production Endpoints
                  </h4>
                  <p className="text-xs text-ink-muted mt-0.5">
                    Full NestJS, TypeScript & PostgreSQL multi-domain architecture with granular RBAC and ACID ledgers.
                  </p>
                </div>
              </div>

              <div className="bg-paper-100 border border-border-subtle p-3.5 sm:p-4 rounded-xl flex items-start gap-3">
                <Zap className="w-5 h-5 text-ink shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-mono font-semibold uppercase text-ink">
                    Dual Gateway Failover
                  </h4>
                  <p className="text-xs text-ink-muted mt-0.5">
                    Paystack primary processing with automated Flutterwave fallback for zero transaction downtime.
                  </p>
                </div>
              </div>

              <div className="bg-paper-100 border border-border-subtle p-3.5 sm:p-4 rounded-xl flex items-start gap-3">
                <ShieldCheck className="w-5 h-5 text-ink shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-mono font-semibold uppercase text-ink">
                    CI/CD & Cloud Infrastructure
                  </h4>
                  <p className="text-xs text-ink-muted mt-0.5">
                    Automated GitHub Actions pipelines, Docker deployment, and environment branch controls.
                  </p>
                </div>
              </div>
            </div>

            {/* Live Play Store & Web Links */}
            <div className="mt-6 pt-5 border-t border-border-subtle flex flex-wrap items-center gap-2.5">
              <span className="text-[11px] font-mono text-ink-faint uppercase tracking-wider block mr-1">
                Live Releases:
              </span>
              <a
                href="https://play.google.com/store/apps/details?id=com.goodtenants.upward"
                target="_blank"
                rel="noopener noreferrer"
                className="editorial-btn-secondary !py-1.5 !px-3 !text-xs !gap-1.5"
              >
                <span>Upward Pay (Play Store)</span>
                <ExternalLink className="w-3 h-3 text-ink-faint" />
              </a>
              <a
                href="https://play.google.com/store/apps/details?id=com.goodtenants.upward.pm"
                target="_blank"
                rel="noopener noreferrer"
                className="editorial-btn-secondary !py-1.5 !px-3 !text-xs !gap-1.5"
              >
                <span>Upward PM (Play Store)</span>
                <ExternalLink className="w-3 h-3 text-ink-faint" />
              </a>
              <a
                href="https://upward.goodtenants.io"
                target="_blank"
                rel="noopener noreferrer"
                className="editorial-btn-secondary !py-1.5 !px-3 !text-xs !gap-1.5"
              >
                <span>Upward Web Portal</span>
                <ExternalLink className="w-3 h-3 text-ink-faint" />
              </a>
            </div>

            {/* Tech Stack Chips & Case Study Button */}
            <div className="mt-5 pt-5 border-t border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex flex-wrap gap-1.5 max-w-xl">
                {flagshipProject.tags.map((tag) => (
                  <span key={tag} className="editorial-tag text-xs">
                    {tag}
                  </span>
                ))}
              </div>

              {/* Deep Dive Action Button */}
              <button
                type="button"
                onClick={() => setSelectedProject(flagshipProject)}
                className="editorial-btn-primary !py-2.5 !px-5 w-full sm:w-auto shrink-0 justify-center cursor-pointer shadow-md hover:shadow-lg"
              >
                <BookOpen className="w-4 h-4 text-paper-200" />
                <span>Open Full Architecture Case Study</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Secondary Projects Grid */}
        <div className="space-y-4">
          <div className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold">
            Other Production & Research Systems
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {secondaryProjects.map((project) => (
              <div
                key={project.id}
                className="editorial-card p-6 flex flex-col justify-between space-y-6 bg-white cursor-pointer group hover:border-ink/60"
                onClick={() => setSelectedProject(project)}
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="editorial-tag text-[11px]">
                      {project.category}
                    </span>
                    <span className="text-[11px] font-mono text-emerald-700 font-medium">
                      {project.status}
                    </span>
                  </div>

                  <h3 className="text-lg font-bold text-ink font-editorial group-hover:text-accent-blue transition-colors">
                    {project.title}
                  </h3>

                  <p className="text-xs text-ink-muted leading-relaxed line-clamp-3">
                    {project.summary}
                  </p>
                </div>

                <div className="space-y-4 pt-2 border-t border-border-subtle">
                  <div className="flex flex-wrap gap-1">
                    {project.tags.slice(0, 4).map((tag) => (
                      <span key={tag} className="text-[10px] font-mono px-2 py-0.5 rounded bg-paper-100 text-ink-faint border border-border-subtle">
                        {tag}
                      </span>
                    ))}
                    {project.tags.length > 4 && (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 text-ink-faint">
                        +{project.tags.length - 4}
                      </span>
                    )}
                  </div>

                  {/* Actions row: Case study button + direct external link if present */}
                  <div className="flex items-center justify-between pt-1 text-xs">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedProject(project);
                      }}
                      className="font-medium text-ink group-hover:text-accent-blue flex items-center gap-1 hover:underline underline-offset-4"
                    >
                      <BookOpen className="w-3.5 h-3.5" />
                      <span>View Case Study</span>
                    </button>

                    {project.links && project.links.length > 0 && (
                      <a
                        href={project.links[0].url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="text-xs font-mono text-ink-faint hover:text-ink flex items-center gap-1"
                      >
                        <span>{project.links[0].label}</span>
                        <ArrowUpRight className="w-3.5 h-3.5" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Case Study Deep-Dive Modal */}
      <CaseStudyModal
        project={selectedProject}
        onClose={() => setSelectedProject(null)}
      />
    </section>
  );
};
export default SelectedWork;

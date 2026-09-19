import React from 'react';
import { experiences } from '../../data/experience';
import { Briefcase, MapPin, CheckCircle2 } from 'lucide-react';

export const ExperienceTimeline: React.FC = () => {
  return (
    <section id="experience" className="border-b border-border/60 bg-paper-50">
      <div className="section-container">
        {/* Section Header */}
        <div className="mb-12">
          <div className="section-heading">
            <span className="w-2 h-2 rounded-full bg-ink" />
            <span>Chapter 03 — Engineering Experience & Track Record</span>
          </div>
          <h2 className="section-title">
            Professional Experience & Career Progression
          </h2>
          <p className="section-desc">
            A progression from deep learning research and computer vision foundations into distributed RAG architectures, cloud infrastructure, and full-lifecycle production systems.
          </p>
        </div>

        {/* Editorial Timeline Container */}
        <div className="relative border-l border-border-strong ml-3 sm:ml-4 pl-6 sm:pl-8 space-y-12">
          {experiences.map((exp, index) => (
            <div key={exp.id} className="relative group">
              {/* Timeline Dot Indicator */}
              <div
                className={`absolute -left-[31px] sm:-left-[39px] top-1.5 w-3.5 h-3.5 rounded-full border-2 bg-white transition-colors ${
                  exp.current
                    ? 'border-ink bg-ink ring-4 ring-ink/10'
                    : 'border-ink-muted bg-white group-hover:border-ink'
                }`}
              />

              {/* Experience Card */}
              <div
                className={`p-6 sm:p-8 rounded-xl border transition-all duration-200 ${
                  exp.current
                    ? 'bg-white border-border shadow-elevated'
                    : 'bg-white/80 border-border hover:bg-white hover:border-ink-muted/30'
                }`}
              >
                {/* Header: Company, Role & Meta */}
                <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 pb-4 border-b border-border-subtle">
                  <div className="space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-mono px-2 py-0.5 rounded bg-paper-100 text-ink-muted font-medium border border-border-subtle">
                        {exp.domain}
                      </span>
                      {exp.current && (
                        <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
                          CURRENT ROLE
                        </span>
                      )}
                    </div>
                    <h3 className="text-xl sm:text-2xl font-bold text-ink font-editorial">
                      {exp.company}
                    </h3>
                    <p className="text-sm font-semibold text-ink-light">
                      {exp.role}
                    </p>
                  </div>

                  <div className="text-xs font-mono text-ink-faint space-y-1 sm:text-right shrink-0">
                    <div className="font-medium text-ink-muted">{exp.period}</div>
                    <div className="flex items-center sm:justify-end gap-1">
                      <MapPin className="w-3 h-3 text-ink-faint" />
                      <span>{exp.location} ({exp.type})</span>
                    </div>
                  </div>
                </div>

                {/* Summary */}
                <p className="text-xs sm:text-sm text-ink-muted leading-relaxed mt-4">
                  {exp.summary}
                </p>

                {/* Highlights List */}
                <div className="mt-5 space-y-2.5">
                  {exp.highlights.map((hl, idx) => (
                    <div key={idx} className="flex items-start gap-2.5 text-xs sm:text-sm text-ink-light leading-relaxed">
                      <span className="w-1.5 h-1.5 rounded-full bg-ink-muted mt-2 shrink-0" />
                      <span>{hl}</span>
                    </div>
                  ))}
                </div>

                {/* Tech Chips */}
                <div className="mt-6 pt-4 border-t border-border-subtle flex flex-wrap gap-1.5">
                  {exp.technologies.map((tech) => (
                    <span
                      key={tech}
                      className="text-[11px] font-mono px-2 py-0.5 rounded bg-paper-100 text-ink-muted border border-border-subtle"
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
export default ExperienceTimeline;

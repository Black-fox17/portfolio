import React, { useState } from 'react';
import { experiences } from '../../data/experience';
import { leadershipData } from '../../data/leadership';
import { ChevronDown, ChevronUp } from 'lucide-react';

export const ExperienceTimeline: React.FC = () => {
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({
    'goodtenants-upward': true
  });

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <section id="experience" className="scroll-mt-24">
      {/* Section Header */}
      <div className="space-y-3 mb-10">
        <div className="section-kicker">
          <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
          <span>03 / Career Progression & Leadership</span>
        </div>
        <h2 className="text-2xl sm:text-4xl font-bold tracking-[-0.025em] text-ink font-editorial leading-[1.15]">
          Professional Track Record & Community Impact
        </h2>
        <p className="text-[15.5px] sm:text-base text-ink-muted leading-relaxed max-w-2xl">
          From clinical computer vision research at University College Hospital to enterprise RAG pipelines and full-lifecycle PropTech/FinTech systems.
        </p>
      </div>

      {/* Clean Editorial Experience Feed */}
      <div className="relative border-l border-border-strong ml-2.5 sm:ml-3.5 pl-6 sm:pl-8 space-y-8">
        {experiences.map((exp) => {
          const isExpanded = !!expandedIds[exp.id];
          const visibleHighlights = isExpanded ? exp.highlights : exp.highlights.slice(0, 2);
          const hasMore = exp.highlights.length > 2;

          return (
            <div key={exp.id} className="relative group">
              {/* Timeline Indicator Dot */}
              <div
                className={`absolute -left-[31px] sm:-left-[39px] top-2 w-3.5 h-3.5 rounded-full border-2 transition-colors ${
                  exp.current
                    ? 'border-accent-cobalt bg-accent-cobalt ring-4 ring-accent-cobalt/15'
                    : 'border-ink bg-paper group-hover:bg-ink'
                }`}
              />

              <div
                className={`p-6 sm:p-7 rounded-xl border transition-all ${
                  exp.current
                    ? 'bg-white border-ink shadow-card'
                    : 'bg-white border-border hover:border-border-strong'
                }`}
              >
                {/* Top Meta Row */}
                <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-mono font-semibold text-ink">
                      {exp.period}
                    </span>
                    {exp.current && (
                      <span className="text-[10.5px] font-mono font-semibold px-2 py-0.5 rounded bg-paper-100 text-ink inline-flex items-center gap-1.5 border border-border">
                        <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald" />
                        <span>CURRENT ROLE</span>
                      </span>
                    )}
                  </div>
                  <div className="text-xs font-mono text-ink-faint">
                    {exp.location} · {exp.type}
                  </div>
                </div>

                {/* Role & Company */}
                <div className="space-y-0.5">
                  <h3 className="text-xl sm:text-2xl font-bold tracking-[-0.02em] text-ink font-editorial">
                    {exp.role}{' '}
                    <span className="text-ink-faint font-normal">·</span>{' '}
                    <span className="text-accent-cobalt">{exp.company}</span>
                  </h3>
                  <p className="text-xs font-mono text-ink-muted">{exp.domain}</p>
                </div>

                {/* Narrative Summary */}
                <p className="text-[15px] text-ink-light leading-relaxed mt-3.5 font-medium">
                  {exp.summary}
                </p>

                {/* Key Engineering Highlights (Geometric Dots) */}
                <ul className="mt-4 space-y-2.5">
                  {visibleHighlights.map((hl, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-3 text-[14px] sm:text-[14.5px] text-ink-light leading-[1.7]"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-ink mt-2.5 shrink-0" />
                      <span>{hl}</span>
                    </li>
                  ))}
                </ul>

                {hasMore && (
                  <button
                    type="button"
                    onClick={() => toggleExpand(exp.id)}
                    className="mt-3.5 inline-flex items-center gap-1 text-xs font-mono font-semibold text-accent-cobalt hover:text-ink transition-colors cursor-pointer"
                  >
                    {isExpanded ? (
                      <>
                        <span>Show fewer details</span>
                        <ChevronUp className="w-3.5 h-3.5" />
                      </>
                    ) : (
                      <>
                        <span>Read {exp.highlights.length - 2} more engineering contributions</span>
                        <ChevronDown className="w-3.5 h-3.5" />
                      </>
                    )}
                  </button>
                )}

                {/* Technologies */}
                <div className="mt-5 pt-4 border-t border-border-subtle flex flex-wrap gap-1.5">
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
          );
        })}
      </div>

      {/* Integrated Technical Leadership & Mentorship Block (Dark Spec Header + Hairline Grid, Zero Icons) */}
      <div className="mt-14 pt-10 border-t border-border">
        <div className="mb-6">
          <span className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-ink" />
            <span>Multiplying Engineering Talent</span>
          </span>
          <h3 className="text-2xl font-bold tracking-[-0.02em] text-ink font-editorial">
            Mentorship, Teaching & Community Leadership
          </h3>
          <p className="text-sm text-ink-muted mt-1">
            Great engineers elevate the people around them. Alongside shipping production software, I actively teach and mentor emerging engineers.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 bg-paper-100/80 border border-border rounded-xl divide-y sm:divide-y-0 sm:divide-x divide-border overflow-hidden">
          {leadershipData.slice(0, 2).map((item) => (
            <div key={item.organization} className="p-5 sm:p-6 space-y-2">
              <div className="flex items-baseline gap-2.5">
                <span className="text-2xl sm:text-3xl font-bold font-editorial tracking-tight text-ink">
                  {item.metric}
                </span>
                <span className="text-xs font-mono uppercase tracking-wider text-accent-cobalt font-semibold">
                  {item.label}
                </span>
              </div>
              <div className="text-sm font-bold text-ink font-editorial">
                {item.role} · <span className="font-normal text-ink-muted">{item.organization}</span>
              </div>
              <p className="text-[13.5px] text-ink-light leading-relaxed">
                {item.description}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-3 grid sm:grid-cols-2 bg-paper-100/80 border border-border rounded-xl divide-y sm:divide-y-0 sm:divide-x divide-border overflow-hidden">
          {leadershipData.slice(2, 4).map((item) => (
            <div key={item.organization} className="p-5 sm:p-6 space-y-2">
              <div className="flex items-baseline gap-2.5">
                <span className="text-2xl sm:text-3xl font-bold font-editorial tracking-tight text-ink">
                  {item.metric}
                </span>
                <span className="text-xs font-mono uppercase tracking-wider text-accent-cobalt font-semibold">
                  {item.label}
                </span>
              </div>
              <div className="text-sm font-bold text-ink font-editorial">
                {item.role} · <span className="font-normal text-ink-muted">{item.organization}</span>
              </div>
              <p className="text-[13.5px] text-ink-light leading-relaxed">
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ExperienceTimeline;

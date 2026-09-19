import React from 'react';
import { skillCategories } from '../../data/skills';
import { Cpu, Database, Cloud, Layout, Check } from 'lucide-react';

const icons = [Cpu, Database, Cloud, Layout];

export const TechnicalIndex: React.FC = () => {
  return (
    <section id="skills" className="border-b border-border/60 bg-paper-50">
      <div className="section-container">
        {/* Section Header */}
        <div className="mb-12">
          <div className="section-heading">
            <span className="w-2 h-2 rounded-full bg-ink" />
            <span>Chapter 05 — Technical Capabilities</span>
          </div>
          <h2 className="section-title">
            Technical Index & Systems Stack
          </h2>
          <p className="section-desc">
            An index of technologies, frameworks, and system paradigms applied across production architectures, research papers, and deployed applications.
          </p>
        </div>

        {/* 4-Domain Systems Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {skillCategories.map((category, idx) => {
            const Icon = icons[idx] || Cpu;

            return (
              <div
                key={category.title}
                className="editorial-card p-6 sm:p-8 flex flex-col justify-between space-y-6 bg-white"
              >
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-paper-100 border border-border-subtle">
                      <Icon className="w-5 h-5 text-ink" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-ink font-editorial">
                        {category.title}
                      </h3>
                      <p className="text-xs text-ink-faint">
                        Domain {idx + 1}
                      </p>
                    </div>
                  </div>

                  <p className="text-xs sm:text-sm text-ink-muted leading-relaxed">
                    {category.description}
                  </p>
                </div>

                <div className="space-y-2.5 pt-2 border-t border-border-subtle">
                  {category.skills.map((skill) => (
                    <div
                      key={skill.name}
                      className="p-2.5 rounded-lg bg-paper-50 border border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-xs"
                    >
                      <span className="font-semibold text-ink font-mono flex items-center gap-1.5">
                        <Check className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        {skill.name}
                      </span>
                      {skill.focus && (
                        <span className="text-[11px] text-ink-muted sm:text-right">
                          {skill.focus}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
export default TechnicalIndex;

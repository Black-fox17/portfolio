import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, ArrowRight } from 'lucide-react';

interface ExperienceRow {
  company: string;
  role: string;
  period: string;
}

const experienceRows: ExperienceRow[] = [
  {
    company: 'GoodTenants',
    role: 'Lead Backend & Mobile Engineer (Upward Ecosystem)',
    period: '2026 - Present',
  },
  {
    company: 'Codygo',
    role: 'AI Full-Stack Engineer (AWS Data Pipelines & RAG)',
    period: '2025',
  },
  {
    company: 'Tweakrr',
    role: 'AI & Backend Engineer (Academic Citation Search)',
    period: '2024 - 2025',
  },
  {
    company: 'IAMRAT / UCH',
    role: 'Deep Learning Researcher (Medical Computer Vision)',
    period: '2024',
  },
];

export const ExperienceTimeline: React.FC = () => {
  return (
    <section id="experience" className="scroll-mt-24 space-y-5">
      {/* Section Header */}
      <div className="space-y-1">
        <div className="section-kicker">
          <span className="w-1.5 h-1.5 rounded-full bg-neutral-900" />
          <span>Career Progression</span>
        </div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-neutral-900">
          Experience
        </h2>
      </div>

      {/* Minimalist Rows matching project inspiration */}
      <div className="divide-y divide-neutral-100 border-y border-neutral-100">
        {experienceRows.map((exp) => (
          <Link
            key={exp.company}
            to="/experience"
            className="py-3.5 sm:py-4 flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 sm:gap-6 group hover:bg-neutral-50/60 -mx-3 px-3 rounded-lg transition-colors"
          >
            <div className="font-medium text-sm text-neutral-900 group-hover:underline underline-offset-4 decoration-neutral-300 shrink-0 sm:w-44">
              {exp.company}
            </div>

            <div className="text-xs sm:text-sm text-neutral-500 flex-1 truncate">
              {exp.role}
            </div>

            <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
              <span className="text-xs font-mono text-neutral-400">
                {exp.period}
              </span>
              <ArrowUpRight className="w-3.5 h-3.5 text-neutral-400 opacity-60 group-hover:opacity-100 group-hover:text-neutral-900 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
            </div>
          </Link>
        ))}
      </div>

      {/* More on experience link */}
      <div className="pt-1">
        <Link
          to="/experience"
          className="inline-flex items-center gap-1.5 text-xs font-mono text-neutral-600 hover:text-neutral-900 hover:underline underline-offset-4 decoration-neutral-300 transition-colors"
        >
          <span>Full career history &amp; community mentorship</span>
          <ArrowRight className="w-3 h-3" />
        </Link>
      </div>
    </section>
  );
};

export default ExperienceTimeline;

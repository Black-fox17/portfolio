import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, ArrowRight } from 'lucide-react';

interface ProjectRow {
  title: string;
  description: string;
  slug?: string;
  category: string;
}

const featuredRows: ProjectRow[] = [
  {
    title: 'Upward Ecosystem',
    description: 'Multi-platform PropTech operating system (iOS, Android & Web).',
    category: 'Flagship Platform',
  },
  {
    title: 'Breast Cancer AI',
    description: 'Clinical deep learning for biopsy cancer subtyping with UCH.',
    category: 'Medical Vision',
  },
  {
    title: 'RSNA Lumbar Spine',
    description: 'Volumetric 3D MRI multi-target neural network.',
    category: 'Volumetric 3D',
  },
  {
    title: 'LoveDA Segmentation',
    description: 'Aerial satellite land-cover computer vision via point labels.',
    category: 'Aerial Vision',
  },
  {
    title: 'Tweakrr',
    description: 'Academic literature assistant with hybrid vector search.',
    category: 'AI Assistant',
  },
  {
    title: 'Asera Core',
    description: 'Autonomous agent supervisor with recursive context memory.',
    category: 'Agent Engine',
  },
];

export const SelectedWork: React.FC = () => {
  return (
    <section id="work" className="scroll-mt-24 space-y-5">
      {/* Section Header */}
      <div className="space-y-1">
        <div className="section-kicker">
          <span className="w-1.5 h-1.5 rounded-full bg-neutral-900" />
          <span>02 / Systems in Production</span>
        </div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-neutral-900">
          Projects
        </h2>
      </div>

      {/* Minimalist Rows matching inspiration image */}
      <div className="divide-y divide-neutral-100 border-y border-neutral-100">
        {featuredRows.map((project) => (
          <Link
            key={project.title}
            to="/projects"
            className="py-3.5 sm:py-4 flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 sm:gap-6 group hover:bg-neutral-50/60 -mx-3 px-3 rounded-lg transition-colors"
          >
            <div className="font-medium text-sm text-neutral-900 group-hover:underline underline-offset-4 decoration-neutral-300 shrink-0 sm:w-44">
              {project.title}
            </div>

            <div className="text-xs sm:text-sm text-neutral-500 flex-1 truncate">
              {project.description}
            </div>

            <div className="text-neutral-400 group-hover:text-neutral-900 transition-colors shrink-0 self-end sm:self-auto">
              <ArrowUpRight className="w-3.5 h-3.5 opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
            </div>
          </Link>
        ))}
      </div>

      {/* More projects link */}
      <div className="pt-1">
        <Link
          to="/projects"
          className="inline-flex items-center gap-1.5 text-xs font-mono text-neutral-600 hover:text-neutral-900 hover:underline underline-offset-4 decoration-neutral-300 transition-colors"
        >
          <span>More projects &amp; architecture</span>
          <ArrowRight className="w-3 h-3" />
        </Link>
      </div>
    </section>
  );
};

export default SelectedWork;

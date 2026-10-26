import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, ArrowRight } from 'lucide-react';
import { blogPosts } from '../../constants/blogData';

export const TheDeepEnd: React.FC = () => {
  const featuredEssays = blogPosts.slice(0, 4);

  return (
    <section id="writing" className="scroll-mt-24 space-y-5">
      {/* Section Header */}
      <div className="space-y-1.5">
        <div className="section-kicker">
          <span className="w-1.5 h-1.5 rounded-full bg-neutral-900" />
          <span>04 / First Principles &amp; Writing</span>
        </div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-neutral-900">
          The Deep End
        </h2>
        <p className="text-xs sm:text-[13px] text-neutral-500 leading-relaxed font-normal">
          No hype, no black boxes. What happens when you build Transformers, tokenizers, and deep learning architectures entirely from scratch in PyTorch.
        </p>
      </div>

      {/* Minimalist Rows matching project inspiration */}
      <div className="divide-y divide-neutral-100 border-y border-neutral-100">
        {featuredEssays.map((article) => (
          <Link
            key={article.id}
            to={`/blog/${article.slug}`}
            className="py-3.5 sm:py-4 flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 sm:gap-6 group hover:bg-neutral-50/60 -mx-3 px-3 rounded-lg transition-colors"
          >
            <div className="font-medium text-sm text-neutral-900 group-hover:underline underline-offset-4 decoration-neutral-300 shrink-0 sm:w-56 truncate">
              {article.title}
            </div>

            <div className="text-xs sm:text-sm text-neutral-500 flex-1 truncate">
              {article.excerpt}
            </div>

            <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
              <span className="text-xs font-mono text-neutral-400">
                {article.readTime}
              </span>
              <ArrowUpRight className="w-3.5 h-3.5 text-neutral-400 opacity-60 group-hover:opacity-100 group-hover:text-neutral-900 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
            </div>
          </Link>
        ))}
      </div>

      {/* More on writing link */}
      <div className="pt-1">
        <Link
          to="/blog"
          className="inline-flex items-center gap-1.5 text-xs font-mono text-neutral-600 hover:text-neutral-900 hover:underline underline-offset-4 decoration-neutral-300 transition-colors"
        >
          <span>All technical publications ({blogPosts.length} notes)</span>
          <ArrowRight className="w-3 h-3" />
        </Link>
      </div>
    </section>
  );
};

export default TheDeepEnd;

import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft, ArrowUpRight } from 'lucide-react';
import { motion } from 'framer-motion';
import Navbar from '../components/layout/Navbar';
import { projects } from '../data/projects';
import { ProjectLinkBadge } from '../components/common/ProjectLinkBadge';

// Prioritize computer vision projects after Upward Ecosystem
const projectOrder = [
  'upward-ecosystem',
  'breast-cancer-histopathology',
  'rsna-lumbar-spine',
  'point-supervised-loveda-segmentation',
  'retail-shelf-intelligence',
  'asl-fingerspelling',
  'tweakrr',
  'asera',
];

const sortedProjects = [...projects].sort((a, b) => {
  const indexA = projectOrder.indexOf(a.id);
  const indexB = projectOrder.indexOf(b.id);
  return (indexA === -1 ? 99 : indexA) - (indexB === -1 ? 99 : indexB);
});

export const Projects: React.FC = () => {
  const flagshipProject = sortedProjects[0];
  const otherProjects = sortedProjects.slice(1);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="min-h-screen bg-white text-neutral-900 selection:bg-neutral-900 selection:text-white flex flex-col"
    >
      <Helmet>
        <title>Projects &amp; Systems | Abdulsalam Ayeleru</title>
        <meta
          name="description"
          content="Production software systems, 3 shipped mobile applications, and clinical computer vision architectures by Abdulsalam Oluwaseun Ayeleru."
        />
      </Helmet>

      <Navbar />

      <main className="flex-1 max-w-[680px] w-full mx-auto px-5 sm:px-6 pt-20 sm:pt-28 pb-20 space-y-12">
        {/* Back Link */}
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-900 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Home</span>
        </Link>

        {/* Page Header */}
        <header className="space-y-3 pb-8 border-b border-neutral-100">
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-neutral-900">
            Projects.
          </h1>
          <p className="text-sm sm:text-base text-neutral-600 leading-relaxed">
            Detailed architectures of systems I have taken to production—from cross-platform mobile apps to clinical computer vision pipelines.
          </p>
        </header>

        {/* Flagship Section */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-medium">
              Flagship Production System
            </span>
            <span className="text-xs font-mono text-neutral-400">
              {flagshipProject.period}
            </span>
          </div>

          <div className="bg-white border border-neutral-200/90 rounded-2xl p-6 sm:p-7 shadow-subtle space-y-4">
            <div className="space-y-1.5">
              <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-neutral-900">
                {flagshipProject.title}
              </h2>
              <p className="text-sm font-medium text-neutral-500">
                {flagshipProject.subtitle}
              </p>
            </div>

            <p className="text-[15px] text-neutral-600 leading-relaxed">
              A unified property management and financial operating system connecting tenants, landlords, and property managers. It handles rent collections, automated lease workflows, and tenant verification. Shipped across 3 live applications on the Apple App Store and Google Play, supported by a centralized NestJS server handling concurrent requests and PostgreSQL synchronization.
            </p>

            {/* Live Store Badges */}
            <div className="pt-2 flex flex-wrap items-center gap-2">
              {flagshipProject.links.map((link) => (
                <ProjectLinkBadge key={link.label} link={link} variant="light" />
              ))}
            </div>

            {/* Tech Tags */}
            <div className="pt-3 border-t border-neutral-100 flex flex-wrap gap-1.5">
              {flagshipProject.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-[11px] font-mono px-2 py-0.5 rounded bg-neutral-100 text-neutral-600"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </section>

        {/* Other Projects Section */}
        <section className="space-y-6 pt-4 border-t border-neutral-100">
          <div className="space-y-1">
            <h2 className="text-base font-semibold text-neutral-900">
              Computer Vision &amp; Applied AI Systems
            </h2>
            <p className="text-xs text-neutral-500">
              Medical imaging, 3D volumetric diagnostics, satellite segmentation, and autonomous agents.
            </p>
          </div>

          <div className="space-y-5">
            {otherProjects.map((project) => (
              <article
                key={project.id}
                className="bg-white border border-neutral-200/80 rounded-xl p-5 sm:p-6 space-y-3.5 hover:border-neutral-300 transition-colors"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <div>
                    <span className="text-[11px] font-mono uppercase tracking-wider text-neutral-400 block mb-0.5">
                      {project.category}
                    </span>
                    <h3 className="text-lg font-bold tracking-tight text-neutral-900">
                      {project.title}
                    </h3>
                  </div>
                  <span className="text-xs font-mono text-neutral-400">
                    {project.period}
                  </span>
                </div>

                <p className="text-sm text-neutral-600 leading-relaxed">
                  {project.summary}
                </p>

                {/* Links */}
                {project.links && project.links.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 pt-1">
                    {project.links.map((link) => (
                      <a
                        key={link.label}
                        href={link.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-xs font-mono text-neutral-700 hover:text-neutral-900 bg-neutral-100 hover:bg-neutral-200/70 px-2.5 py-1 rounded transition-colors"
                      >
                        <span>{link.label}</span>
                        <ArrowUpRight className="w-3 h-3 opacity-60" />
                      </a>
                    ))}
                  </div>
                )}

                {/* Tech tags */}
                <div className="pt-2 border-t border-neutral-100 flex flex-wrap gap-1.5">
                  {project.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[10.5px] font-mono text-neutral-400"
                    >
                      {tag} ·
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="pt-12 border-t border-neutral-100 text-xs text-neutral-400 flex items-center justify-between">
          <span>© {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru</span>
          <Link
            to="/contact"
            className="hover:text-neutral-900 transition-colors font-medium"
          >
            Get in touch →
          </Link>
        </footer>
      </main>
    </motion.div>
  );
};

export default Projects;

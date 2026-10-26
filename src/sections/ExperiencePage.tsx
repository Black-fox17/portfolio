import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';
import Navbar from '../components/layout/Navbar';
import { experiences } from '../data/experience';
import { leadershipData } from '../data/leadership';

export const ExperiencePage: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="min-h-screen bg-white text-neutral-900 selection:bg-neutral-900 selection:text-white flex flex-col"
    >
      <Helmet>
        <title>Experience &amp; Leadership | Abdulsalam Ayeleru</title>
        <meta
          name="description"
          content="Professional career history, production milestones at GoodTenants, Codygo, and UCH Ibadan, and community leadership by Abdulsalam Oluwaseun Ayeleru."
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
            Experience.
          </h1>
          <p className="text-sm sm:text-base text-neutral-600 leading-relaxed">
            Professional track record taking complex systems to production, architecting dependable backends, and mentoring engineers.
          </p>
        </header>

        {/* Roles List */}
        <section className="space-y-8">
          {experiences.map((exp) => (
            <div
              key={exp.id}
              className="bg-white border border-neutral-200/80 rounded-xl p-6 space-y-4 hover:border-neutral-300 transition-colors"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-neutral-100 pb-3">
                <div>
                  <h2 className="text-lg font-bold tracking-tight text-neutral-900">
                    {exp.role}{' '}
                    <span className="text-neutral-400 font-normal">at</span>{' '}
                    <span className="text-neutral-900 font-semibold">{exp.company}</span>
                  </h2>
                  <span className="text-xs text-neutral-400 font-mono">
                    {exp.location} · {exp.type}
                  </span>
                </div>
                <span className="text-xs font-mono font-medium text-neutral-600 bg-neutral-100 px-2.5 py-1 rounded">
                  {exp.period}
                </span>
              </div>

              <p className="text-sm text-neutral-700 leading-relaxed font-normal">
                {exp.summary}
              </p>

              {/* Highlights */}
              <div className="space-y-2">
                <span className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-medium block">
                  Key Accomplishments
                </span>
                <ul className="space-y-2 text-xs sm:text-[13px] text-neutral-600">
                  {exp.highlights.map((hl, idx) => (
                    <li key={idx} className="flex items-start gap-2.5 leading-relaxed">
                      <span className="w-1.5 h-1.5 rounded-full bg-neutral-400 mt-2 shrink-0" />
                      <span>{hl}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Technologies */}
              <div className="pt-3 border-t border-neutral-100 flex flex-wrap gap-1.5">
                {exp.technologies.map((tech) => (
                  <span
                    key={tech}
                    className="text-[10.5px] font-mono px-2 py-0.5 rounded bg-neutral-50 text-neutral-600 border border-neutral-200/60"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </section>

        {/* Mentorship & Leadership */}
        <section className="pt-6 border-t border-neutral-100 space-y-6">
          <div className="space-y-1">
            <h2 className="text-xl font-bold tracking-tight text-neutral-900">
              Community Leadership &amp; Mentorship
            </h2>
            <p className="text-sm text-neutral-600">
              Alongside shipping software, I actively teach, guide emerging engineers, and contribute to developer communities.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            {leadershipData.map((item) => (
              <div
                key={item.organization}
                className="p-5 space-y-2 bg-neutral-50 border border-neutral-200/70 rounded-xl"
              >
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold tracking-tight text-neutral-900">
                    {item.metric}
                  </span>
                  <span className="text-xs font-mono text-neutral-500 font-medium uppercase">
                    {item.label}
                  </span>
                </div>
                <div className="text-sm font-semibold text-neutral-900">
                  {item.role} · <span className="font-normal text-neutral-500">{item.organization}</span>
                </div>
                <p className="text-xs text-neutral-600 leading-relaxed">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="pt-12 border-t border-neutral-100 text-xs text-neutral-400 flex items-center justify-between">
          <span>© {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru</span>
          <Link to="/contact" className="hover:text-neutral-900 transition-colors font-medium">
            Get in touch →
          </Link>
        </footer>
      </main>
    </motion.div>
  );
};

export default ExperiencePage;

import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft, ArrowUpRight, BookOpen, GitBranch, Cpu, Microscope } from 'lucide-react';
import { openSourceReproductions, publishedArticles } from '../data/articles';

export const Research: React.FC = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="min-h-screen bg-paper text-ink flex flex-col selection:bg-ink selection:text-paper">
      <Helmet>
        <title>Research & First Principles — Abdulsalam Ayeleru</title>
        <meta
          name="description"
          content="Academic and first-principles machine learning research by Abdulsalam Oluwaseun Ayeleru: medical computer vision, transformer mechanics, and open-source paper implementations."
        />
        <link rel="canonical" href="https://salam-portfolio-three.vercel.app/research" />
      </Helmet>

      <Navbar />

      <main className="flex-1 pt-32 pb-24">
        <div className="max-w-4xl mx-auto px-5 sm:px-8 space-y-12">
          {/* Back Link */}
          <Link
            to="/"
            className="inline-flex items-center gap-1.5 text-xs font-mono text-ink-muted hover:text-ink transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Portfolio</span>
          </Link>

          {/* Header */}
          <div className="border-b border-border pb-8 space-y-4">
            <div className="section-heading">
              <span className="w-2 h-2 rounded-full bg-ink" />
              <span>Research & First Principles</span>
            </div>
            <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-ink font-editorial">
              Applied Machine Learning Research & Paper Reproductions
            </h1>
            <p className="text-sm sm:text-base text-ink-muted max-w-2xl leading-relaxed">
              Investigating deep learning architectures from mathematical fundamentals to clinical and edge applications.
            </p>
          </div>

          {/* Highlighted Research Project: IAMRAT Cancer Histopathology */}
          <div className="bg-white border-2 border-ink/70 p-6 sm:p-8 rounded-2xl shadow-elevated space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-border">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-1 rounded bg-ink text-paper text-xs font-mono font-semibold">
                  CLINICAL AI RESEARCH
                </span>
                <span className="text-xs font-mono text-ink-muted">
                  IAMRAT · UCH Ibadan
                </span>
              </div>
              <span className="text-xs font-mono text-ink-faint">
                Dec 2024 – March 2025
              </span>
            </div>

            <div className="space-y-3">
              <h2 className="text-xl sm:text-2xl font-bold text-ink font-editorial">
                Breast Cancer Subtype Classification & Diagnostic Explanation Generation
              </h2>
              <p className="text-xs sm:text-sm text-ink-light leading-relaxed">
                Researched deep convolutional neural networks (ResNet and MobileNet architectures) for subtype classification from histopathological biopsy imagery. Engineered a pipeline connecting classification activations to clinician-friendly natural language diagnostic explanations.
              </p>
            </div>

            <div className="grid sm:grid-cols-3 gap-3 pt-2 text-xs font-mono">
              <div className="p-3 rounded-lg bg-paper-100 border border-border-subtle">
                <span className="text-ink-faint block">Domain</span>
                <span className="text-ink font-semibold mt-0.5 block">Medical Imaging & Pathology</span>
              </div>
              <div className="p-3 rounded-lg bg-paper-100 border border-border-subtle">
                <span className="text-ink-faint block">Architectures</span>
                <span className="text-ink font-semibold mt-0.5 block">ResNet, MobileNet, LLMs</span>
              </div>
              <div className="p-3 rounded-lg bg-paper-100 border border-border-subtle">
                <span className="text-ink-faint block">Tooling</span>
                <span className="text-ink font-semibold mt-0.5 block">PyTorch, OpenCV, FastAPI</span>
              </div>
            </div>
          </div>

          {/* Technical Publications ("The Deep End") */}
          <div className="space-y-4">
            <div className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
              <BookOpen className="w-3.5 h-3.5 text-ink-muted" />
              <span>Theoretical Investigations & Writing</span>
            </div>

            <div className="divide-y divide-border border-y border-border bg-white rounded-xl overflow-hidden shadow-subtle">
              {publishedArticles.map((art) => (
                <Link
                  key={art.id}
                  to={`/blog/${art.slug}`}
                  className="p-5 sm:p-6 block hover:bg-paper-50 transition-colors group"
                >
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                    <div className="space-y-1 max-w-2xl">
                      <span className="text-[10px] font-mono uppercase text-ink-faint">
                        {art.category} • {art.readTime}
                      </span>
                      <h3 className="text-base font-bold text-ink group-hover:text-accent-blue transition-colors font-editorial">
                        {art.title}
                      </h3>
                      <p className="text-xs text-ink-muted leading-relaxed">
                        {art.excerpt}
                      </p>
                    </div>
                    <ArrowUpRight className="w-4 h-4 text-ink-faint group-hover:text-ink shrink-0" />
                  </div>
                </Link>
              ))}
            </div>
          </div>

          {/* First Principles Reproductions */}
          <div className="space-y-4">
            <div className="text-xs font-mono uppercase tracking-widest text-ink-faint font-semibold flex items-center gap-1.5">
              <GitBranch className="w-3.5 h-3.5 text-ink-muted" />
              <span>Open Source Paper Implementations</span>
            </div>

            <div className="grid sm:grid-cols-2 gap-4">
              {openSourceReproductions.map((repo) => (
                <a
                  key={repo.id}
                  href={repo.githubUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="editorial-card p-5 bg-white space-y-3 group"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono text-ink-faint px-2 py-0.5 rounded bg-paper-100">
                      {repo.category}
                    </span>
                    <ArrowUpRight className="w-3.5 h-3.5 text-ink-faint group-hover:text-ink" />
                  </div>
                  <h4 className="text-sm font-bold text-ink font-editorial group-hover:text-ink-light">
                    {repo.title}
                  </h4>
                  <p className="text-xs text-ink-muted">
                    {repo.description}
                  </p>
                </a>
              ))}
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Research;

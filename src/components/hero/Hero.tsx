import React from 'react';
import { ArrowDown, ArrowUpRight, FileText, Cpu, Server, Layers, Rocket } from 'lucide-react';
import { Link } from 'react-router-dom';

export const Hero: React.FC = () => {
  return (
    <section className="pt-32 pb-16 sm:pt-40 sm:pb-24 border-b border-border/60">
      <div className="max-w-5xl mx-auto px-5 sm:px-8">
        {/* Availability / Status Pill */}
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-paper-100 border border-border-subtle text-xs font-mono text-ink-muted mb-8">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-subtle" />
          <span>Building Production AI & SaaS Systems</span>
          <span className="text-border-strong">•</span>
          <span className="text-ink-faint">Lagos / Hybrid & Remote</span>
        </div>

        {/* Hero Title / Value Proposition */}
        <div className="space-y-6 max-w-4xl">
          <h1 className="text-3xl sm:text-5xl lg:text-[3.25rem] font-bold tracking-tight text-ink font-editorial leading-[1.15]">
            I engineer AI systems, distributed backends, and software products that make it into production.
          </h1>

          <p className="text-base sm:text-lg text-ink-muted leading-relaxed max-w-2xl">
            Software & AI Systems Engineer with depth in semantic retrieval, document intelligence, payment orchestration, and first-principles machine learning. I own systems from architectural specification to store deployment.
          </p>
        </div>

        {/* Primary Action Buttons */}
        <div className="mt-8 flex flex-wrap items-center gap-3.5">
          <a href="#work" className="editorial-btn-primary">
            <span>Explore Selected Work</span>
            <ArrowDown className="w-4 h-4" />
          </a>
          <Link to="/blog" className="editorial-btn-secondary">
            <span>Read "The Deep End"</span>
            <ArrowUpRight className="w-4 h-4" />
          </Link>
          <a
            href="/assets/ayeleru_cv.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="editorial-btn-secondary"
            download="ayeleru_cv.pdf"
          >
            <FileText className="w-4 h-4 text-ink-muted" />
            <span>Download CV</span>
          </a>
        </div>

        {/* Restrained Architectural Motif: AI -> Systems -> Product -> Production */}
        <div className="mt-14 pt-8 border-t border-border-subtle">
          <div className="text-[11px] font-mono uppercase tracking-widest text-ink-faint font-semibold mb-4">
            End-to-End Systems Lifecycle
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-white border border-border rounded-lg p-3.5 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between text-ink-faint">
                <span className="font-mono text-[11px] font-semibold text-ink">01 / Machine Learning</span>
                <Cpu className="w-4 h-4 text-ink-muted" />
              </div>
              <p className="text-xs text-ink-muted">
                RAG pipelines, custom tokenization, vector search, transformers & PyTorch models.
              </p>
            </div>

            <div className="bg-white border border-border rounded-lg p-3.5 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between text-ink-faint">
                <span className="font-mono text-[11px] font-semibold text-ink">02 / Distributed Backends</span>
                <Server className="w-4 h-4 text-ink-muted" />
              </div>
              <p className="text-xs text-ink-muted">
                NestJS & FastAPI microservices, 200+ REST endpoints, PostgreSQL, SQS queues, and payment gateways.
              </p>
            </div>

            <div className="bg-white border border-border rounded-lg p-3.5 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between text-ink-faint">
                <span className="font-mono text-[11px] font-semibold text-ink">03 / Product Engineering</span>
                <Layers className="w-4 h-4 text-ink-muted" />
              </div>
              <p className="text-xs text-ink-muted">
                Next.js web portals, Capacitor Android bridging, and intuitive, accessible user interfaces.
              </p>
            </div>

            <div className="bg-white border border-border rounded-lg p-3.5 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between text-ink-faint">
                <span className="font-mono text-[11px] font-semibold text-ink">04 / Cloud & Operations</span>
                <Rocket className="w-4 h-4 text-ink-muted" />
              </div>
              <p className="text-xs text-ink-muted">
                Google Play Console rollouts, Docker containerization, AWS Lambda, and CI/CD pipelines.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
export default Hero;

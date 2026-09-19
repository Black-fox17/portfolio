import React from 'react';
import { Award, BookOpen, Terminal, Sparkles, ArrowUpRight } from 'lucide-react';

export const AboutSection: React.FC = () => {
  return (
    <section id="about" className="border-b border-border/60 bg-paper">
      <div className="section-container">
        {/* Section Header */}
        <div className="mb-12">
          <div className="section-heading">
            <span className="w-2 h-2 rounded-full bg-ink" />
            <span>Chapter 08 — About & Engineering Identity</span>
          </div>
          <h2 className="section-title">
            Who I Am as an Engineer
          </h2>
          <p className="section-desc">
            Bridging theoretical machine learning rigor with production software craftsmanship.
          </p>
        </div>

        {/* Narrative & Credentials Grid */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Editorial Narrative */}
          <div className="lg:col-span-2 space-y-5 text-sm sm:text-base text-ink-light leading-relaxed">
            <p>
              I am an <strong className="text-ink font-semibold">AI Systems Engineer and Full-Stack Software Engineer</strong> focused on taking complex ideas from mathematical formulation to reliable software running in production.
            </p>
            <p>
              My journey began in first-principles machine learning and medical computer vision—training deep convolutional networks and decomposing transformer architectures. As I built, I realized that the true frontier of AI lies not merely in fine-tuning models, but in the <strong className="text-ink font-semibold">systems engineering around them</strong>: configurable ingestion pipelines, distributed asynchronous queues, sub-second vector retrieval, and resilient payment and ledger infrastructures.
            </p>
            <p>
              At <strong className="text-ink font-semibold">GoodTenants</strong>, I turned this perspective into reality by architecting and operating the <strong className="text-ink font-semibold">Upward Ecosystem</strong>—shipping two production Android apps on Google Play, engineering 30+ NestJS REST APIs, and designing automated multi-gateway payment failover that ensures continuous operations.
            </p>
            <p>
              I operate across the entire product lifecycle: from data structures and database schema design, to cloud infrastructure and CI/CD automation, down to the micro-interactions and accessibility of cross-platform client interfaces.
            </p>
          </div>

          {/* Sidebar: Formal Background & Certifications */}
          <div className="space-y-4">
            <div className="bg-white border border-border p-6 rounded-xl space-y-4 shadow-subtle">
              <div className="space-y-1">
                <span className="text-xs font-mono uppercase tracking-wider text-ink-faint font-semibold flex items-center gap-1.5">
                  <GraduationCapIcon className="w-3.5 h-3.5 text-ink-muted" />
                  <span>Academic Foundations</span>
                </span>
                <h4 className="text-sm font-bold text-ink font-editorial">
                  University of Ibadan
                </h4>
                <p className="text-xs text-ink-muted">
                  B.Sc. in Computer Science
                </p>
                <p className="text-[11px] font-mono text-ink-faint">
                  Expected 2027 · Ibadan, Nigeria
                </p>
              </div>

              <div className="pt-3 border-t border-border-subtle space-y-2">
                <span className="text-xs font-mono uppercase tracking-wider text-ink-faint font-semibold flex items-center gap-1.5">
                  <Award className="w-3.5 h-3.5 text-ink-muted" />
                  <span>Certifications</span>
                </span>
                <div className="space-y-1.5">
                  <div className="p-2 rounded bg-paper-50 border border-border-subtle text-xs">
                    <span className="font-semibold text-ink block">Deep Learning Specialization</span>
                    <span className="text-[11px] text-ink-muted">DeepLearning.AI</span>
                  </div>
                  <div className="p-2 rounded bg-paper-50 border border-border-subtle text-xs">
                    <span className="font-semibold text-ink block">Transformers & NLP Course</span>
                    <span className="text-[11px] text-ink-muted">Hugging Face</span>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-border-subtle">
                <a
                  href="/assets/ayeleru_cv.pdf"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="editorial-btn-secondary w-full justify-center !text-xs !py-2"
                  download="ayeleru_cv.pdf"
                >
                  <span>Download Full Curriculum Vitae</span>
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

const GraduationCapIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
    <path d="M6 12v5c3 3 9 3 12 0v-5"/>
  </svg>
);

export default AboutSection;

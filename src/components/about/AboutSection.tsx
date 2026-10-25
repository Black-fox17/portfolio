import React, { useState } from 'react';
import { skillCategories } from '../../data/skills';

const corePillars = [
  {
    number: '01',
    title: 'First Principles Over Hype',
    body: 'I evaluate speed requirements and system rules before choosing tools, always checking whether a problem truly needs machine learning or just clean, straightforward logic.'
  },
  {
    number: '02',
    title: 'Solid Architecture & Reliable Data',
    body: 'Managing 500+ APIs across mobile and web apps demands clear organization. I structure systems with organized code modules, clean database models, and safe payment transactions.'
  },
  {
    number: '03',
    title: 'From Code to App Store',
    body: 'Software only matters when real people use it. I handle the entire process from core logic and backend APIs to web portals and live releases on the Apple App Store and Google Play.'
  }
];

export const AboutSection: React.FC = () => {
  const [activeSkillTab, setActiveSkillTab] = useState<number>(0);

  return (
    <section id="story" className="scroll-mt-24">
      {/* Chapter Kicker & Headline */}
      <div className="space-y-3 mb-8">
        <div className="section-kicker">
          <span className="w-2 h-2 rounded-full bg-accent-cobalt" />
          <span>01 / The Story & Craft</span>
        </div>
        <h2 className="text-2xl sm:text-4xl font-bold tracking-[-0.025em] text-ink font-editorial leading-[1.15]">
          A single-page resume lists tools. Here is how I actually think, architect, and ship.
        </h2>
      </div>

      {/* Main Narrative Story */}
      <div className="space-y-5 story-prose">
        <p>
          My path into software engineering started with a desire to understand machine learning from the ground up, not as a black box. While researching medical imaging at the{' '}
          <strong>Institute of Advanced Medical Research and Training (UCH Ibadan)</strong>, I trained computer vision models to identify breast cancer subtypes from biopsy slides. Alongside that, I spent my nights building foundational AI components from scratch in PyTorch, including{' '}
          <strong>text tokenizers, word embeddings, and Transformer language models</strong>. Knowing how these models work internally makes building reliable applications with them second nature.
        </p>

        <p>
          As I transitioned into building enterprise AI and document search systems at{' '}
          <strong>Codygo</strong> and <strong>Tweakrr</strong>, I learned what a resume bullet rarely captures: the hardest part of production AI isn’t the model itself. It is the{' '}
          <strong>software engineering surrounding it</strong>. True reliability comes from background processing pipelines, fast search indexing, and strict input checks that prevent faulty AI responses before they reach a user.
        </p>

        <p>
          Today at <strong>GoodTenants</strong>, I apply that engineering approach as the lead backend and mobile engineer for the{' '}
          <strong>Upward Ecosystem</strong>. I architected a unified server running over <strong>500+ production REST APIs</strong>. The system handles real-time requests from our property manager app and tenant app, while seamlessly syncing with external property management platforms through a structured PostgreSQL database. I also managed Google Play requirements for both apps and led the release process to launch{' '}
          <strong>UpwardPay live on the Apple App Store</strong>.
        </p>
      </div>

      {/* 3 Core Engineering Pillars - Clean Hairline Grid, No AI Icons */}
      <div className="mt-10 pt-10 border-t border-border">
        <div className="flex items-center justify-between mb-5">
          <span className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-ink" />
            <span>Engineering Principles</span>
          </span>
          <span className="text-xs font-mono text-ink-faint">01 / 03</span>
        </div>

        <div className="grid sm:grid-cols-3 bg-paper-100/70 border border-border rounded-xl divide-y sm:divide-y-0 sm:divide-x divide-border overflow-hidden">
          {corePillars.map((pillar) => (
            <div
              key={pillar.number}
              className="p-5 flex flex-col justify-between space-y-3"
            >
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt" />
                  <span className="text-xs font-mono font-bold text-ink-muted">
                    {pillar.number}
                  </span>
                </div>
                <h3 className="text-base font-bold text-ink font-editorial leading-snug">
                  {pillar.title}
                </h3>
                <p className="text-[13.5px] text-ink-muted leading-relaxed">
                  {pillar.body}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive Technical Capabilities Switcher (Dots instead of Checkmarks) */}
      <div className="mt-10 pt-10 border-t border-border">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 mb-5">
          <div>
            <span className="text-xs font-mono uppercase tracking-widest text-ink-muted font-semibold flex items-center gap-2 mb-1">
              <span className="w-1.5 h-1.5 rounded-full bg-ink" />
              <span>Technical Stack & Capabilities</span>
            </span>
            <h3 className="text-xl font-bold text-ink font-editorial">
              What I build with across the stack
            </h3>
          </div>
          {/* Domain Tabs */}
          <div className="flex flex-wrap gap-1.5">
            {skillCategories.map((cat, idx) => (
              <button
                key={cat.title}
                type="button"
                onClick={() => setActiveSkillTab(idx)}
                className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all cursor-pointer ${
                  activeSkillTab === idx
                    ? 'bg-ink text-paper-50 font-semibold'
                    : 'bg-paper-100 border border-border text-ink-muted hover:text-ink'
                }`}
              >
                {cat.title.split('&')[0].trim()}
              </button>
            ))}
          </div>
        </div>

        {/* Active Domain Detail Card */}
        <div className="bg-white border border-border rounded-xl p-6 shadow-subtle">
          <div className="pb-4 mb-4 border-b border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h4 className="text-base font-bold text-ink font-editorial">
                {skillCategories[activeSkillTab].title}
              </h4>
              <p className="text-sm text-ink-muted mt-0.5">
                {skillCategories[activeSkillTab].description}
              </p>
            </div>
            <span className="text-xs font-mono font-semibold text-ink-faint shrink-0">
              0{activeSkillTab + 1} / 0{skillCategories.length}
            </span>
          </div>

          <div className="grid sm:grid-cols-2 gap-3">
            {skillCategories[activeSkillTab].skills.map((skill) => (
              <div
                key={skill.name}
                className="p-3.5 rounded-lg bg-paper/80 border border-border-subtle flex items-start gap-3"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-accent-cobalt mt-2 shrink-0" />
                <div>
                  <div className="text-[13.5px] font-bold text-ink font-mono">
                    {skill.name}
                  </div>
                  {skill.focus && (
                    <div className="text-xs text-ink-muted mt-0.5 leading-relaxed">
                      {skill.focus}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Compact Academic & Credentials Strip (Clean Dots, No AI Icons) */}
      <div className="mt-6 grid sm:grid-cols-2 bg-paper-100/80 border border-border rounded-xl divide-y sm:divide-y-0 sm:divide-x divide-border overflow-hidden">
        <div className="p-4 sm:p-5 space-y-1">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-ink" />
            <span className="text-[11px] font-mono uppercase tracking-wider text-ink-muted font-semibold">
              Education
            </span>
          </div>
          <h4 className="text-sm font-bold text-ink font-editorial pt-0.5">
            B.Sc. Computer Science, University of Ibadan
          </h4>
          <p className="text-xs text-ink-muted">
            Expected 2027 · Systems, Algorithms & Machine Learning Research
          </p>
        </div>

        <div className="p-4 sm:p-5 space-y-1">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-ink" />
            <span className="text-[11px] font-mono uppercase tracking-wider text-ink-muted font-semibold">
              Specialist Certifications
            </span>
          </div>
          <h4 className="text-sm font-bold text-ink font-editorial pt-0.5">
            DeepLearning.AI & Hugging Face
          </h4>
          <p className="text-xs text-ink-muted">
            Deep Learning Specialization · Transformers & NLP Course
          </p>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;

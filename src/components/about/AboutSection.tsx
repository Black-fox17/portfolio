import React from 'react';

export const AboutSection: React.FC = () => {
  return (
    <section id="story" className="scroll-mt-24 space-y-4">
      {/* Chapter Kicker & Headline */}
      <div className="space-y-2">
        <div className="section-kicker">
          <span className="w-1.5 h-1.5 rounded-full bg-neutral-900" />
          <span>01 / The Story &amp; Craft</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-neutral-900 leading-[1.25]">
          How I think, build, and ship software.
        </h2>
      </div>

      {/* Main Narrative Story */}
      <div className="space-y-3.5 story-prose">
        <p>
          I started out wanting to understand how machine learning and neural networks work from first principles, not just calling black-box APIs. That led me into clinical computer vision research at the{' '}
          <strong>Institute of Advanced Medical Research and Training (UCH Ibadan)</strong>, where I trained deep learning models on cancer biopsy slides and wrote foundational PyTorch architectures (tokenizers, KV cache, and Transformers) from scratch.
        </p>

        <p>
          Transitioning into engineering at <strong>Codygo</strong>, I built conversational AI assistants and asynchronous background data pipelines on AWS. It reinforced what truly matters in production AI: the model is only one piece, while real reliability comes from resilient backends, fast retrieval, and strict validation checks.
        </p>

        <p>
          Today at <strong>GoodTenants</strong>, I lead backend and mobile engineering for the{' '}
          <strong>Upward Ecosystem</strong>. I architected our centralized backend handling high-volume concurrent requests across web and mobile clients, and led the release process to ship 3 live apps across the{' '}
          <strong>Apple App Store</strong> and <strong>Google Play Store</strong>, backed by PostgreSQL.
        </p>
      </div>
    </section>
  );
};

export default AboutSection;

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';

export const ContactSection: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'info' | 'error'; text: string } | null>(null);

  const [form, setForm] = useState({
    name: '',
    email: '',
    message: ''
  });

  const handleCopyEmail = () => {
    navigator.clipboard.writeText('ayeleru1234@gmail.com');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const gmailComposeUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=ayeleru1234@gmail.com&su=${encodeURIComponent(
    form.name ? `Engineering Inquiry from ${form.name}` : 'Engineering Opportunity / Inquiry'
  )}&body=${encodeURIComponent(
    form.message ? `${form.message}\n\nFrom: ${form.name} (${form.email})` : ''
  )}`;

  const mailtoUrl = `mailto:ayeleru1234@gmail.com?subject=${encodeURIComponent(
    form.name ? `Engineering Inquiry from ${form.name}` : 'Engineering Opportunity / Inquiry'
  )}&body=${encodeURIComponent(
    form.message ? `${form.message}\n\nFrom: ${form.name} (${form.email})` : ''
  )}`;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setStatusMessage(null);

    try {
      const response = await fetch('https://formsubmit.co/ajax/ayeleru1234@gmail.com', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          name: form.name,
          email: form.email,
          message: form.message,
          _subject: `Portfolio Message from ${form.name} (${form.email})`,
          _template: 'table'
        })
      });

      const data = await response.json();

      if (response.ok || data.success === 'true' || data.success === true) {
        setLoading(false);
        setStatusMessage({
          type: 'success',
          text: 'Message sent! It has been routed directly to ayeleru1234@gmail.com.'
        });
        setForm({ name: '', email: '', message: '' });
      } else {
        throw new Error('FormSubmit error');
      }
    } catch {
      setLoading(false);
      window.location.href = mailtoUrl;
      setStatusMessage({
        type: 'info',
        text: 'Opening your email client with your draft pre-filled to ayeleru1234@gmail.com.'
      });
    }
  };

  return (
    <section id="contact" className="scroll-mt-24">
      {/* Section Header */}
      <div className="space-y-3 mb-8">
        <div className="section-kicker">
          <span className="w-2 h-2 rounded-full bg-neutral-900" />
          <span>05 / Initiate Contact</span>
        </div>
        <h2 className="text-2xl sm:text-4xl font-bold tracking-[-0.025em] text-ink font-editorial leading-[1.15]">
          Let’s Discuss Systems, Products, or Engineering Roles
        </h2>
        <p className="text-[15.5px] sm:text-base text-ink-muted leading-relaxed max-w-2xl">
          Whether you are hiring for an AI Systems or Full-Stack Product Engineering role, or need an architect to take a complex product to production, my inbox is open.
        </p>
      </div>

      {/* Quick Coordinates Strip (Dark Carbon + Tinted Stone, Zero Decorative Icons) */}
      <div className="grid sm:grid-cols-2 gap-3 mb-6">
        <button
          type="button"
          onClick={handleCopyEmail}
          className="p-4 rounded-xl bg-ink text-paper-50 hover:bg-ink-light transition-all flex items-center justify-between group text-left cursor-pointer shadow-subtle"
        >
          <div className="space-y-0.5">
            <span className="text-[10.5px] font-mono text-white/65 uppercase tracking-wider flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-neutral-300" />
              <span>Direct Email (Click to Copy)</span>
            </span>
            <span className="text-sm font-mono font-semibold text-white block">
              ayeleru1234@gmail.com
            </span>
          </div>
          <div className="px-2.5 py-1 rounded bg-white/10 border border-white/15 text-xs font-mono text-white">
            {copied ? 'Copied' : 'Copy'}
          </div>
        </button>

        <a
          href="/assets/ayeleru_cv.pdf"
          target="_blank"
          rel="noopener noreferrer"
          download="ayeleru_cv.pdf"
          className="p-4 rounded-xl bg-neutral-100 border border-neutral-200 hover:border-neutral-400 transition-all flex items-center justify-between group text-left"
        >
          <div className="space-y-0.5">
            <span className="text-[10.5px] font-mono text-neutral-500 uppercase tracking-wider flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-neutral-800" />
              <span>Single-Page Resume</span>
            </span>
            <span className="text-sm font-bold text-neutral-900 group-hover:text-neutral-600 transition-colors block">
              Download Full Curriculum Vitae (PDF)
            </span>
          </div>
          <ArrowUpRight className="w-4 h-4 text-neutral-400 group-hover:text-neutral-900" />
        </a>
      </div>

      {/* Direct Message Form */}
      <div className="bg-white border border-neutral-200 p-6 sm:p-8 rounded-xl shadow-subtle space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-lg font-bold text-neutral-900">
              Send a Direct Message
            </h3>
            <p className="text-xs text-neutral-500">
              Routes directly to <strong className="text-neutral-900 font-mono">ayeleru1234@gmail.com</strong>
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-neutral-300">·</span>
            <a
              href={gmailComposeUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-neutral-500 hover:text-neutral-900 inline-flex items-center gap-1 underline decoration-neutral-300 underline-offset-4"
            >
              <span>Compose in Gmail</span>
              <ArrowUpRight className="w-3 h-3" />
            </a>
          </div>
        </div>

        {statusMessage && (
          <div
            className={`p-3.5 rounded-lg text-xs font-medium ${
              statusMessage.type === 'success'
                ? 'bg-emerald-50 border border-emerald-200 text-emerald-800'
                : statusMessage.type === 'info'
                ? 'bg-blue-50 border border-blue-200 text-blue-800'
                : 'bg-red-50 border border-red-200 text-red-800'
            }`}
          >
            {statusMessage.text}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label htmlFor="name" className="text-xs font-mono font-semibold text-ink">
                Your Name *
              </label>
              <input
                id="name"
                type="text"
                name="name"
                value={form.name}
                onChange={handleChange}
                required
                placeholder="e.g. Alex Johnson"
                className="w-full px-3.5 py-2.5 rounded-lg border border-border bg-paper/70 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:border-ink focus:bg-white transition-colors"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="email" className="text-xs font-mono font-semibold text-ink">
                Your Email *
              </label>
              <input
                id="email"
                type="email"
                name="email"
                value={form.email}
                onChange={handleChange}
                required
                placeholder="e.g. alex@company.com"
                className="w-full px-3.5 py-2.5 rounded-lg border border-border bg-paper/70 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:border-ink focus:bg-white transition-colors"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label htmlFor="message" className="text-xs font-mono font-semibold text-ink">
              Message / Role or Project Context *
            </label>
            <textarea
              id="message"
              name="message"
              value={form.message}
              onChange={handleChange}
              required
              rows={4}
              placeholder="Tell me about your team, system requirements, or engineering role..."
              className="w-full px-3.5 py-2.5 rounded-lg border border-border bg-paper/70 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:border-ink focus:bg-white transition-colors resize-y"
            />
          </div>

          <div className="pt-1 flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs font-mono text-ink-faint flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-neutral-700" />
              <span>Typically responds within 24 hours.</span>
            </p>
            <button
              type="submit"
              disabled={loading}
              className="editorial-btn-primary !py-2.5 !px-5 disabled:opacity-50 cursor-pointer"
            >
              <span>{loading ? 'Sending Message...' : 'Send Message →'}</span>
            </button>
          </div>
        </form>
      </div>
    </section>
  );
};

export default ContactSection;

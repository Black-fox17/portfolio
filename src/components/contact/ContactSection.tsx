import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import Toast, { ToastData } from '../common/Toast';
import { PORTFOLIO_EMAIL } from '../../constants/contact';

export const ContactSection: React.FC = () => {
  const [copied, setCopied] = useState(false);
  const [toast, setToast] = useState<ToastData | null>(null);

  const [form, setForm] = useState({
    name: '',
    email: '',
    message: ''
  });

  const handleCopyEmail = () => {
    navigator.clipboard.writeText(PORTFOLIO_EMAIL);
    setCopied(true);
    setToast({
      type: 'info',
      message: `Copied ${PORTFOLIO_EMAIL} to clipboard.`
    });
    setTimeout(() => setCopied(false), 2000);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const emailSubject = encodeURIComponent(
    form.name ? `Engineering Inquiry from ${form.name}` : 'Engineering Opportunity / Inquiry'
  );
  const emailBody = encodeURIComponent(
    form.message
      ? `${form.message}\n\n---\nSender: ${form.name || 'Anonymous'}\nEmail: ${form.email || 'Not provided'}`
      : ''
  );

  const mailtoUrl = `mailto:${PORTFOLIO_EMAIL}?subject=${emailSubject}&body=${emailBody}`;
  const gmailComposeUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=${PORTFOLIO_EMAIL}&su=${emailSubject}&body=${emailBody}`;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Directly open default mail client with formatted draft
    window.location.href = mailtoUrl;

    setToast({
      type: 'success',
      message: 'Opening your default email app with draft pre-filled.',
      action: {
        label: 'Or open in Gmail Web',
        onClick: () => {
          window.open(gmailComposeUrl, '_blank');
        }
      }
    });
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
              <span>Pre-fills directly into your email client.</span>
            </p>
            <button
              type="submit"
              className="editorial-btn-primary !py-2.5 !px-5 cursor-pointer"
            >
              <span>Send Message →</span>
            </button>
          </div>
        </form>
      </div>

      {/* Floating Toast Notification */}
      <Toast toast={toast} onClose={() => setToast(null)} />
    </section>
  );
};

export default ContactSection;

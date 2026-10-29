import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft, ArrowUpRight, Copy, Check } from 'lucide-react';
import { motion } from 'framer-motion';
import Navbar from '../components/layout/Navbar';
import Toast, { ToastData } from '../components/common/Toast';
import { PORTFOLIO_EMAIL } from '../constants/contact';

export const Contact: React.FC = () => {
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
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="min-h-screen bg-white text-neutral-900 selection:bg-neutral-900 selection:text-white flex flex-col"
    >
      <Helmet>
        <title>Contact | Abdulsalam Ayeleru</title>
        <meta
          name="description"
          content="Get in touch with Abdulsalam Oluwaseun Ayeleru for software engineering roles, distributed systems, and AI architecture."
        />
      </Helmet>

      <Navbar />

      <main className="flex-1 max-w-[680px] w-full mx-auto px-5 sm:px-6 pt-20 sm:pt-28 pb-20 space-y-10">
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
            Contact.
          </h1>
          <p className="text-sm sm:text-base text-neutral-600 leading-relaxed">
            Whether you have an engineering opportunity, a technical question, or an interesting project, my inbox is always open.
          </p>
        </header>

        {/* Quick Email Coordinates */}
        <div className="flex flex-wrap items-center gap-4 text-sm pt-1">
          <button
            type="button"
            onClick={handleCopyEmail}
            className="inline-flex items-center gap-2 text-neutral-900 font-medium hover:underline underline-offset-4 decoration-neutral-300 transition-colors cursor-pointer"
          >
            <span>ayeleruseun@gmail.com</span>
            {copied ? (
              <span className="inline-flex items-center gap-1 text-xs text-neutral-500">
                <Check className="w-3.5 h-3.5 text-neutral-700" />
                <span>copied</span>
              </span>
            ) : (
              <Copy className="w-3.5 h-3.5 text-neutral-400" />
            )}
          </button>
          <span className="text-neutral-300">/</span>
          <a
            href="https://x.com/ayeleruseun"
            target="_blank"
            rel="noopener noreferrer"
            className="text-neutral-500 hover:text-neutral-900 transition-colors"
          >
            X(Twitter) ↗
          </a>
          <span className="text-neutral-300">/</span>
          <a
            href="https://github.com/Black-fox17"
            target="_blank"
            rel="noopener noreferrer"
            className="text-neutral-500 hover:text-neutral-900 transition-colors"
          >
            GitHub ↗
          </a>
          <span className="text-neutral-300">/</span>
          <a
            href="https://www.linkedin.com/in/abdulsalam-ayeleru-384976219/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-neutral-500 hover:text-neutral-900 transition-colors"
          >
            LinkedIn ↗
          </a>
        </div>

        {/* Message Form */}
        <div className="bg-white border border-neutral-200 p-6 sm:p-7 rounded-xl shadow-subtle space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-neutral-100 pb-3">
            <div>
              <h2 className="text-base font-semibold text-neutral-900">
                Send a Message
              </h2>
              <p className="text-xs text-neutral-500">
                Direct draft to {PORTFOLIO_EMAIL}
              </p>
            </div>
            <a
              href={gmailComposeUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-neutral-500 hover:text-neutral-900 inline-flex items-center gap-1 underline decoration-neutral-300 underline-offset-4"
            >
              <span>Compose in Gmail Web</span>
              <ArrowUpRight className="w-3 h-3" />
            </a>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4 pt-1">
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label htmlFor="name" className="text-xs font-mono text-neutral-500">
                  Your Name
                </label>
                <input
                  id="name"
                  type="text"
                  name="name"
                  required
                  value={form.name}
                  onChange={handleChange}
                  placeholder="e.g. Alex Chen"
                  className="w-full px-3 py-2 rounded-lg border border-neutral-200 text-sm text-neutral-900 placeholder:text-neutral-400 focus:outline-none focus:border-neutral-900 transition-colors"
                />
              </div>

              <div className="space-y-1.5">
                <label htmlFor="email" className="text-xs font-mono text-neutral-500">
                  Your Email
                </label>
                <input
                  id="email"
                  type="email"
                  name="email"
                  required
                  value={form.email}
                  onChange={handleChange}
                  placeholder="e.g. alex@company.com"
                  className="w-full px-3 py-2 rounded-lg border border-neutral-200 text-sm text-neutral-900 placeholder:text-neutral-400 focus:outline-none focus:border-neutral-900 transition-colors"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label htmlFor="message" className="text-xs font-mono text-neutral-500">
                Message
              </label>
              <textarea
                id="message"
                name="message"
                required
                rows={4}
                value={form.message}
                onChange={handleChange}
                placeholder="Tell me about your team, role, or project..."
                className="w-full px-3 py-2 rounded-lg border border-neutral-200 text-sm text-neutral-900 placeholder:text-neutral-400 focus:outline-none focus:border-neutral-900 transition-colors resize-y"
              />
            </div>

            <div className="pt-1 flex items-center justify-between">
              <span className="text-xs text-neutral-400">
                Pre-fills directly into your email client.
              </span>
              <button
                type="submit"
                className="px-4 py-2 rounded-lg bg-neutral-900 text-white text-xs font-mono font-medium hover:bg-neutral-800 transition-colors cursor-pointer"
              >
                Send Message →
              </button>
            </div>
          </form>
        </div>

        {/* Footer */}
        <footer className="pt-12 border-t border-neutral-100 text-xs text-neutral-400 flex items-center justify-between">
          <span>© {new Date().getFullYear()} Abdulsalam Oluwaseun Ayeleru</span>
          <Link to="/" className="hover:text-neutral-900 transition-colors">
            Back to Home ↑
          </Link>
        </footer>
      </main>

      {/* Floating Toast Notification */}
      <Toast toast={toast} onClose={() => setToast(null)} />
    </motion.div>
  );
};

export default Contact;

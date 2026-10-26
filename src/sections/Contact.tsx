import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { ArrowLeft, ArrowUpRight, Copy, Check } from 'lucide-react';
import { motion } from 'framer-motion';
import Navbar from '../components/layout/Navbar';

export const Contact: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

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
        })
      });

      const data = await response.json();
      if (response.ok || data.success === 'true' || data.success === true) {
        setStatusMessage({
          type: 'success',
          text: 'Thank you! Your message was sent successfully. I will get back to you shortly.'
        });
        setForm({ name: '', email: '', message: '' });
      } else {
        throw new Error('Failed to send');
      }
    } catch {
      setStatusMessage({
        type: 'error',
        text: 'Something went wrong sending the message. Please email me directly at ayeleru1234@gmail.com'
      });
    } finally {
      setLoading(false);
    }
  };

  const gmailComposeUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=ayeleru1234@gmail.com&su=${encodeURIComponent(
    form.name ? `Engineering Inquiry from ${form.name}` : 'Engineering Opportunity / Inquiry'
  )}&body=${encodeURIComponent(
    form.message ? `${form.message}\n\nFrom: ${form.name} (${form.email})` : ''
  )}`;

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
            <span>ayeleru1234@gmail.com</span>
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
            href="https://x.com/Ayeleru_Salam"
            target="_blank"
            rel="noopener noreferrer"
            className="text-neutral-500 hover:text-neutral-900 transition-colors"
          >
            Twitter ↗
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
                Directly sent to ayeleru1234@gmail.com
              </p>
            </div>
            <a
              href={gmailComposeUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-neutral-500 hover:text-neutral-900 inline-flex items-center gap-1 underline decoration-neutral-300 underline-offset-4"
            >
              <span>Compose in Gmail</span>
              <ArrowUpRight className="w-3 h-3" />
            </a>
          </div>

          {statusMessage && (
            <div
              className={`p-3 rounded text-xs leading-relaxed ${
                statusMessage.type === 'success'
                  ? 'bg-neutral-100 text-neutral-900'
                  : 'bg-neutral-100 text-neutral-700'
              }`}
            >
              {statusMessage.text}
            </div>
          )}

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
                Typically responds within 24 hours.
              </span>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 rounded-lg bg-neutral-900 text-white text-xs font-mono font-medium hover:bg-neutral-800 transition-colors disabled:opacity-50 cursor-pointer"
              >
                {loading ? 'Sending...' : 'Send Message →'}
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
    </motion.div>
  );
};

export default Contact;

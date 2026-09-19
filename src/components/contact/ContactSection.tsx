import React, { useState } from 'react';
import { Mail, Copy, Check, Send, Github, ArrowUpRight, FileText, ExternalLink, Sparkles } from 'lucide-react';

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

  // Pre-filled Gmail Compose URL
  const gmailComposeUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=ayeleru1234@gmail.com&su=${encodeURIComponent(
    form.name ? `Engineering Inquiry from ${form.name}` : 'Engineering Opportunity / Inquiry'
  )}&body=${encodeURIComponent(
    form.message ? `${form.message}\n\nFrom: ${form.name} (${form.email})` : ''
  )}`;

  // Pre-filled Native Mailto Link
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
      // Robust, zero-signup FormSubmit AJAX endpoint (delivers directly to your inbox)
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
          text: 'Message sent successfully! It has been routed directly to ayeleru1234@gmail.com.'
        });
        setForm({ name: '', email: '', message: '' });
      } else {
        throw new Error('FormSubmit error');
      }
    } catch (error) {
      // Graceful fallback: Open default email client with draft pre-filled
      setLoading(false);
      window.location.href = mailtoUrl;
      setStatusMessage({
        type: 'info',
        text: 'Opening your email client with your draft message pre-filled to ayeleru1234@gmail.com.'
      });
    }
  };

  return (
    <section id="contact" className="border-b border-border/60 bg-paper-50">
      <div className="section-container">
        {/* Section Header */}
        <div className="mb-12">
          <div className="section-heading">
            <span className="w-2 h-2 rounded-full bg-ink" />
            <span>Chapter 09 — Initiate Contact</span>
          </div>
          <h2 className="section-title">
            Let's Discuss Systems, Products or Engineering Roles
          </h2>
          <p className="section-desc">
            Whether you're looking for an AI systems engineer to architect RAG and retrieval pipelines, build scalable distributed backends, or deliver a multi-platform SaaS product, I'd love to connect.
          </p>
        </div>

        <div className="grid lg:grid-cols-5 gap-8">
          {/* Left Column: Direct Coordinates & Direct Actions */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white border border-border p-6 rounded-xl space-y-5 shadow-subtle">
              <span className="text-xs font-mono uppercase tracking-wider text-ink-faint font-semibold block">
                Direct Coordinates
              </span>

              {/* One-Click Copy Email */}
              <div className="space-y-3">
                <button
                  type="button"
                  onClick={handleCopyEmail}
                  className="w-full p-3.5 rounded-lg bg-paper-50 border border-border hover:border-ink-muted/40 transition-all flex items-center justify-between group text-left cursor-pointer shadow-subtle"
                >
                  <div className="flex items-center gap-2.5">
                    <Mail className="w-4 h-4 text-ink-muted group-hover:text-ink" />
                    <div>
                      <span className="text-[11px] font-mono text-ink-faint block">Direct Email (Click to Copy)</span>
                      <span className="text-sm font-mono font-semibold text-ink">ayeleru1234@gmail.com</span>
                    </div>
                  </div>
                  <div className="p-1.5 rounded bg-white border border-border-subtle text-ink-muted group-hover:text-ink">
                    {copied ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
                  </div>
                </button>

                {/* Instant Email Direct Links */}
                <div className="grid grid-cols-2 gap-2">
                  <a
                    href={gmailComposeUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="editorial-btn-secondary !py-2 !px-3 !text-xs justify-center !gap-1.5"
                  >
                    <span>Open in Gmail</span>
                    <ExternalLink className="w-3 h-3 text-ink-faint" />
                  </a>
                  <a
                    href={mailtoUrl}
                    className="editorial-btn-secondary !py-2 !px-3 !text-xs justify-center !gap-1.5"
                  >
                    <span>Default Mail App</span>
                    <ArrowUpRight className="w-3 h-3 text-ink-faint" />
                  </a>
                </div>

                {/* Current Status */}
                <div className="p-3.5 rounded-lg bg-paper-50 border border-border text-xs space-y-1">
                  <div className="flex items-center gap-2 text-ink font-medium">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-subtle" />
                    <span>Current Availability</span>
                  </div>
                  <p className="text-ink-muted leading-relaxed text-[11px]">
                    Open to full-time engineering roles, AI systems engineering contracts, and high-impact distributed teams.
                  </p>
                </div>
              </div>

              {/* Profiles & Resume */}
              <div className="pt-3 border-t border-border-subtle space-y-2">
                <span className="text-xs font-mono uppercase tracking-wider text-ink-faint font-semibold block">
                  Networks & Code
                </span>
                <div className="space-y-1.5">
                  <a
                    href="https://github.com/Black-fox17"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2 rounded hover:bg-paper-100 text-xs text-ink transition-colors"
                  >
                    <span className="flex items-center gap-2">
                      <Github className="w-3.5 h-3.5" />
                      <span>GitHub (@Black-fox17)</span>
                    </span>
                    <ArrowUpRight className="w-3.5 h-3.5 text-ink-faint" />
                  </a>
                  <a
                    href="https://x.com/Ayeleru_Salam"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2 rounded hover:bg-paper-100 text-xs text-ink transition-colors"
                  >
                    <span>X / Twitter (@Ayeleru_Salam)</span>
                    <ArrowUpRight className="w-3.5 h-3.5 text-ink-faint" />
                  </a>
                  <a
                    href="/assets/ayeleru_cv.pdf"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2 rounded hover:bg-paper-100 text-xs text-ink transition-colors font-medium"
                    download="ayeleru_cv.pdf"
                  >
                    <span className="flex items-center gap-2">
                      <FileText className="w-3.5 h-3.5" />
                      <span>Download Latest Resume (PDF)</span>
                    </span>
                    <ArrowUpRight className="w-3.5 h-3.5 text-ink-faint" />
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Contact Dispatch Form */}
          <div className="lg:col-span-3">
            <div className="bg-white border border-border p-6 sm:p-8 rounded-xl shadow-subtle space-y-5">
              <div className="space-y-1">
                <h3 className="text-lg font-bold text-ink font-editorial">
                  Send a Direct Message
                </h3>
                <p className="text-xs text-ink-muted">
                  Fill out the form to route directly to <strong className="text-ink">ayeleru1234@gmail.com</strong>.
                </p>
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
                    <label htmlFor="name" className="text-xs font-mono font-medium text-ink">
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
                      className="w-full px-3.5 py-2.5 rounded-lg border border-border bg-paper-50 text-xs text-ink placeholder:text-ink-faint focus:outline-none focus:border-ink focus:bg-white transition-colors"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label htmlFor="email" className="text-xs font-mono font-medium text-ink">
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
                      className="w-full px-3.5 py-2.5 rounded-lg border border-border bg-paper-50 text-xs text-ink placeholder:text-ink-faint focus:outline-none focus:border-ink focus:bg-white transition-colors"
                    />
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label htmlFor="message" className="text-xs font-mono font-medium text-ink">
                    Message / Opportunity Scope *
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    value={form.message}
                    onChange={handleChange}
                    required
                    rows={5}
                    placeholder="Tell me about your project, team, or engineering requirements..."
                    className="w-full px-3.5 py-2.5 rounded-lg border border-border bg-paper-50 text-xs text-ink placeholder:text-ink-faint focus:outline-none focus:border-ink focus:bg-white transition-colors resize-y"
                  />
                </div>

                <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                  <button
                    type="submit"
                    disabled={loading}
                    className="editorial-btn-primary flex-1 !py-2.5 !text-xs disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer shadow-sm"
                  >
                    {loading ? (
                      <span>Sending Message...</span>
                    ) : (
                      <>
                        <span>Send Message</span>
                        <Send className="w-3.5 h-3.5" />
                      </>
                    )}
                  </button>

                  <a
                    href={gmailComposeUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="editorial-btn-secondary !py-2.5 !text-xs justify-center shrink-0"
                  >
                    <span>Or compose in Gmail</span>
                    <ExternalLink className="w-3.5 h-3.5 text-ink-faint" />
                  </a>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
export default ContactSection;

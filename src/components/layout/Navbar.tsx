import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { FileText, Menu, X, ArrowUpRight, Github, Mail } from 'lucide-react';

interface NavItem {
  name: string;
  href: string;
}

const navItems: NavItem[] = [
  { name: 'Work', href: '#work' },
  { name: 'Experience', href: '#experience' },
  { name: 'Philosophy', href: '#philosophy' },
  { name: 'Stack', href: '#skills' },
  { name: 'Writing', href: '/blog' },
  { name: 'About', href: '#about' },
  { name: 'Contact', href: '#contact' },
];

export const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleNavClick = (href: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    setMobileMenuOpen(false);

    if (href.startsWith('/')) {
      navigate(href);
      return;
    }

    if (location.pathname !== '/') {
      navigate('/');
      setTimeout(() => {
        const el = document.querySelector(href);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    } else {
      const el = document.querySelector(href);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-40 transition-all duration-300 ${
        isScrolled
          ? 'bg-paper/95 backdrop-blur-md border-b border-border/80 shadow-subtle'
          : 'bg-paper/80 backdrop-blur-sm border-b border-border/40'
      }`}
    >
      <div className="max-w-5xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between gap-4">
        {/* Brand / Name */}
        <Link
          to="/"
          className="group flex flex-col items-start shrink-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-ink"
        >
          <span className="font-semibold text-sm tracking-tight text-ink font-editorial group-hover:text-ink-light transition-colors whitespace-nowrap">
            Abdulsalam Ayeleru
          </span>
          <span className="text-[11px] font-mono text-ink-faint hidden sm:inline-block whitespace-nowrap">
            AI Systems & Full-Stack
          </span>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="hidden md:flex items-center gap-4 lg:gap-6 shrink-0" aria-label="Main Navigation">
          {navItems.map((item) => (
            <a
              key={item.name}
              href={item.href}
              onClick={handleNavClick(item.href)}
              className="text-xs font-medium text-ink-muted hover:text-ink transition-colors whitespace-nowrap py-1"
            >
              {item.name}
            </a>
          ))}
        </nav>

        {/* Desktop Actions */}
        <div className="hidden sm:flex items-center gap-2 shrink-0">
          <a
            href="https://github.com/Black-fox17"
            target="_blank"
            rel="noopener noreferrer"
            className="p-1.5 text-ink-muted hover:text-ink transition-colors rounded-md hover:bg-paper-100 flex items-center justify-center"
            aria-label="GitHub Profile"
          >
            <Github className="w-4 h-4" />
          </a>
          <a
            href="/assets/ayeleru_cv.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="editorial-btn-secondary !py-1.5 !px-3 !text-xs !gap-1.5 whitespace-nowrap"
            download="ayeleru_cv.pdf"
          >
            <FileText className="w-3.5 h-3.5 text-ink-muted" />
            <span>CV</span>
          </a>
          <a
            href="#contact"
            onClick={handleNavClick('#contact')}
            className="editorial-btn-primary !py-1.5 !px-3.5 !text-xs !gap-1.5 whitespace-nowrap"
          >
            <Mail className="w-3.5 h-3.5 text-paper-200" />
            <span>Get in touch</span>
          </a>
        </div>

        {/* Mobile Menu Toggle Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 text-ink rounded-lg hover:bg-paper-100 focus:outline-none shrink-0"
          aria-label={mobileMenuOpen ? 'Close Menu' : 'Open Menu'}
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-white border-b border-border px-6 py-6 space-y-4 shadow-elevated animate-in fade-in slide-in-from-top-2 duration-200">
          <nav className="flex flex-col space-y-3">
            {navItems.map((item) => (
              <a
                key={item.name}
                href={item.href}
                onClick={handleNavClick(item.href)}
                className="text-sm font-medium text-ink-muted hover:text-ink py-1.5 border-b border-border-subtle"
              >
                {item.name}
              </a>
            ))}
          </nav>
          <div className="pt-3 flex flex-col gap-2.5">
            <a
              href="/assets/ayeleru_cv.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="editorial-btn-secondary w-full justify-center !text-xs"
              download="ayeleru_cv.pdf"
            >
              <FileText className="w-4 h-4" />
              <span>Download CV (PDF)</span>
            </a>
            <div className="flex gap-2">
              <a
                href="https://github.com/Black-fox17"
                target="_blank"
                rel="noopener noreferrer"
                className="editorial-btn-secondary flex-1 justify-center !text-xs"
              >
                <Github className="w-4 h-4" />
                <span>GitHub</span>
              </a>
              <a
                href="https://x.com/Ayeleru_Salam"
                target="_blank"
                rel="noopener noreferrer"
                className="editorial-btn-secondary flex-1 justify-center !text-xs"
              >
                <span>X / Twitter</span>
                <ArrowUpRight className="w-3.5 h-3.5" />
              </a>
            </div>
            <a
              href="#contact"
              onClick={handleNavClick('#contact')}
              className="editorial-btn-primary w-full justify-center !text-xs !py-2.5 mt-1"
            >
              <Mail className="w-3.5 h-3.5" />
              <span>Get in touch</span>
            </a>
          </div>
        </div>
      )}
    </header>
  );
};
export default Navbar;

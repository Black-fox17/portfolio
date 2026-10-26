import React, { useState } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

interface NavItem {
  name: string;
  href: string;
}

const navItems: NavItem[] = [
  { name: 'Home', href: '/' },
  { name: 'Projects', href: '/projects' },
  { name: 'Experience', href: '/experience' },
  { name: 'Writing', href: '/blog' },
  { name: 'Contact', href: '/contact' },
];

export const Navbar: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const handleNavClick = (href: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    setMobileMenuOpen(false);

    if (location.pathname === href) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }

    navigate(href);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-40 bg-white/90 backdrop-blur-sm border-b border-neutral-100">
      <div className="max-w-[680px] mx-auto px-6 h-16 flex items-center justify-between">
        <Link
          to="/"
          onClick={(e) => {
            setMobileMenuOpen(false);
            if (location.pathname === '/') {
              e.preventDefault();
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }
          }}
          className="font-bold text-base tracking-tight text-neutral-900 hover:text-neutral-700 transition-colors"
        >
          Abdulsalam.
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden sm:flex items-center gap-6 text-sm">
          {navItems.map((item) => {
            const isHomeActive = item.href === '/' && location.pathname === '/';
            const isProjectsActive = item.href === '/projects' && location.pathname === '/projects';
            const isExperienceActive = item.href === '/experience' && location.pathname === '/experience';
            const isWritingActive = item.href === '/blog' && location.pathname.startsWith('/blog');
            const isContactActive = item.href === '/contact' && location.pathname === '/contact';
            const isActive = isHomeActive || isProjectsActive || isExperienceActive || isWritingActive || isContactActive;

            return (
              <a
                key={item.name}
                href={item.href}
                onClick={handleNavClick(item.href)}
                className={`relative py-1 transition-colors ${
                  isActive
                    ? 'text-neutral-900 font-medium'
                    : 'text-neutral-500 hover:text-neutral-900'
                }`}
              >
                {item.name}
                {isActive && (
                  <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-neutral-900" />
                )}
              </a>
            );
          })}
        </nav>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="sm:hidden p-1.5 text-neutral-700 rounded hover:bg-neutral-100 focus:outline-none"
          aria-label={mobileMenuOpen ? 'Close Menu' : 'Open Menu'}
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="sm:hidden bg-white border-b border-neutral-200 px-6 py-4 space-y-2">
          {navItems.map((item) => (
            <a
              key={item.name}
              href={item.href}
              onClick={handleNavClick(item.href)}
              className="block text-sm text-neutral-600 hover:text-neutral-900 py-2 border-b border-neutral-100 last:border-b-0"
            >
              {item.name}
            </a>
          ))}
        </div>
      )}
    </header>
  );
};

export default Navbar;

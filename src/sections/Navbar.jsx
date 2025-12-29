import React from 'react';
import { useState } from 'react';
import { navLinks } from '../constants/index.ts';
import { useLocation, useNavigate } from 'react-router-dom';

const NavItems = ({ onClick = () => {} }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const handleNav = (href) => (e) => {
    e.preventDefault();
    onClick();

    // If href is a full path
    if (href.startsWith('/')) {
      navigate(href);
      return;
    }

    // Hash navigation (#about, #work, etc.)
    if (location.pathname !== '/') {
      // Go home first, then scroll
      navigate('/');
      setTimeout(() => {
        const el = document.querySelector(href);
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }, 50);
    } else {
      // Already on home → scroll immediately
      const el = document.querySelector(href);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <ul className="nav-ul">
      {navLinks.map((item) => (
        <li key={item.id} className="nav-li">
          <a
            href={item.href}
            className="nav-li_a"
            onClick={handleNav(item.href)}
          >
            {item.name}
          </a>
        </li>
      ))}
    </ul>
  );
};



function Navbar(){
    const [isOpen, setIsOpen] = useState(false);

    const toggleMenu = () => setIsOpen(!isOpen);
    const closeMenu = () => setIsOpen(false);
    
    return(
        <header className='fixed top-0 left-0 right-0 z-50 bg-black/90'>
            <div className='max-w-7xl mx-auto'>
                <div className='flex justify-between items-center py-5 nx-auto c-space'>
                    <a href = '/' className='text-neutral-400 fon-bold text-xl hover:text-white transition duration-300'>
                        Abdulsalam
                    </a>
                    <button
                        onClick={toggleMenu}
                        className="text-neutral-400 hover:text-white focus:outline-none sm:hidden flex"
                        aria-label="Toggle menu">
                        <img src={isOpen ? 'assets/close.svg' : 'assets/menu.svg'} alt="toggle" className="w-6 h-6" />
                    </button>

                    <nav className="sm:flex hidden">
                        <NavItems />
                    </nav>
                </div>
            </div>
            <div className={`nav-sidebar ${isOpen ? 'max-h-screen' : 'max-h-0'}`}>
                <nav className="p-5">
                    <NavItems onClick={closeMenu} />
                </nav>
            </div>

        </header>
    )
}
export default Navbar;
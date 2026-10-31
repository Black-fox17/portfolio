import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';
import { AnimatePresence } from 'framer-motion';
import Main from './sections/main';
import Projects from './sections/Projects';
import Contact from './sections/Contact';
import ExperiencePage from './sections/ExperiencePage';
import BlogList from './sections/BlogList';
import BlogPost from './sections/BlogPost';
import { trackPageView } from './constants/analytics';

if (typeof window !== 'undefined' && 'scrollRestoration' in window.history) {
  window.history.scrollRestoration = 'manual';
}

const AnalyticsTracker: React.FC = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    trackPageView(pathname);
  }, [pathname]);

  return null;
};

const AnimatedRoutes: React.FC = () => {
  const location = useLocation();

  return (
    <AnimatePresence
      mode="wait"
      initial={false}
      onExitComplete={() => {
        window.scrollTo(0, 0);
      }}
    >
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Main />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/experience" element={<ExperiencePage />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/blog" element={<BlogList />} />
        <Route path="/blog/:slug" element={<BlogPost />} />
        {/* Catch-all fallback */}
        <Route path="*" element={<Main />} />
      </Routes>
    </AnimatePresence>
  );
};

export const App: React.FC = () => {
  return (
    <HelmetProvider>
      <Router>
        <AnalyticsTracker />
        <AnimatedRoutes />
      </Router>
    </HelmetProvider>
  );
};

export default App;

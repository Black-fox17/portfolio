import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';
import Main from './sections/main';
import Research from './sections/Research';
import BlogList from './sections/BlogList';
import BlogPost from './sections/BlogPost';
import { trackPageView } from './constants/analytics';

const ScrollToTop: React.FC = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
};

const AnalyticsTracker: React.FC = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    trackPageView();
  }, [pathname]);

  return null;
};

export const App: React.FC = () => {
  return (
    <HelmetProvider>
      <Router>
        <ScrollToTop />
        <AnalyticsTracker />
        <Routes>
          <Route path="/" element={<Main />} />
          <Route path="/research" element={<Research />} />
          <Route path="/blog" element={<BlogList />} />
          <Route path="/blog/:id" element={<BlogPost />} />
          {/* Catch-all fallback */}
          <Route path="*" element={<Main />} />
        </Routes>
      </Router>
    </HelmetProvider>
  );
};

export default App;

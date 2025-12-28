import React from 'react';
import Main from './sections/main.jsx'
import Research from './sections/Research.jsx'
import BlogList from './sections/BlogList.tsx'
import BlogPost from './sections/BlogPost.tsx'
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { trackPageView } from "./constants/analytics.ts"
import { useEffect } from 'react';

function App() {
  useEffect(() => {
    trackPageView();
  }, []); // Tracks on initial load

  return (
    <main className="max-w-7xl mx-auto text-center">
      <Router>
        <Routes>
          <Route path='/' element={<Main />} />
          <Route path='/research' element={<Research />} />
          <Route path='/blog' element={<BlogList />} />
          <Route path='/blog/:id' element={<BlogPost />} />
        </Routes>
      </Router>
    </main>
  )
}

export default App;


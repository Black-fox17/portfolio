import React from 'react';
import Navbar from './sections/Navbar.jsx';
import Hero from './sections/Hero.jsx';
import About from './sections/About.jsx';
import Project from './sections/Project.jsx'
import Contact from './sections/Contact.jsx'
import Footer from './sections/Footer.jsx'
import Skills from './sections/Skills.jsx'
import Experience from './sections/Experience.jsx';

function App (){
  return(
    <main className="max-w-7xl mx-auto text-center">
      <Navbar />
      <Hero />
      <About />
      <Project />
      <Experience/>
      <Skills />
      <Contact />
      <Footer />
    </main>
  )
}

export default App;
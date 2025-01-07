import React from 'react';
import Navbar from './Navbar.jsx';
import Hero from './Hero.jsx';
import About from './About.jsx';
import Project from './Project.jsx'
import Contact from './Contact.jsx'
import Footer from './Footer.jsx'
import Skills from './Skills.jsx'
import Experience from './Experience.jsx';

function Main (){
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

export default Main;

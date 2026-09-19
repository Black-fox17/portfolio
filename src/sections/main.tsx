import React from 'react';
import Navbar from '../components/layout/Navbar';
import Hero from '../components/hero/Hero';
import SelectedWork from '../components/work/SelectedWork';
import ExperienceTimeline from '../components/experience/ExperienceTimeline';
import HowIBuild from '../components/philosophy/HowIBuild';
import TechnicalIndex from '../components/skills/TechnicalIndex';
import TheDeepEnd from '../components/writing/TheDeepEnd';
import Leadership from '../components/leadership/Leadership';
import AboutSection from '../components/about/AboutSection';
import ContactSection from '../components/contact/ContactSection';
import Footer from '../components/layout/Footer';

export const Main: React.FC = () => {
  return (
    <div className="min-h-screen bg-paper text-ink flex flex-col selection:bg-ink selection:text-paper">
      <Navbar />
      <main className="flex-1">
        <Hero />
        <SelectedWork />
        <ExperienceTimeline />
        <HowIBuild />
        <TechnicalIndex />
        <TheDeepEnd />
        <Leadership />
        <AboutSection />
        <ContactSection />
      </main>
      <Footer />
    </div>
  );
};

export default Main;

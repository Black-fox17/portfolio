import React from 'react';
import { motion } from 'framer-motion';
import { skills } from '../constants/index.ts';
import { FileDown } from 'lucide-react';


const SkillCard = ({ skill }) => (
  <motion.div
    whileHover={{ scale: 1.05 }}
    className="bg-black-200 border border-black-300 rounded-lg p-4 flex flex-col items-center gap-3 hover:bg-black-300 transition-colors duration-300"
  >
    <div className="w-16 h-16 rounded-lg bg-black-300 p-3 flex items-center justify-center">
      <img
        src={skill.img}
        alt={skill.name}
        className="w-full h-full object-contain"
      />
    </div>
    <span className="text-white text-sm font-medium">{skill.name}</span>
  </motion.div>
);

const Skills = () => {
    const handleDownload = () => {
        // Replace with your actual resume PDF URL
        const resumeUrl = '/assets/resume.pdf';
        const link = document.createElement('a');
        link.href = resumeUrl;
        link.download = 'resume.pdf';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      };
  return (
    <section className="w-full py-16 c-space">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="head-text mb-4">Technical Skills</h2>
          <p className="text-[#afb0b6] max-w-2xl mx-auto">
            A comprehensive showcase of my technical expertise and tools I work with
          </p>
        </div>
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4"
        >
          {skills.map((skill, index) => (
            <SkillCard key={skill.name} skill={skill} />
          ))}
        </motion.div>
        <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleDownload}
            className="bg-black-300 hover:bg-black-500 text-white px-6 py-3 rounded-lg flex items-center gap-2 transition-colors duration-300 mx-auto mt-8"
            >
            <FileDown className="w-5 h-5" />
            <span>Download Resume</span>
        </motion.button>
      </div>
    </section>
  );
};

export default Skills;
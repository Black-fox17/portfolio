import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const projects = [
  {
    id: 1,
    title: "Tweakr",
    role: "Backend & AI Engineer",
    description: "A document processing platform that generates citations and references using AI. Built with FastAPI and advanced NLP models.",
    detailedDescription: [
      "Developing an advanced document processing system that automatically generates citations and references using state-of-the-art NLP models.",
      "Implementing a robust backend architecture with FastAPI for high-performance document processing.",
      "Creating an AI pipeline that understands academic writing styles and citation formats.",
      "Building a scalable system that can handle multiple document formats and citation styles.",
      "Focusing on accuracy and reliability in citation generation to support academic and research work."
    ],
    tags: ["FastAPI", "NLP", "AI", "Document Processing", "Academic Writing"]
  },
  {
    id: 2,
    title: "Oncolens",
    role: "AI Engineer & Researcher",
    description: "Working on an AI pipeline to assist pathologists in identifying cancer regions in breast tissue samples.",
    detailedDescription: [
      "Developing advanced computer vision models for medical image analysis.",
      "Creating an AI pipeline that can identify and highlight potential cancer regions in breast tissue samples.",
      "Working closely with medical professionals to ensure accuracy and clinical relevance.",
      "Implementing state-of-the-art deep learning techniques for image segmentation.",
      "Focusing on creating a tool that enhances pathologists' capabilities rather than replacing them."
    ],
    tags: ["Computer Vision", "Medical AI", "Research", "Deep Learning", "Healthcare"]
  },
  {
    id: 3,
    title: "SheetFlow",
    role: "Full Stack & AI Engineer",
    description: "A collaborative spreadsheet platform with AI-powered features. Enables real-time collaboration and advanced data analysis.",
    detailedDescription: [
      "Building a next-generation spreadsheet platform that combines real-time collaboration with AI-powered features.",
      "Implementing advanced data analysis capabilities using machine learning algorithms.",
      "Developing a robust backend system for real-time updates and collaboration.",
      "Creating an intuitive UI that makes complex data operations accessible to all users.",
      "Integrating AI features for data prediction, pattern recognition, and automated insights."
    ],
    tags: ["React", "Node.js", "AI", "Real-time Collaboration", "Data Analysis"]
  },
  {
    id: 4,
    title: "News Organization Backend",
    role: "Backend Engineer",
    description: "Developing and maintaining backend systems for a major news organization, ensuring high performance and reliability.",
    detailedDescription: [
      "Architecting and implementing a high-performance backend system for a major news organization.",
      "Optimizing content delivery and data processing pipelines for maximum efficiency.",
      "Implementing robust caching mechanisms to handle high traffic loads.",
      "Developing secure and scalable APIs for content distribution.",
      "Ensuring system reliability and uptime through comprehensive monitoring and testing."
    ],
    tags: ["Backend", "Scalability", "Performance", "Content Delivery", "API Development"]
  },
];

const ProjectCarousel = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [direction, setDirection] = useState(0);

  const slideVariants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 1000 : -1000,
      opacity: 0
    }),
    center: {
      zIndex: 1,
      x: 0,
      opacity: 1
    },
    exit: (direction: number) => ({
      zIndex: 0,
      x: direction < 0 ? 1000 : -1000,
      opacity: 0
    })
  };

  const swipeConfidenceThreshold = 10000;
  const swipePower = (offset: number, velocity: number) => {
    return Math.abs(offset) * velocity;
  };

  const paginate = (newDirection: number) => {
    setDirection(newDirection);
    setCurrentIndex((prevIndex) => (prevIndex + newDirection + projects.length) % projects.length);
  };

  return (
    <>
    <p className="head-text mb-2">Ongoing Projects</p>
    <div className="w-full py-0">
      <div className="max-w-7xl mx-auto px-4">
        <div className="relative h-[600px] overflow-hidden rounded-xl bg-black/50">
          <AnimatePresence initial={false} custom={direction}>
            <motion.div
              key={currentIndex}
              custom={direction}
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{
                x: { type: "spring", stiffness: 300, damping: 30 },
                opacity: { duration: 0.2 }
              }}
              drag="x"
              dragConstraints={{ left: 0, right: 0 }}
              dragElastic={1}
              onDragEnd={(e, { offset, velocity }) => {
                const swipe = swipePower(offset.x, velocity.x);
                if (swipe < -swipeConfidenceThreshold) {
                  paginate(1);
                } else if (swipe > swipeConfidenceThreshold) {
                  paginate(-1);
                }
              }}
              className="absolute w-full h-full p-8"
            >
              <div className="relative w-full h-full">
                <div className="relative h-full flex flex-col justify-center">
                  <h3 className="text-3xl md:text-4xl font-medium mb-4 text-white font-generalsans">
                    {projects[currentIndex].title}
                  </h3>
                  <p className="text-xl text-gray-400 mb-6 font-generalsans">
                    {projects[currentIndex].role}
                  </p>
                  <div className="mb-8">
                    {projects[currentIndex].detailedDescription.map((point, index) => (
                      <p key={index} className="text-gray-300 mb-3 text-lg font-generalsans">
                        • {point}
                      </p>
                    ))}
                  </div>
                  <div className="flex flex-wrap gap-3">
                    {projects[currentIndex].tags.map((tag, index) => (
                      <span
                        key={index}
                        className="px-4 py-2 bg-white/10 rounded-full text-sm text-white font-generalsans"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>

          <button
            className="absolute left-4 top-1/2 transform -translate-y-1/2 z-10 bg-white/10 p-3 rounded-full hover:bg-white/20 transition-all"
            onClick={() => paginate(-1)}
          >
            <ChevronLeft className="w-6 h-6 text-white" />
          </button>
          <button
            className="absolute right-4 top-1/2 transform -translate-y-1/2 z-10 bg-white/10 p-3 rounded-full hover:bg-white/20 transition-all"
            onClick={() => paginate(1)}
          >
            <ChevronRight className="w-6 h-6 text-white" />
          </button>

          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
            {projects.map((_, index) => (
              <button
                key={index}
                className={`w-2 h-2 rounded-full transition-all ${
                  index === currentIndex ? 'bg-white w-4' : 'bg-white/50'
                }`}
                onClick={() => {
                  setDirection(index > currentIndex ? 1 : -1);
                  setCurrentIndex(index);
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
    </>
  );
};

export default ProjectCarousel; 
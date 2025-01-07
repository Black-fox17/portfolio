import React from 'react';
import { useState } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { myProjects } from '../constants/index.ts';

const length = myProjects.length;
const Project = () => {
    const [currentProjectindex,setCurrentProjectindex] = useState(0);

    const handleNavigation = (direction) => {
        setCurrentProjectindex((prevIndex) => {
          if (direction === 'previous') {
            return prevIndex === 0 ? length - 1 : prevIndex - 1;
          } else {
            return prevIndex === length - 1 ? 0 : prevIndex + 1;
          }
        });
      };
    
      useGSAP(() => {
        gsap.fromTo(`.animatedText`, { opacity: 0 }, { opacity: 1, duration: 1, stagger: 0.2, ease: 'power2.inOut' });
      }, [currentProjectindex]);
      
    useGSAP(() => {
        gsap.fromTo('.video-animate', { opacity: 0, x: -100 }, { opacity: 1, x: 0, duration: 1, ease: 'power2.inOut' });
    }, [currentProjectindex]);
      
    const currentProject = myProjects[currentProjectindex];
    return (
        <section className="c-space my-20" id= "work">
            <p className="head-text">My Selected Work</p>
            <div className="grid lg:grid-cols-2 grid-cols-1 mt-12 gap-5 w-full">
                <div className='relative grid'>
                    <video src = {currentProject.texture} autoPlay loop muted playsInline className='video-animate'/>

                    <div className="flex justify-between items-center mt-5">
                        <button className="arrow-btn" onClick={() => handleNavigation('previous')}>
                            <img src="/assets/left-arrow.png" alt="left arrow" />
                        </button>

                        <button className="arrow-btn" onClick={() => handleNavigation('next')}>
                            <img src="/assets/right-arrow.png" alt="right arrow" className="w-4 h-4" />
                        </button>
                    </div>
                </div>
                <div className="flex flex-col gap-5 relative sm:p-10 py-10 px-5 shadow-2xl shadow-black-200">
                <div className="absolute top-0 right-0">
                    <img src={currentProject.spotlight} alt="spotlight" className="w-full h-96 object-cover rounded-xl" />
                </div>

                <div className="p-3 backdrop-filter backdrop-blur-3xl w-fit rounded-lg" style={currentProject.logoStyle}>
                    <img className="w-10 h-10 shadow-sm" src={currentProject.logo} alt="logo" />
                </div>

                <div className="flex flex-col gap-5 text-white-600 my-5">
                    <p className="text-white text-2xl font-semibold animatedText">{currentProject.title}</p>

                    <p className="animatedText">{currentProject.desc}</p>
                    <p className="animatedText">{currentProject.subdesc}</p>
                </div>

                <div className="flex items-center justify-between flex-wrap gap-5">
                    <div className="flex items-center gap-3">
                    {currentProject.tags.map((tag, index) => (
                        <div key={index} className="tech-logo">
                        <img src={tag.path} alt={tag.name} />
                        </div>
                    ))}
                    </div>

                    <a
                    className="flex items-center gap-2 cursor-pointer text-white-600"
                    href={currentProject.href}
                    target="_blank"
                    rel="noreferrer">
                    <p>Check Live Site</p>
                    <img src="/assets/arrow-up.png" alt="arrow" className="w-3 h-3" />
                    </a>

                    <a 
                    className='flex items-center gap-2 cursor-pointer text-white-600'
                    href = {currentProject.github ? currentProject.github : "https://github.com/Black-fox17"}
                    target='_blank'>
                        <p>Github</p>
                        <img src= "/assets/github.svg" alt="github" className='w-4 h-4 '/>
                    </a>
                </div>
        
                
                
            </div>

           
           </div>
        </section>
    );
};

export default Project;
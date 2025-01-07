import React from 'react';
import { Suspense } from 'react';
import {Canvas} from '@react-three/fiber';
import { PerspectiveCamera } from '@react-three/drei';
import CanvasLoader from '../components/CanvasLoader';
import { useMediaQuery } from 'react-responsive';
import {calculateSizes} from '../constants/index.ts';
import HeroCamera from '../components/HeroCamera.jsx';
import Robot from '../components/Robot.jsx';
import Button from '../components/Button.jsx';
const Hero = () => {
    const issmall = useMediaQuery({maxWidth: 440});
    const ismobile = useMediaQuery({maxWidth: 768});
    const istablet = useMediaQuery({minwidth:768,maxWidth: 1024});

    const sizes = calculateSizes(issmall,ismobile,istablet);
    return (
        <section className="min-h-screen w-full flex flex-col relative" id="home">
            <div className="w-full mx-auto flex flex-col sm:mt-36 mt-20 c-space gap-3">
                <p className="sm:text-3xl text-xl font-medium text-white text-center font-generalsans">
                Hi, I am Oluwaseun  <span className="waving-hand">👋</span>
                </p>
                <p className="hero_tag text-gray_gradient">Building Products & Brands</p>
            </div>
            <div className="w-full h-full absolute inset-0">
                <Canvas className="w-full h-full">
                  <Suspense fallback={<CanvasLoader />}>
                    <PerspectiveCamera makeDefault position={[0, 0, 20]} />
                    <HeroCamera isMobile={issmall}>
                      <Robot scale = {sizes.deskScale} position = {sizes.deskPosition} rotation = {[0,-5.6,0]}/>
                    </HeroCamera>
                    <ambientLight intensity={1} />
                    <directionalLight position={[10, 10, 10]} intensity={0.5} />
                  </Suspense>
                </Canvas>
                <div className="absolute bottom-7 left-0 right-0 w-full z-10 c-space">
                <a href="#about" className="w-fit">
                  <Button name="Let's work together" isBeam containerClass="sm:w-fit w-full sm:min-w-96" />
                </a>
              </div>
            </div>
        </section>
    );
};  

export default Hero;
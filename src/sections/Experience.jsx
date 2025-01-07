import { useState } from "react";
import { FileText } from 'react-feather';
import {Link} from 'react-router-dom'
 
const Experience = () => {
    const [hasCopied, setHasCopied] = useState(false);

    const handleCopy = () => {
        navigator.clipboard.writeText('ayeleru1234@gmail.com');
        setHasCopied(true);

        setTimeout(() => {
            setHasCopied(false);
        }, 2000);
    };
    return (
      <section className="c-space my-20" id="research">
        <div className="w-full text-white-600">
          <p className="head-text">Research Work</p>
          <div className="work-container w-full">
            <div className="xl:col-span-1 xl:row-span-2">
              <div className="grid-container">
                <img
                  src="https://images.unsplash.com/photo-1635070041078-e363dbe005cb?auto=format&fit=crop&q=80&w=800&h=400"
                  alt="Neural Network Visualization"
                  className="w-full h-[250px] object-cover rounded-lg"
                />
                <div className="space-y-2">
                  <p className="grid-subtext text-center">Contact for Collaboration</p>
                  <div className="copy-container" onClick={handleCopy}>
                    <img src={hasCopied ? 'assets/tick.svg' : 'assets/copy.svg'} alt="copy" />
                    <p className="lg:text-2xl md:text-xl font-medium text-gray_gradient">ayeleru1234@gmail.com</p>
                  </div>
                </div>
              </div>
            </div>
  
            <div className="col-span-2 rounded-lg bg-black-200 border border-black-300 p-6">
              <div className="flex flex-col space-y-6">
                <article className="group hover:bg-black-300 transition-all duration-300 rounded-lg p-6">
                  <div className="flex items-start gap-4">
                    <div className="rounded-lg bg-black-300 p-3">
                      <FileText className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-white mb-2">
                        Comparative Analysis: Embedding Models vs CNN for Cancer Classification
                      </h3>
                      <p className="text-gray-400 mb-4">
                        A comprehensive study comparing the effectiveness of embedding-based neural networks against 
                        traditional Convolutional Neural Networks (CNNs) in identifying various classes of cancer 
                        through medical imaging.
                      </p>
                      <div className="flex flex-wrap gap-2 mb-4">
                        {['Neural Networks', 'Medical Imaging', 'Cancer Detection', 'Deep Learning'].map((tag) => (
                          <span key={tag} className="px-3 py-1 text-sm bg-black-300 rounded-full text-gray-400">
                            {tag}
                          </span>
                        ))}
                      </div>
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-gray-400">To Published: March 2025</span>
                          <span className="text-sm text-gray-400">•</span>
                          <span className="text-sm text-gray-400">15 min read</span>
                        </div>
                        <Link
                          to="/research"
                          target= '_blank'
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-black-300 text-white hover:bg-black-500 transition-colors"
                        >
                          View Paper
                          <svg 
                            className="w-4 h-4 transition-transform group-hover:translate-x-1" 
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24"
                          >
                            <path 
                              strokeLinecap="round" 
                              strokeLinejoin="round" 
                              strokeWidth={2} 
                              d="M9 5l7 7-7 7" 
                            />
                          </svg>
                        </Link>
                      </div>
                    </div>
                  </div>
                </article>
              </div>
            </div>
          </div>
        </div>
      </section>
    )
};
export default Experience;
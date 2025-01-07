import React, {useEffect} from 'react';
import { Construction, Clock, Mail } from 'lucide-react';

const Research = () => {
    useEffect(() => {
        document.title = "Research Article - Coming Soon";
      }, []);
  return (
    <div className="min-h-screen relative flex items-center justify-center">
      {/* Background Image with Overlay */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: 'url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80")',
        }}
      >
        <div className="absolute inset-0 bg-black/50"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 text-center px-4 sm:px-6 lg:px-8">
        <Construction className="w-16 h-16 text-yellow-400 mx-auto mb-8 animate-bounce" />
        <h1 className="text-4xl sm:text-6xl font-bold text-white mb-4">
          Coming Soon
        </h1>
        <p className="text-xl text-gray-200 mb-8 max-w-2xl mx-auto">
        Thank you for your interest in our research. The article is currently under review and will be available soon. We appreciate your patience and look forward to sharing our findings with you.
        </p>
        
        {/* Status Indicators */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-12">
          <div className="flex items-center gap-2 text-yellow-400">
            <Clock className="w-5 h-5" />
            <span>Launching Soon</span>
          </div>
          <div className="flex items-center gap-2 text-yellow-400">
            <Mail className="w-5 h-5" />
            <span>ayeleru1234@gmail.com</span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full max-w-md mx-auto bg-gray-700 rounded-full h-2.5">
          <div className="bg-yellow-400 h-2.5 rounded-full w-3/4 animate-pulse"></div>
        </div>
      </div>
    </div>
  );
}

export default Research;
export const navLinks = [
    {
      id: 1,
      name: 'Home',
      href: '#home',
    },
    {
      id: 2,
      name: 'About',
      href: '#about',
    },
    {
      id: 3,
      name: 'Work',
      href: '#work',
    },
    {
      id: 4,
      name: 'Contact',
      href: '#contact',
    },
    {
      id: 5,
      name: 'Blog',
      href: 'https://abdulsalam.bearblog.dev/',
      target: '_blank',
    }
  ];
  
  export const clientReviews = [
    {
      id: 1,
      name: 'Emily Johnson',
      position: 'Marketing Director at GreenLeaf',
      img: 'assets/review1.png',
      review:
        'Working with Adrian was a fantastic experience. He transformed our outdated website into a modern, user-friendly platform. His attention to detail and commitment to quality are unmatched. Highly recommend him for any web dev projects.',
    },
    {
      id: 2,
      name: 'Mark Rogers',
      position: 'Founder of TechGear Shop',
      img: 'assets/review2.png',
      review:
        'Adrian’s expertise in web development is truly impressive. He delivered a robust and scalable solution for our e-commerce site, and our online sales have significantly increased since the launch. He’s a true professional! Fantastic work.',
    },
    {
      id: 3,
      name: 'John Dohsas',
      position: 'Project Manager at UrbanTech ',
      img: 'assets/review3.png',
      review:
        'I can’t say enough good things about Adrian. He was able to take our complex project requirements and turn them into a seamless, functional website. His problem-solving abilities are outstanding.',
    },
    {
      id: 4,
      name: 'Ether Smith',
      position: 'CEO of BrightStar Enterprises',
      img: 'assets/review4.png',
      review:
        'Adrian was a pleasure to work with. He understood our requirements perfectly and delivered a website that exceeded our expectations. His skills in both frontend backend dev are top-notch.',
    },
  ];
  export const skills = [
    {
      name: "PyTorch",
      img: "assets/logo/pytorch.svg"
    },
    {
      name: "Computer Vision",
      img: "assets/logo/vision.svg"
    },
    {
      name: "Backend",
      img: "assets/logo/fastapi.svg"
    },
    {
      name: "ReactJS",
      img: "assets/react.svg"
    },
    {
      name: "Docker",
      img: "assets/logo/docker.svg"
    },
    {
      name: "Java",
      img: "assets/logo/java.svg"
    },
    {
      name: "Laravel and Database",
      img: "assets/logo/laravel.svg"
    },
    {
      name: "GitHub",
      img: "assets/github.png"
    },
    {
      name: "Data Analysis",
      img: "assets/logo/analysis.svg"
    },
    {
      name: "AI Agents",
      img: "assets/logo/langchain.svg"
    },
];

  
  export const myProjects = [
    {
      "title": "SalamAnalyzer",
      "desc": "SalamAnalyzer is an advanced web application that processes audio recordings and PDF slide documents to generate concise summaries. Additionally, users can engage in interactive conversations with the document, making it a powerful tool for understanding and analyzing content efficiently.",
      "subdesc": "Built with React.js and TypeScript on the frontend, SalamAnalyzer features an intuitive UI designed for seamless user experience. The backend, powered by FastAPI, integrates Gemini AI for accurate text extraction and summarization. Users can upload audio or PDF files and receive structured summaries along with a chat interface to further explore the document’s content.",
      "href": "https://salam-summary-two.vercel.app/",
      "texture": "/textures/project/salamanalayzer.mp4",
      "logo": "/assets/logo/brain.svg",
      "github": "https://github.com/Black-fox17/SalamSummary",
      "logoStyle": {
        "backgroundColor": "#F3F4F6",
        "border": "0.2px solid #4B5563",
        "boxShadow": "0px 0px 60px 0px #6B728E4D"
      },
      "spotlight": "/assets/spotlight2.png",
      "tags": [
        {
          "id": 1,
          "name": "React.js",
          "path": "/assets/react.svg"
        },
        {
          "id": 2,
          "name": "Tailwind",
          "path": "/assets/tailwindcss.png"
        },
        {
          "id": 3,
          "name": "FastAPI",
          "path": "/assets/logo/fastapi.svg"
        },
        {
          "id": 4,
          "name": "Gemini AI",
          "path": "/assets/logo/langchain.svg"
        }
      ]
    },
    {
      title: 'SalamSignal',
      desc: 'SalamSignal is an interactive web application designed to act as a frequency generator and tester. Users can select different waveforms such as sine, square, triangle, or sawtooth, adjust the frequency, and control the volume in real time. The application serves as a versatile tool for audio testing and signal simulation.',
      subdesc: 
        'Built entirely with React.js, SalamSignal offers a responsive and dynamic UI for waveform visualization and control. The app is simulated in C for precise testing, ensuring high accuracy in frequency generation and performance. It features a clean and intuitive interface, making it an excellent choice for hobbyists, engineers, and educators exploring signal processing concepts.',
      href: 'https://salam-signal.vercel.app/',
      texture: '/textures/project/salamsignal.mp4',
      logo: "/assets/logo/sound.svg",
      github: "https://github.com/Black-fox17/SalamSignal",
      logoStyle: {
        backgroundColor: '#F3F4F6',
        border: '0.2px solid #4B5563',
        boxShadow: '0px 0px 60px 0px #6B728E4D',
      },
      spotlight: '/assets/spotlight2.png',
      tags: [
        {
          id: 1,
          name: 'React.js',
          path: '/assets/react.svg',
        },
        {
          id: 2,
          name: 'Tailwind',
          path: '/assets/tailwindcss.png',
        },
        {
          id: 3,
          name: 'C Simulation',
          path: '/assets/logo/c.svg',
        },
      ],
    },
    {
      title: 'SalamStudy',
      desc: 'SalamStudy is an intelligent web application designed to transform educational slides into quiz questions, aiding students in efficient learning and preparation. Users can upload PDF slides, specify the discipline (course), and select the desired number of questions through a sleek, user-friendly interface.',
      subdesc: 
        'Built with React.js and TypeScript on the frontend, SalamStudy features a responsive and intuitive UI. The backend, powered by FastAPI, leverages Gemini AI to analyze slide content and generate high-quality quiz questions based on the selected course and the number of questions specified. At the end of each session, users receive detailed results and corrections for their attempted quizzes, making it a powerful tool for self-assessment and learning.',
      href: 'https://salam-study.vercel.app/',
      texture: '/textures/project/SalamStudy.mp4',
      logo: "/assets/logo/student.svg",
      github: "https://github.com/Black-fox17/SalamStudy",
      logoStyle: {
        backgroundColor: '#F3F4F6',
        border: '0.2px solid #4B5563',
        boxShadow: '0px 0px 60px 0px #6B728E4D',
      },
      spotlight: '/assets/spotlight2.png',
      tags: [
        {
          id: 1,
          name: 'React.js',
          path: '/assets/react.svg',
        },
        {
          id: 2,
          name: 'Tailwind',
          path: '/assets/tailwindcss.png',
        },
        {
          id: 3,
          name: 'FastAPI',
          path: '/assets/logo/fastapi.svg',
        },
        {
          id: 4,
          name: 'Gemini AI',
          path: '/assets/logo/langchain.svg',
        },
        {
          id: 5,
          name: 'TypeScript',
          path: '/assets/typescript.png',
        },
      ],
    },    
    {
      title: 'SalamGym',
      desc: 'SalamGym is an innovative web platform designed to provide real-time feedback on workout postures. Users can upload videos of their exercise routines, which are analyzed using advanced pose detection and AI technologies to improve their fitness and reduce the risk of injuries.',
      subdesc: 
        'Built with React, TypeScript, and Vite on the frontend, and powered by FastAPI on the backend, SalamGym processes user-uploaded workout videos with MediaPipe pose detection. The backend leverages an advanced LLM model to analyze user posture, delivering tailored text feedback highlighting key areas for improvement. The platform also includes a comprehensive user management system, based on a PostgreSQL database, which tracks session accuracy streaks and provides personalized insights for each user.',
      href: 'https://salam-gym.vercel.app',
      texture: '/textures/project/salamgym.mp4',
      logo: "/assets/logo/gym.svg",
      github: "https://github.com/Black-fox17/AiCoach.git",
      logoStyle: {
        backgroundColor: '#F3F4F6',
        border: '0.2px solid #1D3557',
        boxShadow: '0px 0px 60px 0px #457B9D4D',
      },
      spotlight: '/assets/spotlight2.png',
      tags: [
        {
          id: 1,
          name: 'React.js',
          path: '/assets/react.svg',
        },
        {
          id: 2,
          name: 'Tailwind',
          path: '/assets/tailwindcss.png',
        },
        {
          id: 4,
          name: 'FastAPI',
          path: '/assets/logo/fastapi.svg',
        },
        {
          id: 5,
          name: 'MediaPipe',
          path: '/assets/logo/vision.svg',
        },
        {
          id: 6,
          name: 'PostgreSQL',
          path: '/assets/logo/postgresql.svg',
        },
      ],
    },    
    {
      title: 'SalamStocks - Stock Analysis Platform',
      desc: 'SalamStock is a comprehensive stock analysis and prediction platform designed to empower users with informed investment decisions. It provides an intuitive interface for analyzing historical stock performance and predicting future price trends for various stocks.',
      subdesc: 
        'Built with FastAPI, React, and WebSocket technologies, SalamStock fetches real-time stock data from a reliable API, displays historical trends, and overlays AI-driven future price predictions on a visually compelling graph. The platform also offers personalized investment advice, generated by an advanced AI model, to guide users toward strategic financial decisions.'
      ,href: 'https://stockanalysis-frontend.vercel.app/',
      texture: '/textures/project/stockanalysis.mp4',
      logo: "/assets/logo/salamstock.svg",
      github: "https://github.com/Black-fox17/Stockanalysis.git",
      logoStyle: {
        backgroundColor: '#0E1F38',
        border: '0.2px solid #0E2D58',
        boxShadow: '0px 0px 60px 0px #2F67B64D',
      },
      spotlight: '/assets/spotlight4.png',
      tags: [
        {
          id: 1,
          name: 'React.js',
          path: '/assets/react.svg',
        },
        {
          id: 2,
          name: 'TailwindCSS',
          path: '/assets/tailwindcss.png',
        },
        {
          id: 3,
          name: 'TypeScript',
          path: '/assets/typescript.png',
        },
        {
          id: 4,
          name: 'Fastapi',
          path: '/assets/logo/fastapi.svg',
        },
      ],
    },
    {
      "title": "SalamPick",
      "desc": "Salampick is a modern e-commerce platform that provides a seamless shopping experience for users and comprehensive admin tools for managing products, orders, and payments.",
      "subdesc": "Built with Laravel, PostgreSQL, and Tailwind Blade, ShopSphere offers powerful features including user authentication, product management, a dynamic cart, and Google payment integration, all wrapped in a beautifully styled interface.",
      "href": "https://salam-pick.onrender.com",
      "texture": "/textures/project/salampick.mp4",
      "logo": "/assets/logo/loropiana.svg",
      "github": "https://github.com/Black-fox17/Salampick",
      "logoStyle": {
        "backgroundColor": "#1A202C",
        "border": "0.2px solid #2D3748",
        "boxShadow": "0px 0px 50px 0px #4A55684D"
      },
      "spotlight": "/assets/spotlight4.png",
      "tags": [
        {
          "id": 1,
          "name": "Laravel",
          "path": "/assets/logo/laravel.svg"
        },
        {
          "id": 2,
          "name": "PostgreSQL",
          "path": "/assets/logo/postgresql.svg"
        },
        {
          "id": 3,
          "name": "TailwindCSS",
          "path": "/assets/tailwindcss.png"
        },
        {
          "id": 4,
          "name": "Javascript",
          "path": "/assets/react.svg"
        },
        {
          "id": 5,
          "name": "Google Payments",
          "path": "/assets/logo/googlepay.svg"
        }
      ]
    }    
  ];
  
  export const calculateSizes = (isSmall, isMobile, isTablet) => {
    return {
      deskScale: isSmall ? 3 : isMobile ? 4 : 4,
      deskPosition: isMobile ? [0.5, -4.5, 0] : [0.25, -7, 0],
      cubePosition: isSmall ? [4, -5, 0] : isMobile ? [5, -5, 0] : isTablet ? [5, -5, 0] : [9, -5.5, 0],
      reactLogoPosition: isSmall ? [3, 4, 0] : isMobile ? [5, 4, 0] : isTablet ? [5, 4, 0] : [12, 3, 0],
      ringPosition: isSmall ? [-5, 7, 0] : isMobile ? [-10, 10, 0] : isTablet ? [-12, 10, 0] : [-24, 10, 0],
      targetPosition: isSmall ? [-5, -10, -10] : isMobile ? [-9, -10, -10] : isTablet ? [-11, -7, -10] : [-13, -13, -10],
    };
  };
  
  export const workExperiences = [
    {
      id: 1,
      name: 'Freelance',
      pos: 'Software Engineer & AI Developer',
      duration: '2022 - Present',
      title: "As a self-employed professional, I have been working on freelance projects as a Software Engineer and AI Developer. I specialize in integrating AI agents to solve real-world problems and building complex AI applications.",
      icon: '/assets/logo/upwork.svg',
      animation: 'idea',
    },
    {
      id: 2,
      name: 'HNG Internship',
      pos: 'Tech Role (Details Pending)',
      duration: 'Starting Soon',
      title: "Excited to embark on an internship with HNG, where I'll be taking on technical roles yet to be discussed. This opportunity will allow me to contribute to impactful projects and further hone my skills.",
      icon: '/assets/logo/hng.svg',
      animation: 'rocket',
    },
  ];
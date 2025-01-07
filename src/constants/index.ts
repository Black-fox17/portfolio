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
      title: 'Horizon - Online Banking Platform',
      desc: 'Horizon is a comprehensive online banking platform that offers users a centralized finance management dashboard. It allows users to connect multiple bank accounts, monitor real-time transactions, and seamlessly transfer money to other users.',
      subdesc:
        'Built with Next.js 14 Appwrite, Dwolla and Plaid, Horizon ensures a smooth and secure banking experience, tailored to meet the needs of modern consumers.',
      href: 'https://www.youtube.com/watch?v=PuOVqP_cjkE',
      texture: '/textures/project/project4.mp4',
      logo: '/assets/project-logo4.png',
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
          path: 'assets/tailwindcss.png',
        },
        {
          id: 3,
          name: 'TypeScript',
          path: '/assets/typescript.png',
        },
        {
          id: 4,
          name: 'Framer Motion',
          path: '/assets/framer.png',
        },
      ],
    },
    {
      "title": "SalamPick",
      "desc": "Salampick is a modern e-commerce platform that provides a seamless shopping experience for users and comprehensive admin tools for managing products, orders, and payments.",
      "subdesc": "Built with Laravel, PostgreSQL, and Tailwind Blade, ShopSphere offers powerful features including user authentication, product management, a dynamic cart, and Google payment integration, all wrapped in a beautifully styled interface.",
      "href": "https://www.youtube.com/watch?v=example_demo",
      "texture": "/textures/project/project3.mp4",
      "logo": "/assets/logo/loropiana.svg",
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
      deskScale: isSmall ? 0.001 : isMobile ? 0.003 : 0.005,
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
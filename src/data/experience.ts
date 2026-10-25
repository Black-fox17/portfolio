import { ExperienceItem } from '../types/portfolio';

export const experiences: ExperienceItem[] = [
  {
    id: 'goodtenants-upward',
    company: 'GoodTenants / Upward Ecosystem',
    role: 'Software Engineer & Product Engineer',
    location: 'Lagos, Nigeria',
    type: 'Hybrid',
    period: 'March 2026 - Present',
    current: true,
    domain: 'PropTech / FinTech & Mobile Systems (iOS & Android)',
    summary:
      'Leading engineering and mobile app releases for the Upward Ecosystem across iOS, Android, and web platforms.',
    highlights: [
      'Took complete ownership of launching UpwardPay on the Apple App Store, handling Apple guidelines, TestFlight review, and the production release.',
      'Shipped and maintained two Android apps, Upward Pay and Upward PM, on the Google Play Store, managing releases, testing tracks, and app health.',
      'Built and maintained 500+ production REST APIs for rent collection, digital lease agreements, tenant verification, and property operations using NestJS, TypeScript, and PostgreSQL.',
      'Built a central NestJS server to process simultaneous requests from our property manager app and tenant app, while syncing with external property management platforms using a structured PostgreSQL database.',
      'Handled full-stack development, UI design, cloud deployment, and cross-platform optimization across iOS, Android, and web using Next.js, Capacitor, and GitHub Actions.'
    ],
    technologies: [
      'iOS (App Store)',
      'Android (Google Play)',
      'NestJS',
      'TypeScript',
      'PostgreSQL',
      'Next.js',
      'Capacitor',
      'Paystack',
      'Flutterwave',
      'GitHub Actions',
      'Docker'
    ]
  },
  {
    id: 'codygo',
    company: 'Codygo',
    role: 'AI Full-Stack Engineer',
    location: 'Remote',
    type: 'Remote',
    period: 'May 2025 - December 2025',
    current: false,
    domain: 'Enterprise AI & Distributed Systems',
    summary:
      'Built enterprise AI search assistants and background data pipelines for fast business workflows.',
    highlights: [
      'Developed conversational AI assistants with FastAPI, TypeScript, and MongoDB, enabling fast search and clear answers across internal company documents.',
      'Created data pipelines to automatically import and update content from websites, PDFs, Notion, Confluence, and Jira for AI search.',
      'Built background processing systems with AWS Lambda, Amazon SQS, and DynamoDB to handle heavy AI jobs smoothly without slowing down user requests.'
    ],
    technologies: [
      'FastAPI',
      'Python',
      'TypeScript',
      'AWS Lambda',
      'Amazon SQS',
      'DynamoDB',
      'MongoDB',
      'RAG / Vector DBs',
      'Docker',
      'GitHub Actions'
    ]
  },
  {
    id: 'tweakrr-exp',
    company: 'Tweakrr',
    role: 'AI Engineer & Backend Engineer',
    location: 'Remote',
    type: 'Remote',
    period: 'March 2025 - May 2025',
    current: false,
    domain: 'NLP & Document Intelligence',
    summary:
      'Built document analysis pipelines and automatic citation tools for academic papers.',
    highlights: [
      'Created text analysis pipelines to parse research papers, flag unsupported claims, and suggest relevant academic citations automatically.',
      'Built backend services with Python, FastAPI, and MongoDB for file uploads, user accounts, paper management, and formatted bibliography generation.'
    ],
    technologies: [
      'Python',
      'FastAPI',
      'NLP / Transformers',
      'MongoDB',
      'Vector Search',
      'Async Processing'
    ]
  },
  {
    id: 'iamrat',
    company: 'Institute of Advanced Medical Research and Training (IAMRAT)',
    role: 'AI Research & Systems Engineer',
    location: 'University College Hospital, Ibadan',
    type: 'On-site',
    period: 'December 2024 - March 2025',
    current: false,
    domain: 'Medical AI & Computer Vision Research',
    summary:
      'Trained computer vision models to classify breast cancer subtypes from biopsy slides and built clinical support tools to explain results to doctors.',
    highlights: [
      'Trained AI models using Google Path Foundation and deep neural networks to accurately identify breast cancer subtypes from high-resolution biopsy images.',
      'Combined image classification models with Gemini 1.5 to generate clear diagnostic summaries and explanations for pathologists.',
      'Built full-stack web applications with FastAPI and React for researchers to upload slides, label samples, and track clinical workflows.',
      'Open-sourced the complete clinical pathology AI codebase on GitHub.'
    ],
    technologies: [
      'Path-Foundation',
      'PyTorch',
      'TensorFlow',
      'Gemini 1.5',
      'FastAPI',
      'Docker',
      'React',
      'Medical Image Analysis'
    ]
  }
];

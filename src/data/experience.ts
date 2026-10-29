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
    company: 'Codygo (Worknet)',
    role: 'AI Full-Stack & Serverless Engineer',
    location: 'Remote',
    type: 'Remote',
    period: 'May 2025 - December 2025',
    current: false,
    domain: 'Enterprise B2B AI & Distributed Vector Pipelines',
    summary:
      'Owned the core serverless RAG synchronization engine and B2B embeddable AI copilot widget, enabling enterprise clients to connect knowledge sources and embed contextual AI on their web apps.',
    highlights: [
      'Took full engineering ownership of sync-datasource, an event-driven RAG pipeline built with AWS SAM, Lambda, SQS, EventBridge, and DynamoDB, using cursor-based pagination to sync massive data sources past Lambda timeout limits.',
      'Implemented document chunking, SHA-256 content deduplication, and automated Pinecone vector indexing to minimize embedding compute costs and eliminate redundant updates.',
      'Built and maintained serverless cloud infrastructure with AWS SAM (sam-app), packaging shared domain models and AWS SDK clients into high-performance Lambda Layers.',
      'Developed the Worknet B2B embeddable widget (embed.js & secure cross-origin iframe runtime) served dynamically via AWS Lambda Function URLs for one-line script integration on customer websites.',
      'Built self-serve data source configuration screens and batch upload processors in the admin dashboard, allowing organizations to configure connectors and manage sync schedules.'
    ],
    technologies: [
      'AWS SAM',
      'AWS Lambda',
      'Amazon SQS',
      'Amazon EventBridge',
      'DynamoDB',
      'Pinecone (Vector DB)',
      'TypeScript',
      'MongoDB Atlas',
      'RAG Architecture',
      'Embeddable Widgets / iFrames',
      'Docker'
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

import { ExperienceItem } from '../types/portfolio';

export const experiences: ExperienceItem[] = [
  {
    id: 'goodtenants-upward',
    company: 'GoodTenants / Upward Ecosystem',
    role: 'Software Engineer & Product Engineer',
    location: 'Lagos, Nigeria',
    type: 'Hybrid',
    period: 'March 2026 – Present',
    current: true,
    domain: 'PropTech / FinTech & Mobile Systems (iOS & Android)',
    summary:
      'Leading engineering, deployment, and operational architecture for the Upward Ecosystem across iOS, Android, and web platforms.',
    highlights: [
      'Took complete ownership of launching UpwardPay on the Apple App Store, managing the full submission lifecycle, App Store guidelines review, TestFlight builds, and production release.',
      'Shipped and operated two production Android applications, Upward Pay and Upward PM, on the Google Play Store; supervised release tracks, Play Console rollouts, and crash monitoring.',
      'Architected and maintained 500+ production REST APIs across core domain planes for rent collection, automated lease agreements, document workflows, tenant verification, and property operations using NestJS, TypeScript, and PostgreSQL.',
      'Architected a monolithic NestJS server handling concurrent internal requests from the PM app and Tenant app, while orchestrating real-time integrations with external property management platforms—backed by a complex, high-integrity PostgreSQL database schema.',
      'Owned architecture, UI/UX, frontend, backend, cloud infrastructure, and deployment strategy with Next.js, Capacitor, and GitHub Actions; directed interface systems and cross-platform optimization across iOS, Android, and web.'
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
    period: 'May 2025 – December 2025',
    current: false,
    domain: 'Enterprise AI / RAG & Distributed Systems',
    summary:
      'Built enterprise-grade LLM/RAG systems and distributed asynchronous data ingestion pipelines for low-latency business automation.',
    highlights: [
      'Built enterprise LLM/RAG assistants for business automation with FastAPI, TypeScript, and MongoDB, combining semantic search, contextual memory, and low-latency organizational knowledge access.',
      'Architected configurable ingestion pipelines for websites, PDFs, knowledge bases, Confluence, Jira, and custom business connectors, including extraction, normalization, hashing, and incremental synchronization for semantic retrieval.',
      'Orchestrated asynchronous AI processing, background worker jobs, and distributed execution with AWS Lambda, Amazon SQS, and DynamoDB; containerized services with Docker and contributed to GitHub Actions CI/CD workflows.'
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
    period: 'March 2025 – May 2025',
    current: false,
    domain: 'NLP & Document Intelligence',
    summary:
      'Architected scholarly document analysis pipelines and automated citation generation services.',
    highlights: [
      'Architected NLP/RAG document pipelines for parsing, semantic context extraction, unsupported-claim detection, scholarly retrieval, and contextual citation recommendations aligned with academic citation requirements.',
      'Built Python, FastAPI, and MongoDB services for uploads, authentication, asynchronous processing, document management, formatted in-text citations, and reference-list generation.'
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
    period: 'December 2024 – March 2025',
    current: false,
    domain: 'Medical AI & Computer Vision Research',
    summary:
      'Engineered pathology foundation model pipelines and deep convolutional networks for breast cancer subtype classification and built multimodal clinical decision-support tooling.',
    highlights: [
      'Developed histopathological breast cancer subtype classification pipelines leveraging Google Path Foundation Vision Transformers and convolutional architectures, establishing strong accuracy across whole slide image biopsy datasets.',
      'Constructed multimodal clinical decision-support workflows combining computer vision diagnostic predictions with Gemini 1.5 diagnostic explanation generators.',
      'Engineered full-stack FastAPI and React research portals for pathologist data collection, sample annotation, and trial workflow tracking.',
      'Open-sourced the end-to-end clinical pathology AI codebase at github.com/Black-fox17/BreastCancer.'
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

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
    domain: 'PropTech / FinTech & Mobile Systems',
    summary:
      'Leading engineering, deployment, and operational architecture for the Upward Ecosystem across web and mobile platforms.',
    highlights: [
      'Shipped and operated two production Android applications, Upward Pay and Upward PM, on the Google Play Store; supervised release preparation, Play Console operations, rollout strategies, crash monitoring, and maintenance.',
      'Architected a secure, resilient PropTech ecosystem linking tenants, landlords, property managers, payment providers, insurance services, and external property management platforms.',
      'Architected and maintained 200+ production REST endpoints across 6 domain planes for rent collection, automated lease agreements, document workflows, tenant verification, and property operations using NestJS, TypeScript, and PostgreSQL.',
      'Built payment orchestration for rent, verification, property transactions, and financial workflows: Paystack primary processing with automatic Flutterwave failover for continuity during gateway degradation or outages.',
      'Owned architecture, UI/UX, frontend, backend, cloud infrastructure, and deployment strategy with Next.js, Capacitor, and GitHub Actions; directed branding, interface systems, SEO, and technical optimization across web and mobile.'
    ],
    technologies: [
      'NestJS',
      'TypeScript',
      'PostgreSQL',
      'Next.js',
      'Capacitor',
      'Android',
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
      'Engineered deep convolutional neural networks for histopathological cancer image classification and built clinical decision-support tooling.',
    highlights: [
      'Developed deep convolutional architectures (ResNet/MobileNet) for automated histopathological breast cancer subtype classification, establishing baseline accuracy across whole slide image datasets.',
      'Constructed multimodal decision-support workflows combining computer vision diagnostic predictions with structured clinical explanation generators.',
      'Engineered full-stack FastAPI and React research portals for pathologist data collection, sample annotation, and trial workflow tracking.'
    ],
    technologies: [
      'PyTorch',
      'Computer Vision',
      'ResNet',
      'MobileNet',
      'FastAPI',
      'React',
      'Medical Image Analysis'
    ]
  }
];

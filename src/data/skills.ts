import { SkillCategory } from '../types/portfolio';

export const skillCategories: SkillCategory[] = [
  {
    title: 'AI & Advanced Computer Vision',
    description: 'Neural networks, computer vision models, medical imaging, and AI agent frameworks.',
    skills: [
      { name: 'Semantic Segmentation & Weak Supervision', focus: 'PyTorch U-Net models, point labeling, partial cross-entropy, and full-image scanning' },
      { name: 'Deep CNNs & Modern Backbones', focus: 'Image classification, feature extraction, and modern neural network backbones' },
      { name: 'Medical Vision & Volumetric Imaging', focus: 'Biopsy image analysis, 3D MRI scans, DICOM processing, and clinical workflows' },
      { name: 'Edge Kinematics & Gesture Vision', focus: 'Hand and body tracking with MediaPipe, real-time mobile inference with TFLite' },
      { name: 'Object Detection & Spatial Intelligence', focus: 'Real-time object detection with YOLO and OpenCV for store inventory audits' },
      { name: 'Agentic Systems & Recursive Context (RLM)', focus: 'Coordinating multi-agent teams, tool calling, and long-document reasoning' },
      { name: 'Transformers & NLP From Scratch', focus: 'Building language models, tokenizers, word embeddings, and attention from scratch' }
    ]
  },
  {
    title: 'Backend & Data Engineering',
    description: 'Reliable API design, structured databases, background worker queues, and payment systems.',
    skills: [
      { name: 'TypeScript & JavaScript', focus: 'Type-safe services and modern asynchronous code' },
      { name: 'Python', focus: 'FastAPI backends, asynchronous processing, and data workflows' },
      { name: 'NestJS', focus: 'Clean modular architecture, dependency injection, and guards' },
      { name: 'PostgreSQL', focus: 'Reliable database transactions, table indexing, and query optimization' },
      { name: 'Monolithic Backends & Data Integrity', focus: 'Building large-scale NestJS backends with 500+ APIs and clean PostgreSQL databases' },
      { name: 'MongoDB', focus: 'Document databases, flexible data models, and aggregation queries' },
      { name: 'REST & Webhook APIs', focus: 'Idempotency, payment webhooks, and OpenAPI documentation' }
    ]
  },
  {
    title: 'Cloud, DevOps & Systems',
    description: 'Cloud deployment, background task processing, containers, and automated delivery pipelines.',
    skills: [
      { name: 'AWS Lambda & SQS', focus: 'Background workers, asynchronous job handling, and event triggers' },
      { name: 'DynamoDB', focus: 'Fast key-value storage and scalable database design' },
      { name: 'Docker', focus: 'Containerizing services and creating reproducible environments' },
      { name: 'GitHub Actions', focus: 'Automated testing, continuous integration, and cloud deployments' },
      { name: 'Release Engineering', focus: 'Version tracking, clean release notes, and deployment rollouts' },
      { name: 'Distributed Systems', focus: 'Message queues, safe retries, and network failure handling' }
    ]
  },
  {
    title: 'Product Engineering & Mobile',
    description: 'Full-stack web apps, native mobile store releases, and clean user interfaces.',
    skills: [
      { name: 'Next.js & React', focus: 'Fast server-rendered pages, modern UI state, and responsive styling' },
      { name: 'Capacitor', focus: 'Packaging web applications into native iOS and Android apps' },
      { name: 'Apple App Store & Play Console', focus: 'App Store review guidelines, TestFlight beta builds, and Google Play tracks' },
      { name: 'SaaS Architecture', focus: 'User authentication, role-based access, and payment integrations' },
      { name: 'UI / UX Design Systems', focus: 'Clean typography, responsive layouts, accessibility, and Tailwind CSS' }
    ]
  }
];

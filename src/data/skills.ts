import { SkillCategory } from '../types/portfolio';

export const skillCategories: SkillCategory[] = [
  {
    title: 'AI & Advanced Computer Vision',
    description: 'First-principles model design, semantic segmentation, deep convolutional backbones, medical vision, and agentic systems.',
    skills: [
      { name: 'Semantic Segmentation & Weak Supervision', focus: 'PyTorch U-Net, Point Supervision, Partial Cross-Entropy (pCE), sliding tiled inference' },
      { name: 'Deep CNNs & Modern Backbones', focus: 'ResNet, MobileNet, EfficientNetV2, ConvNeXt, Generalized Mean (GeM) pooling' },
      { name: 'Medical Vision & Volumetric Imaging', focus: 'Google Path Foundation ViT, Whole-Slide Biopsies, DICOM 3D projection & VOI LUT windowing' },
      { name: 'Edge Kinematics & Gesture Vision', focus: 'MediaPipe 543 holistic landmarks, 1D conv temporal downsampling, sub-30ms TFLite' },
      { name: 'Object Detection & Spatial Intelligence', focus: 'Custom YOLO, OpenCV spatial heuristics, bounding box planogram auditing' },
      { name: 'Agentic Systems & Recursive Context (RLM)', focus: 'Supervisor agents, dynamic routing schemas, isolated REPLs (MIT CSAIL RLM)' },
      { name: 'Transformers & NLP From Scratch', focus: 'Self-attention, BPE tokenization, Skip-Gram, positional encodings, KV caching' }
    ]
  },
  {
    title: 'Backend & Data Engineering',
    description: 'Resilient API architectures, relational ledgers, asynchronous queues, and database schemas.',
    skills: [
      { name: 'TypeScript & JavaScript', focus: 'Type-safe services, modern async paradigms' },
      { name: 'Python', focus: 'FastAPI, asynchronous event loops, data processing' },
      { name: 'NestJS', focus: 'Modular architecture, dependency injection, guards/interceptors' },
      { name: 'PostgreSQL', focus: 'ACID transactions, relational indexing, query optimization' },
      { name: 'Financial Ledgers & ACID Safety', focus: 'Double-entry accounting, distributed Redis idempotency locks, transaction boundaries' },
      { name: 'MongoDB', focus: 'Document stores, aggregations, schema validation' },
      { name: 'REST & Webhook APIs', focus: 'Idempotency, contract testing, OpenAPI schemas' }
    ]
  },
  {
    title: 'Cloud, DevOps & Systems',
    description: 'Event-driven serverless architectures, container orchestration, and automated CI/CD.',
    skills: [
      { name: 'AWS Lambda & SQS', focus: 'Asynchronous workers, distributed processing, event triggers' },
      { name: 'DynamoDB', focus: 'Single-table design, key-value state indexing' },
      { name: 'Docker', focus: 'Multi-stage builds, containerization, reproducible envs' },
      { name: 'GitHub Actions', focus: 'CI/CD pipelines, automated testing, deployment controls' },
      { name: 'Release Engineering', focus: 'Branch strategies, semantic versioning, rollback protocols' },
      { name: 'Distributed Systems', focus: 'Message queues, idempotency, retry mechanisms' }
    ]
  },
  {
    title: 'Product Engineering & Mobile',
    description: 'End-to-end full-stack execution, mobile app stores, cross-platform apps, and accessible UX.',
    skills: [
      { name: 'Next.js & React', focus: 'App Router, SSR, performant client state, responsive design' },
      { name: 'Capacitor', focus: 'Native web-to-mobile bridge, iOS & Android runtime integration' },
      { name: 'Apple App Store & Play Console', focus: 'iOS App Store submission, TestFlight, Play Store tracks, release lifecycle' },
      { name: 'SaaS Architecture', focus: 'Multi-tenancy, billing, RBAC, analytics integration' },
      { name: 'UI / UX Design Systems', focus: 'Minimalist editorial UI, accessibility (WCAG), Tailwind' }
    ]
  }
];

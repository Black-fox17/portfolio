import { SkillCategory } from '../types/portfolio';

export const skillCategories: SkillCategory[] = [
  {
    title: 'AI & Machine Learning',
    description: 'First-principles model design, generative AI pipelines, retrieval architectures, and computer vision.',
    skills: [
      { name: 'LLMs & Fine-Tuning', focus: 'Prompt engineering, eval harnesses, structured JSON' },
      { name: 'RAG & Vector Retrieval', focus: 'Hybrid search, reranking, chunking strategies' },
      { name: 'NLP & Transformers', focus: 'Self-attention, tokenization, positional encodings' },
      { name: 'PyTorch', focus: 'Model training, custom layers, embeddings, inference' },
      { name: 'AI Agents & Tool Use', focus: 'Multi-agent orchestration, state machines, memory' },
      { name: 'Computer Vision', focus: 'YOLO, object detection, ResNet/MobileNet, OpenCV' },
      { name: 'Inference Optimization', focus: 'KV caching, quantization, sliding window attention' }
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
      { name: 'MongoDB', focus: 'Document stores, aggregations, schema validation' },
      { name: 'REST & Webhook APIs', focus: 'Idempotency, contract testing, OpenAPI schemas' },
      { name: 'Payment Orchestration', focus: 'Paystack, Flutterwave, gateway failover, ledger safety' }
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
      { name: 'Capacitor', focus: 'Native web-to-mobile bridge, Android runtime integration' },
      { name: 'Google Play Console', focus: 'App store releases, test tracks, rollout management' },
      { name: 'SaaS Architecture', focus: 'Multi-tenancy, billing, RBAC, analytics integration' },
      { name: 'UI / UX Design Systems', focus: 'Minimalist editorial UI, accessibility (WCAG), Tailwind' }
    ]
  }
];

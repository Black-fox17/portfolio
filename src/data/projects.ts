import { Project } from '../types/portfolio';

export const projects: Project[] = [
  {
    id: 'upward-ecosystem',
    slug: 'upward-ecosystem',
    title: 'Upward Ecosystem',
    subtitle: 'Production PropTech & FinTech Multi-Platform Operating System',
    category: 'PropTech / FinTech',
    role: 'Software Engineer & Product Engineer',
    status: 'In Production',
    featured: true,
    priority: 1,
    period: 'March 2026 – Present',
    summary:
      'Architected and shipped a unified PropTech & FinTech ecosystem linking tenants, landlords, property managers, insurance providers, and payment processors. Operates 3 live production applications across Apple App Store (UpwardPay iOS), Google Play Store (Upward Pay & Upward PM Android), and web, backed by a high-throughput NestJS backend with 200+ production REST endpoints.',
    metrics: [
      '3 Production Apps Shipped (Apple App Store & Google Play)',
      '200+ Production Endpoints & Multi-Domain Ledger Services',
      'Paystack + Flutterwave Automated Multi-Gateway Failover',
      'Unified iOS, Android, Web & Cloud Delivery'
    ],
    tags: [
      'iOS (App Store)',
      'Android (Play Store)',
      'NestJS',
      'TypeScript',
      'PostgreSQL',
      'Next.js',
      'Capacitor',
      'Paystack',
      'Flutterwave',
      'AWS',
      'Docker',
      'GitHub Actions'
    ],
    links: [
      {
        label: 'UpwardPay (iOS App Store)',
        url: 'https://apps.apple.com/us/app/upwardpay/id6807898152',
        type: 'appstore'
      },
      {
        label: 'Upward Pay (Play Store)',
        url: 'https://play.google.com/store/apps/details?id=com.goodtenants.upward',
        type: 'playstore'
      },
      {
        label: 'Upward PM (Play Store)',
        url: 'https://play.google.com/store/apps/details?id=com.goodtenants.upward.pm',
        type: 'playstore'
      },
      {
        label: 'Upward Web Portal',
        url: 'https://upward.goodtenants.io',
        type: 'live'
      }
    ],
    caseStudy: {
      overview:
        'The Upward Ecosystem addresses critical friction in the rental housing and property operations sector. It replaces fragmented manual rent collection, unverified tenancy records, and delayed maintenance with an interconnected ecosystem of production mobile applications (UpwardPay on Apple App Store, Upward Pay & Upward PM on Google Play Store) and a high-availability backend cluster.',
      problem:
        'Rental transactions in emerging markets suffer from high payment failure rates, lack of verifiable credit profiling for tenants, and disconnected communication between landlords and property managers. A single payment gateway outage could halt cash flows and dispute resolution.',
      architecture: [
        'Modular NestJS micro-monolith running with TypeScript, organized into domain modules for Identity & KYC, Lease Lifecycle, Rent Collection, Financial Ledger, and Maintenance Tickets.',
        'PostgreSQL with strict relational constraints, indexed foreign keys for sub-10ms query execution, and ACID transaction boundaries for all ledger postings.',
        'Cross-platform delivery using Next.js for high-performance responsive web and Capacitor for native iOS and Android compilation, maintaining 90%+ code sharing across platforms.',
        'Payment Orchestration Engine with intelligent routing: Paystack handles primary direct debit/virtual accounts with instant webhook reconciliation, and automated fallback to Flutterwave when latency exceeds thresholds or error rates rise.'
      ],
      engineeringHighlights: [
        'Took complete ownership of taking UpwardPay live to the Apple App Store, orchestrating the entire submission process, App Store guidelines compliance, TestFlight review cycles, and release management.',
        'Shipped and maintained both Upward Pay and Upward PM on the Google Play Store, managing release tracks, Play Console testing cycles, crash analytics, and seamless over-the-air asset updates.',
        'Engineered 200+ secure REST endpoints across 6 domain planes (Tenants, Landlords, Property Managers, Banking & Webhooks, Public, Admin) with JWT authorization, granular RBAC, DTO validation, and rate limiting.',
        'Engineered payment webhook idempotency using Redis-backed locking and transaction state verification to eliminate double-crediting during network retries.',
        'Architected end-to-end CI/CD workflows using GitHub Actions to automate linting, unit/e2e testing, Docker container builds, and deployment verification.'
      ],
      complexityAndFailures:
        'Payment gateway degradation was a key point of failure. I implemented an active circuit breaker with exponential backoff and automated failover between Paystack and Flutterwave, guaranteeing zero transaction drop during high-traffic billing cycles.',
      productExperience:
        'Tenants effortlessly manage lease visibility, automate scheduled rent payments, build verified tenancy credit scores, and request instant repairs. Property managers access real-time occupancy heatmaps, automated arrears alerts, lease document signing, and instant financial reporting.',
      shippedArtifacts: [
        'UpwardPay iOS Application (Apple App Store)',
        'Upward Pay Android Application (Google Play Store)',
        'Upward PM Android Application (Google Play Store)',
        'Upward Web Administration & Landlord Portal (upward.goodtenants.io)',
        'NestJS Core API Services & Webhook Receiver Cluster',
        'Automated GitHub Actions Deployment Pipeline'
      ],
      outcomes: [
        'Zero payment drops via automated Paystack + Flutterwave circuit-breaker failover.',
        'Reduced manual property onboarding and tenant verification turnaround from days to minutes.',
        'Zero downtime multi-platform releases across iOS, Android, and web.'
      ]
    },
    architectureDiagram: {
      layers: [
        {
          name: 'Clients (iOS, Android & Web)',
          items: ['UpwardPay (Apple App Store)', 'Upward Pay (Google Play)', 'Upward PM (Google Play)', 'Capacitor Native Bridge', 'Next.js App Router']
        },
        {
          name: 'API Gateway & Services',
          items: ['NestJS REST API (200+ Endpoints)', 'RBAC & Auth Guard', 'DTO Validation & Serialization', 'Rate Limiting']
        },
        {
          name: 'Payment & Third-Party Integration',
          items: ['Payment Orchestrator', 'Paystack (Primary)', 'Flutterwave (Failover)', 'KYC & Identity Verification']
        },
        {
          name: 'Data & Infrastructure',
          items: ['PostgreSQL (ACID Ledger)', 'Redis (Idempotency Cache)', 'AWS Cloud', 'GitHub Actions CI/CD']
        }
      ]
    }
  },
  {
    id: 'tweakrr',
    slug: 'tweakrr',
    title: 'Tweakrr',
    subtitle: 'AI Citation, Unsupported-Claim Detection & Scholarly Intelligence Platform',
    category: 'AI & Document Intelligence',
    role: 'AI Engineer & Backend Engineer',
    status: 'Live Platform',
    featured: true,
    priority: 2,
    period: 'March 2025 – May 2025',
    summary:
      'Engineered an intelligent document analysis and citation recommendation platform. Designed asynchronous NLP pipelines that parse complex manuscripts, detect unsupported factual claims, and retrieve contextual scholarly literature.',
    metrics: [
      'Asynchronous Document Parsing & Ingestion Engine',
      'Automated In-Text Citation & Reference Generation (APA, IEEE, Harvard)',
      'Sub-second Semantic Vector Search across Academic Corpora',
      'FastAPI + MongoDB Scalable Microservice Architecture'
    ],
    tags: [
      'Python',
      'FastAPI',
      'NLP / LLMs',
      'MongoDB',
      'Vector Embeddings',
      'Semantic Search',
      'Document Parsing'
    ],
    links: [
      {
        label: 'Live Platform',
        url: 'https://tweakrr.com',
        type: 'live'
      }
    ],
    caseStudy: {
      overview:
        'Tweakrr bridges the gap between academic writing and scientific rigor. It parses raw manuscripts (PDF/DOCX), evaluates arguments, flags unreferenced claims, and contextually recommends peer-reviewed sources formatted in standard bibliographic styles.',
      problem:
        'Researchers and students spend countless hours manually searching literature to back claims and formatting complex bibliographies. Existing tools only check plagiarism without assessing whether individual factual statements actually possess evidentiary citations.',
      architecture: [
        'FastAPI asynchronous microservice handling file upload streams, text extraction, chunking, and metadata parsing.',
        'NLP pipeline utilizing transformer-based sentence tokenizers, claim extraction models, and semantic embedding representations.',
        'Vector retrieval indexing semantic representations of scholarly abstracts to locate relevant peer-reviewed citations in real-time.',
        'MongoDB document store indexing user manuscripts, citations, and generated bibliographies with fast document updates.'
      ],
      engineeringHighlights: [
        'Architected an asynchronous job queue for non-blocking document analysis, streaming real-time parsing progress to the frontend.',
        'Built automated citation generation adhering strictly to APA, MLA, IEEE, and Chicago formatting rules.',
        'Implemented semantic context filtering to ensure recommended sources directly relate to the specific claim rather than general keyword matches.'
      ],
      complexityAndFailures:
        'Academic manuscripts frequently feature multi-column layouts, mathematical notation, and diverse file encodings. I built robust extraction fallbacks to maintain clean semantic chunking without losing contextual claim boundaries.',
      productExperience:
        'Authors paste or upload papers and receive an interactive split-view interface highlighting unsupported claims with instant, one-click citation insertion.',
      shippedArtifacts: [
        'FastAPI Document & NLP Processing Backend',
        'Citation Generator & Reference List Engine',
        'Academic Retrieval Pipeline & Semantic Search Service'
      ],
      outcomes: [
        'Reduced scholarly citation turnaround time by over 70%.',
        'Processed multi-page scientific manuscripts asynchronously with high claim-detection precision.'
      ]
    }
  },
  {
    id: 'asera',
    slug: 'asera',
    title: 'Asera',
    subtitle: 'Autonomous AI Learning & Knowledge Synthesis Platform',
    category: 'Agentic AI / EdTech',
    role: 'AI Systems Architect & Full-Stack Engineer',
    status: 'Live Platform',
    featured: false,
    priority: 3,
    period: '2025',
    summary:
      'Autonomous learning platform that decomposes raw academic materials, slides, and textbooks into personalized multi-step agentic curricula, interactive memory-grounded quizzes, and adaptive tutoring pathways.',
    metrics: [
      'Multi-Step Agentic Workflow & Memory Grounding',
      'Dynamic Syllabus & Active Recall Generation from Raw Slides/PDFs',
      'Contextual Retrieval-Augmented Generation (RAG) & Next.js Delivery'
    ],
    tags: [
      'Next.js',
      'TypeScript',
      'Python',
      'FastAPI',
      'RAG / Vector DB',
      'AI Agents',
      'PyTorch'
    ],
    links: [
      {
        label: 'Live Platform',
        url: 'https://asera-study.vercel.app/',
        type: 'live'
      }
    ],
    caseStudy: {
      overview:
        'Asera is an autonomous learning platform designed to transform passive study material into active learning systems. It ingests lecture slides, notes, and textbook PDFs, decomposing them into progressive concepts, automated active recall challenges, and a stateful AI tutor with verifiable citations.',
      problem:
        'Traditional study workflows are passive and disjointed. Students struggle to create rigorous assessment questions from unstructured notes, while general-purpose AI chat tools lack contextual memory and pedagogical structure.',
      architecture: [
        'Multi-agent pipeline where specialized agents handle document decomposition, concept graph mapping, active recall challenge generation, and answer evaluation.',
        'FastAPI backend with vector retrieval indexing embedded lecture segments with cosine similarity ranking.',
        'Next.js responsive web application delivering real-time streaming AI feedback and interactive study decks.',
        'Stateful conversational memory preserving student progress and weakness vectors across sessions.'
      ],
      engineeringHighlights: [
        'Designed multi-agent state machines with deterministic Pydantic schema validation loops to guarantee syllabus consistency and hallucination-free quiz generation.',
        'Implemented low-latency token streaming for real-time conversational tutoring and instant quiz grading.',
        'Built full Next.js UI with responsive study session views and interactive active recall evaluation.'
      ],
      complexityAndFailures:
        'Preventing agent drift across long multi-step generation tasks required strict schema enforcement with Pydantic and checkpointed intermediate state.',
      productExperience:
        'Learners upload materials and get structured interactive courses with active recall challenges and an on-demand AI tutor that cites the source material.',
      shippedArtifacts: [
        'Asera Web Application (asera-study.vercel.app)',
        'Agentic Course Synthesis Engine',
        'Adaptive Quiz Evaluation Service',
        'Conversational Memory & Tutoring Subsystem'
      ],
      outcomes: [
        'Delivered 100% grounded explanations with direct citation back to source documents.'
      ]
    }
  },
  {
    id: 'retail-shelf-intelligence',
    slug: 'retail-shelf-intelligence',
    title: 'AI Retail Shelf Intelligence',
    subtitle: 'Edge Computer Vision & Automated Inventory Planogram Verification',
    category: 'Computer Vision / Edge ML',
    role: 'Computer Vision Engineer',
    status: 'Research Prototype',
    featured: false,
    priority: 4,
    period: '2025',
    summary:
      'Engineered an internal computer vision system for real-time retail product recognition, shelf occupancy auditing, and automated out-of-stock anomaly detection using custom-trained YOLO models.',
    metrics: [
      'Real-Time Object Detection & Multi-Class SKU Classification',
      'Automated Out-of-Stock (OOS) & Planogram Anomaly Detection',
      'Edge-Optimized Inference Pipelines'
    ],
    tags: [
      'Python',
      'PyTorch',
      'Computer Vision',
      'YOLO',
      'OpenCV',
      'FastAPI',
      'Docker'
    ],
    links: [
      {
        label: 'GitHub Code',
        url: 'https://github.com/Black-fox17',
        type: 'github'
      }
    ],
    caseStudy: {
      overview:
        'A computer vision system designed to audit physical store shelves, verify compliance with manufacturer planograms, and instantly alert store managers to stockouts.',
      problem:
        'Manual shelf audits are slow, error-prone, and lead to lost retail revenue from unnoticed stockouts and misplaced SKUs.',
      architecture: [
        'Custom YOLO object detection architecture fine-tuned on dense retail shelf imagery.',
        'Spatial heuristic engine that computes bounding box adjacencies to verify shelf facings and planogram compliance.',
        'Lightweight inference server optimized for low-latency batch image processing.'
      ],
      engineeringHighlights: [
        'Implemented data augmentation pipelines (perspective skew, lighting variation) to handle unpredictable in-store lighting and camera angles.',
        'Engineered bounding box overlap filters and density heatmaps for automated empty-shelf detection.'
      ],
      complexityAndFailures:
        'Densely packed, visually identical SKUs with partial occlusions presented high classification error rates. Addressed by fusing fine-grained visual feature embeddings with hierarchical spatial constraints.',
      productExperience:
        'Store operators capture shelf photos on mobile devices and receive instant visual overlays showing out-of-stock alerts and misplaced items.',
      shippedArtifacts: [
        'Trained Object Detection Models & Weights',
        'Spatial Planogram Verification Pipeline',
        'Inference & Reporting REST API'
      ],
      outcomes: [
        'High-accuracy SKU localization and instant stockout identification across complex retail displays.'
      ]
    }
  },
  {
    id: 'asl-fingerspelling',
    slug: 'asl-fingerspelling',
    title: 'Continuous ASL Fingerspelling Translation',
    subtitle: 'Edge Vision Kinematics & Seq2Seq Transformer Engine',
    category: 'Computer Vision & Deep Learning',
    role: 'ML & Computer Vision Engineer',
    status: 'Open Source',
    featured: false,
    priority: 3,
    period: 'November 2024',
    summary:
      'Engineered an end-to-end deep learning system translating continuous American Sign Language fingerspelling gestures from video into text in real time. Slices 543 MediaPipe holistic coordinates down to 48 keypoints, standardizes orientation, compresses temporal dimensions via strided 1D convolutions, and decodes character sequences with a sub-30ms mobile TFLite model.',
    metrics: [
      'Sub-30ms Real-Time On-Device TFLite Edge Inference',
      '64x Attention Compute Reduction via Strided 1D Conv Downsampling',
      'Dominant-Hand Automatic Detection & 1-x Reflection Normalization',
      'Normalized Total Levenshtein Distance Evaluation'
    ],
    tags: [
      'TensorFlow',
      'MediaPipe',
      'Transformers',
      'TFLite',
      'Python',
      'OpenCV',
      'Sequence-to-Sequence'
    ],
    links: [
      {
        label: 'GitHub Code',
        url: 'https://github.com/Black-fox17/ASL.git',
        type: 'github'
      },
      {
        label: 'The Deep End Article',
        url: '/blog/continuous-asl-fingerspelling-mediapipe-transformer',
        type: 'article'
      }
    ],
    caseStudy: {
      overview:
        'Continuous fingerspelling involves rapid joint articulators, subtle finger transitions, and high occlusion across signers. This system decouples perception from sequence modeling by extracting 543 3D keypoints via MediaPipe Holistic, isolating 48 essential joints (144 features), and translating character sequences using an autoregressive Transformer decoder.',
      problem:
        'Traditional 3D convolutional networks require massive parameter counts, compute gigabytes of memory, and fail to run in real time on mobile devices. Furthermore, signer handedness (left vs. right) and spatial drift inflate model variance.',
      architecture: [
        'Kinematic Feature Filtering: Drops face and lower-body landmarks, isolating hands (42 keypoints) and upper-body pose anchors (6 keypoints) into a compact 144-dimensional vector per frame.',
        'Dominant Hand & Invariance Preprocessing: Dynamically identifies the signing hand via missing coordinate counts, reflects left-handed signers (1.0 - x) into canonical right-hand coordinates, and applies zero-mean unit-variance scaling.',
        'Temporal Conv1D Downsampling: Three cascaded 1D convolutions with stride 2 compress 128 frames down to 16 temporal tokens, cutting self-attention compute by 64x.',
        'Sequence-to-Sequence Transformer: 4-layer multi-head self-attention encoder and causal cross-attention decoder with label-smoothed categorical cross-entropy and Levenshtein distance optimization.',
        'On-Device TFLite Graph: Wraps preprocessing, dominant hand reflection, and greedy generation loop into a self-contained 39 MB .tflite binary executing in under 30ms.'
      ],
      engineeringHighlights: [
        'Built full modular Python package separating data extraction, TFRecord serialization, model training, and TFLite export.',
        'Engineered vectorized dominant-hand orientation normalization to eliminate the need for training separate left- and right-handed model representations.',
        'Packaged the complete autoregressive decoding loop into a self-contained TFLite serving graph for mobile runtime.'
      ],
      complexityAndFailures:
        'High frame rates (30+ FPS) created extreme temporal redundancy and attention bottlenecks. Resolved by tailoring Conv1D kernel sizes (k=11) to match the physical duration of a fingerspelled phoneme (~1/3 second), reducing attention complexity from 16,384 to 256 computations per head.',
      productExperience:
        'Signers point a standard smartphone camera at continuous fingerspelled phrases and receive real-time, streaming text output locally without cloud latency or video transmission costs.',
      shippedArtifacts: [
        'Standalone model.tflite Edge Inference Binary (~39 MB)',
        'Modular TFRecord Pipeline & MediaPipe Extractor',
        'End-to-End Seq2Seq Transformer Model Architecture',
        'Offline & Video Parquet Inference CLI Tooling'
      ],
      outcomes: [
        'Sub-30ms latency on edge hardware with zero external preprocessing libraries.',
        '64x reduction in attention computational complexity.',
        'Full open-source codebase published on GitHub.'
      ]
    },
    architectureDiagram: {
      layers: [
        {
          name: 'Perception & Slicing',
          items: ['MediaPipe Holistic (543 3D Keypoints)', 'Feature Filtering (48 Keypoints / 144 Features)', 'NaN Dominant Hand Detection']
        },
        {
          name: 'Normalization & Downsampling',
          items: ['1.0 - x Left-Hand Reflection', 'Zero-Mean / Unit-Variance Scaling', '3x Conv1D Strided Embeddings (128 -> 16 steps)']
        },
        {
          name: 'Transformer Seq2Seq Engine',
          items: ['4-Layer Transformer Encoder (num_heads=2)', 'Causal Masked Cross-Attention Decoder', 'Greedy Autoregressive Generator']
        },
        {
          name: 'Edge Deployment',
          items: ['Self-Contained model.tflite (~39 MB)', 'Sub-30ms Mobile Latency', 'Levenshtein Metric Evaluation']
        }
      ]
    }
  }
];

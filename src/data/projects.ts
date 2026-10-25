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
      'ACID Double-Entry Financial Ledger & Distributed Redis Locks',
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
        'Rental transactions and property operations involve high-velocity multi-party financial flows across tenants, landlords, property managers, and maintenance vendors. Coordinating 3 mobile applications while maintaining absolute financial ledger balance without double-crediting or race conditions is a critical distributed systems challenge.',
      architecture: [
        'Modular NestJS micro-monolith running with TypeScript, organized into domain modules for Identity & KYC, Lease Lifecycle, Rent Collection, Financial Ledger, and Maintenance Tickets.',
        'PostgreSQL with strict relational constraints, indexed foreign keys for sub-10ms query execution, and ACID transaction boundaries for all ledger postings.',
        'Cross-platform delivery using Next.js for high-performance responsive web and Capacitor for native iOS and Android compilation, maintaining 90%+ code sharing across platforms.',
        'Double-Entry Accounting & Ledger Architecture: PostgreSQL transactional engine enforcing mathematical equilibrium (debits equal credits) for all rent charges, escrow deposits, and split payouts, protected by distributed Redis idempotency locks.'
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
    title: 'Asera Core & Autonomous Agent Engine',
    subtitle: 'Supervisor Agent Orchestrator & Recursive Language Model (RLM) Context Engine',
    category: 'Agentic AI & Distributed Systems',
    role: 'AI Systems Architect & Framework Creator',
    status: 'Production Framework & Live',
    featured: false,
    priority: 2,
    period: '2024 – 2025',
    summary:
      'Architected and implemented Asera Core from scratch: a lightweight, enterprise-grade multi-agent supervisor framework powered natively by Google Gemini 2.0. Eliminates long-context rot and latency spikes by implementing the MIT CSAIL Recursive Language Models (RLM) paradigm (Alex L. Zhang et al., arXiv:2512.24601), offloading unbounded documents into an isolated Python REPL for programmatic inspection and recursive sub-query delegation. Powers the live Asera autonomous learning platform.',
    metrics: [
      'Zero-Prompt-Bloat RLM Engine (96.7% Input Token Reduction)',
      'Dynamic Supervisor Routing via Just-in-Time Schema Synthesis',
      'Parallel Worker Delegation (asyncio.gather) & Real-Time Streaming',
      'Dual-Tier Memory (L1 In-Memory + L2 PostgreSQL Checkpointer)'
    ],
    tags: [
      'Python',
      'Gemini 2.0',
      'RLM',
      'Supervisor Agent',
      'AsyncIO',
      'PostgreSQL',
      'FastAPI',
      'Next.js',
      'Docker'
    ],
    links: [
      {
        label: 'GitHub Framework (asera-core)',
        url: 'https://github.com/Black-fox17/asera-core',
        type: 'github'
      },
      {
        label: 'The Deep End Article',
        url: '/blog/recursive-language-models-agent-context-scaling',
        type: 'article'
      },
      {
        label: 'Live Study Platform',
        url: 'https://asera-study.vercel.app/',
        type: 'live'
      }
    ],
    caseStudy: {
      overview:
        'Asera Core is a handbuilt Python multi-agent orchestration framework designed to eliminate the heavy abstractions and slow execution runtimes of legacy agent libraries. It combines native Google Gemini 2.0 support (thought signature preservation, dynamic function calling) with a graph-free Supervisor Agent and an isolated Recursive Language Model (RLM) context reasoning engine for unbounded documents.',
      problem:
        'Existing multi-agent frameworks (LangChain, LangGraph, CrewAI) introduce massive dependency bloat, brittle static graph definitions, and severe prompt stuffing. When analyzing massive slide decks or 200-page academic textbooks, injecting entire documents into the attention window causes context rot, quadratic latency (O(N²)), and needle-in-a-haystack hallucination.',
      architecture: [
        'Graph-Free Supervisor Agent: Synthesizes dynamic routing schemas on the fly from worker agent descriptions, dispatching subtasks concurrently via asyncio.gather without rigid DAG pipelines.',
        'Recursive Language Model (RLM) Engine: Stores raw context payloads inside an isolated Python LocalREPL as an external variable. The root LLM writes Python snippets (regex, slices, chunking) and triggers targeted recursive sub-queries (llm_query) only on clean, focused 500-token sections.',
        'Zero-Overhead Tool Reflection: The @agent.tool decorator inspects Python type hints and docstrings to generate standard JSON schemas automatically with zero manual authoring.',
        'Dual-Tier State Management: Thread-safe in-memory L1 cache (AgentStateManager) paired with pluggable L2 ACID persistence (PostgresCheckpointSaver) for resilient multi-turn sessions.',
        'First-Class Async Streaming: Native ainvoke() generator streaming both internal execution telemetry and token chunks in real-time.'
      ],
      engineeringHighlights: [
        'Implemented the MIT CSAIL Recursive Language Models (arXiv:2512.24601) paradigm, cutting input token usage on 150-page documents by 96.7% while increasing multi-hop reasoning accuracy to 94.8%.',
        'Designed lightweight dynamic tool schema reflection with zero Pydantic overhead.',
        'Engineered concurrent worker execution with asyncio.gather, reducing multi-agent delegation latency from 8.2s to 1.8s.'
      ],
      complexityAndFailures:
        'Unbounded context recursion can risk runaway execution or infinite REPL loops. Engineered strict safety bounds: max turn iterations (8), recursion depth capping (1), stdout redirection limits, and deterministic output markers (FINAL<<<answer>>> and FINAL_VAR<<<var_name>>>).',
      productExperience:
        'Developers register specialized worker agents with simple decorators, attach them to a SupervisorAgent, and query the system with real-time streaming tokens and persistent PostgreSQL session state.',
      shippedArtifacts: [
        'Open-Source asera-core Python Framework on GitHub',
        'Modular RLM Context Subsystem with LocalREPL Execution',
        'Full Test Suite with Pytest Coverage',
        'Live Asera Web Application (asera-study.vercel.app)'
      ],
      outcomes: [
        'Achieved 94.8% multi-hop extraction accuracy over 380,000-token corpora.',
        'Eliminated prompt bloat across multi-turn agent conversations with sub-2s parallel delegation.'
      ]
    },
    architectureDiagram: {
      layers: [
        {
          name: '1. Ingestion & RLM REPL Layer',
          items: ['LocalREPL Variable Memory', 'Context Metadata Generator', 'Safe Python Sandbox (re, json, math)']
        },
        {
          name: '2. Supervisor & Reasoning Engine',
          items: ['Dynamic Schema Synthesizer', 'Gemini 2.0 Flash / Pro Encoders', 'Recursive llm_query Sub-Dispatcher']
        },
        {
          name: '3. Worker Agent Fleet (Parallel)',
          items: ['Researcher Agent', 'Code Specialist Agent', 'Data Analyst Agent (asyncio.gather)']
        },
        {
          name: '4. Memory & Persistence Layer',
          items: ['L1 Memory Cache', 'PostgresCheckpointSaver (ACID)', 'ainvoke Real-Time Token Streamer']
        }
      ]
    }
  },
  {
    id: 'breast-cancer-histopathology',
    slug: 'breast-cancer-histopathology',
    title: 'Breast Cancer Histopathology & Diagnostic AI',
    subtitle: 'Clinical Computer Vision & Multimodal Decision Support (IAMRAT / UCH)',
    category: 'Medical AI & Computer Vision',
    role: 'AI Research & Systems Engineer',
    status: 'Clinical Research',
    featured: false,
    priority: 3,
    period: 'Dec 2024 – March 2025',
    summary:
      'Engineered deep convolutional neural networks (ResNet and MobileNet architectures) for automated breast cancer subtype classification from whole-slide histopathological biopsy imagery at University College Hospital (IAMRAT). Coupled classification activations with clinical explanation generators and full-stack pathologist trial workflows.',
    metrics: [
      'Automated Subtype Classification across Whole-Slide Biopsy Images',
      'Multimodal Clinical Decision-Support & Diagnostic Explanation Generator',
      'Full-Stack FastAPI & React Research Annotation Portal'
    ],
    tags: [
      'Path-Foundation',
      'PyTorch',
      'TensorFlow',
      'Gemini 1.5',
      'FastAPI',
      'Docker',
      'Medical AI'
    ],
    links: [
      {
        label: 'GitHub Code',
        url: 'https://github.com/Black-fox17/BreastCancer.git',
        type: 'github'
      },
      {
        label: 'The Deep End Article',
        url: '/blog/histopathology-breast-cancer-path-foundation-llm',
        type: 'article'
      },
      {
        label: 'IAMRAT Research',
        url: '/research',
        type: 'live'
      }
    ],
    caseStudy: {
      overview:
        'Conducted clinical AI research at the Institute of Advanced Medical Research and Training (IAMRAT), University College Hospital Ibadan. Researched deep learning methods to assist pathologists in accurately classifying breast cancer subtypes from dense histological tissue slides and generating interpretable clinical reports.',
      problem:
        'Histopathological cancer diagnosis from biopsy slides is labor-intensive and subjective, with high inter-observer variability among pathologists. Standard black-box computer vision models lack clinical interpretability required for diagnostic trust in medical settings.',
      architecture: [
        'Deep convolutional backbones (ResNet and MobileNet) fine-tuned on cellular patch extractions from digitized whole-slide histopathological biopsy specimens.',
        'Feature extraction pipeline capturing cell nuclei morphology, mitotic figures, and glandular architecture variations.',
        'Diagnostic Explanation Generator: Coupled classification feature activations with structured natural-language diagnostic summaries for clinicians.',
        'Full-stack research portal built with FastAPI and React for sample annotation, trial tracking, and pathologist validation.'
      ],
      engineeringHighlights: [
        'Developed end-to-end data pipelines for gigapixel whole-slide image tile extraction, stain normalization, and artifact filtering.',
        'Engineered multimodal decision-support workflows combining computer vision diagnostic predictions with structured clinical explanation generators.',
        'Delivered low-latency FastAPI backend services for pathologist annotation and real-time inference serving.'
      ],
      complexityAndFailures:
        'Staining variance and slide preparation artifacts across clinical cohorts produced false-positive cellular boundary detections. Implemented color deconvolution and stain-normalization algorithms to standardize RGB distributions across biopsy cohorts before model inference.',
      productExperience:
        'Pathologists upload tissue biopsy slides, visualize highlighted regions of interest and mitotic densities, and inspect automated subtype predictions accompanied by verifiable clinical rationale.',
      shippedArtifacts: [
        'Trained Histopathology Classification Models & Checkpoints',
        'Stain Normalization & Whole-Slide Tile Extraction Pipeline',
        'FastAPI & React Pathologist Annotation Portal'
      ],
      outcomes: [
        'Established strong diagnostic subtype baseline accuracy across whole slide tissue datasets.',
        'Successfully bridged clinical pathology workflows with computer vision interpretability.'
      ]
    },
    architectureDiagram: {
      layers: [
        {
          name: 'Specimen Ingestion & Preprocessing',
          items: ['Whole-Slide Biopsy Digitization', 'Stain Normalization (Color Deconvolution)', 'Tile Slicing & Cellular Patch Extraction']
        },
        {
          name: 'Deep Convolutional Encoders',
          items: ['ResNet Backbone', 'MobileNet Lightweight Model', 'Nuclear Morphology & Mitotic Density Feature Maps']
        },
        {
          name: 'Decision Support & Explanation',
          items: ['Subtype Classifier Head', 'Diagnostic Explanation Generator', 'Confidence & Anomaly Calibration']
        },
        {
          name: 'Clinical Portal',
          items: ['FastAPI Backend Services', 'React Pathologist Annotation UI', 'Trial Workflow Tracking']
        }
      ]
    }
  },
  {
    id: 'rsna-lumbar-spine',
    slug: 'rsna-lumbar-spine',
    title: 'RSNA Lumbar Spine Degenerative Disease Classification',
    subtitle: 'Volumetric MRI Multi-Level 25-Target Deep Learning System (Kaggle RSNA)',
    category: 'Medical AI & Computer Vision',
    role: 'ML Engineer & Researcher',
    status: 'Research & Benchmark',
    featured: false,
    priority: 4,
    period: 'Sept 2024 – Nov 2024',
    summary:
      'Developed an end-to-end deep learning framework for the RSNA 2024 Lumbar Spine Degenerative Classification Challenge. Evaluates 25 simultaneous structural targets (5 vertebral levels across 3 degenerative conditions) on multi-sequence volumetric MRI scans (Sagittal T1, Axial T2, Sagittal T2/STIR). Engineered DICOM VOI LUT normalization, 3D slice spatial coordinate sorting, Generalized Mean (GeM) pooling, and severe-penalized weighted cross-entropy loss, reducing multi-class log loss from 0.684 to 0.441 with 83.6% severe condition recall.',
    metrics: [
      '25 Simultaneous Multi-Level Diagnostic Targets (L1/L2 – L5/S1)',
      '0.441 Multi-Class Log Loss (Down from 0.684 baseline)',
      '83.6% Recall on High-Risk Severe Degenerative Diagnoses',
      'Multi-Series Ensemble (EfficientNetV2, ConvNeXt, GeM)'
    ],
    tags: [
      'PyTorch',
      'DICOM',
      'pydicom',
      'GeM-Pooling',
      'EfficientNetV2',
      'ConvNeXt',
      'FastAI',
      'Kaggle'
    ],
    links: [
      {
        label: 'GitHub Code',
        url: 'https://github.com/Black-fox17/rsna-lumbar-spine-classification-',
        type: 'github'
      },
      {
        label: 'The Deep End Article',
        url: '/blog/rsna-lumbar-spine-mri-deep-learning',
        type: 'article'
      }
    ],
    caseStudy: {
      overview:
        'Developed an advanced volumetric deep learning pipeline for the RSNA 2024 Lumbar Spine Degenerative Classification Challenge. The system grades three key degenerative conditions (spinal canal stenosis, neural foraminal narrowing, and subarticular stenosis) across five intervertebral disc levels (L1/L2 through L5/S1) for 25 simultaneous multi-class predictions directly from raw multi-series DICOM scans.',
      problem:
        'Lumbar spine MRI studies contain hundreds of unsequenced DICOM slices across multiple pulse sequences (Sagittal T1, Axial T2, Sagittal T2/STIR). Radiologists face significant cognitive fatigue grading 25 targets per patient. Naive computer vision models suffer from spatial slice inversion, small lesion dilution under standard global average pooling, and severe class imbalance where missed high-grade compression severely penalizes clinical utility.',
      architecture: [
        'Spatial Coordinate Ingestion: Reconstructed physical slice order using ImagePositionPatient vectors projected onto the acquisition normal axis, resolving arbitrary DICOM folder ordering.',
        'VOI LUT Windowing & Monochrome Correction: Standardized pixel intensities with scanner-calibrated Look-Up Tables and inverted MONOCHROME1 polarities.',
        '2.5D Volumetric Sampling: Extracted 16 equidistant physical slices across the spine volume, bypassing the extreme GPU VRAM bottlenecks of native 3D convolutions.',
        'Generalized Mean (GeM) Pooling (p=3.0): Replaced Global Average Pooling with parameterized power pooling to focus feature representations on focal cord and nerve root impingements rather than broad healthy background tissue.',
        'Multi-Task Classifier Heads: Built 5 parallel linear heads outputting calibrated Dirichlet severity distributions for L1/L2 through L5/S1.',
        'Severe-Penalized Loss Optimization: Designed RSNAWeightedLogLoss (1.0 : 2.0 : 4.0 class weighting) coupled with AnySevereSpineLoss auxiliary binary cross-entropy on max-severe probabilities.'
      ],
      engineeringHighlights: [
        'Eliminated patient data leakage using strict 5-Fold GroupKFold stratified by patient study ID.',
        'Achieved state-of-the-art 0.441 weighted multi-class log loss and 83.6% severe condition recall, outperforming single-slice baselines (0.684 log loss).',
        'Implemented automatic mixed precision (torch.cuda.amp), cosine annealing with warm-up, and test-time augmentation (TTA).'
      ],
      complexityAndFailures:
        'Standard global average pooling washed out tiny focal stenosis lesions that occupied less than 5% of the total MRI slice area. Implementing Generalized Mean (GeM) pooling with p=3.0 mathematically accentuated high-activation compression zones, yielding a 0.047 reduction in log loss and a 6.6% boost in severe recall.',
      productExperience:
        'Modular pipeline accepting raw DICOM series, sorting physical coordinates, applying medical contrast normalization, and exporting calibrated probability matrices formatted for clinical triage and Kaggle submission standards.',
      shippedArtifacts: [
        'Complete Open-Source PyTorch Repository on GitHub',
        'Modular DICOM Preprocessing & 3D Normalization Utilities',
        'Multi-Task Spine Classifier Architecture Checkpoints',
        'First-Principles Technical Research Monograph'
      ],
      outcomes: [
        'Achieved 0.441 weighted multi-class log loss across all 25 clinical target dimensions.',
        'Attained 83.6% recall on critical high-risk severe spinal stenosis cases.'
      ]
    },
    architectureDiagram: {
      layers: [
        {
          name: '1. DICOM Coordinate Ingestion',
          items: ['ImagePositionPatient Normal Projection', 'VOI LUT Dynamic Windowing', 'Uniform 16-Slice 2.5D Volumetric Sampling']
        },
        {
          name: '2. Deep Feature Backbones',
          items: ['EfficientNetV2-S', 'ConvNeXt-Tiny', 'Generalized Mean (GeM) Pooling (p=3.0)']
        },
        {
          name: '3. Multi-Task Heads (25 Targets)',
          items: ['L1/L2 Head', 'L2/L3 Head', 'L3/L4 Head', 'L4/L5 Head', 'L5/S1 Head']
        },
        {
          name: '4. Optimization & Ensembling',
          items: ['Weighted Log Loss (1:2:4)', 'Auxiliary Any-Severe Loss', 'Multi-Series Ensemble + TTA']
        }
      ]
    }
  },
  {
    id: 'point-supervised-loveda-segmentation',
    slug: 'point-supervised-loveda-segmentation',
    title: 'Point-Supervised LoveDA Remote Sensing Segmentation',
    subtitle: 'Weakly-Supervised Aerial Segmentation via Partial Cross-Entropy (pCE)',
    category: 'Computer Vision & Deep Learning',
    role: 'Computer Vision Researcher & ML Engineer',
    status: 'Research & Benchmark',
    featured: false,
    priority: 4,
    period: 'October 2026',
    summary:
      'Investigated weakly-supervised semantic segmentation for high-resolution aerial remote sensing on the LoveDA Urban benchmark using sparse point supervision (1%, 5%, 10%, 25%) and Partial Cross-Entropy (pCE) loss with an end-to-end PyTorch U-Net. Uncovered empirical non-monotonic scaling where 5% point density achieved peak generalization (+54.1% mIoU jump to 0.1747, 0.2670 Mean Dice), proving how unregularized point supervision beyond critical percolation thresholds causes point-memorization overfitting.',
    metrics: [
      '5% Density Optimal Generalization (+54.1% mIoU Surge to 0.1747)',
      'Partial Cross-Entropy (pCE) with Zero-Gradient Coordinate Masking',
      'End-to-End PyTorch U-Net & Tiled Full-Scene Sliding Inference',
      'Empirical Non-Monotonic Scaling Discovery on LoveDA Urban'
    ],
    tags: [
      'PyTorch',
      'U-Net',
      'Remote Sensing',
      'LoveDA',
      'pCE Loss',
      'Weak Supervision',
      'GIS / Earth Observation'
    ],
    links: [
      {
        label: 'GitHub Code',
        url: 'https://github.com/Black-fox17/point-supervised-loveda-segmentation',
        type: 'github'
      },
      {
        label: 'The Deep End Article',
        url: '/blog/point-supervised-loveda-segmentation-pce',
        type: 'article'
      }
    ],
    caseStudy: {
      overview:
        'Point-supervised semantic segmentation framework for high-resolution aerial remote sensing imagery on the LoveDA Urban benchmark (1,156 scenes, 7 land-cover categories). The study evaluates how sparse random point clicks (1%, 5%, 10%, 25% pixel coverage) govern full-scene generalization when trained solely via Partial Cross-Entropy loss without dense spatial priors.',
      problem:
        'Dense pixel-level polygon annotation of aerial remote sensing imagery costs $50-$150 per km² and requires dozens of human-hours per tile due to complex building shadows, narrow road networks, and irregular water canals. Weak point supervision slashes annotation time by >95%, but training deep networks on sparse point targets risks severe label dilution and boundary degradation.',
      architecture: [
        'Partial Cross-Entropy (pCE) Engine: Formulates loss strictly over labelled coordinate subset L, mapping unannotated coordinates U to ignore_index=-1 with zero backpropagation gradient contribution.',
        'Deep PyTorch U-Net: 3-stage double-conv encoder (64->128->256 channels), 512-dimensional bottleneck, and transposed convolution upsampling decoder with spatial skip connections.',
        'Simulated Point-Supervision Sampler: Extracts random coordinate subsets at parameterized density fractions (1%, 5%, 10%, 25%) while guaranteeing land-cover representation in training crops.',
        'Tiled Full-Scene Sliding Inference: Decomposes large gigapixel scenes into non-overlapping 256x256 windows, executes batch inference, and reconstructs seamless global semantic prediction masks.'
      ],
      engineeringHighlights: [
        'Discovered the Non-Monotonic Scaling Law of Point Supervision: 5% density achieved peak mIoU (0.1747, +54.1% over 1% baseline), while quadrupling density to 25% collapsed mIoU to 0.1192 due to point-memorization overfitting.',
        'Observed emergence of linear and regional features: Road IoU jumped from 0.0000 to 0.0700 and Water IoU jumped from 0.0000 to 0.2724 at 5% density.',
        'Developed full Pytest test suite validating exact numerical equivalence between partial cross-entropy and standard cross-entropy on fully-labelled tensors.'
      ],
      complexityAndFailures:
        'Increasing point density to 10% and 25% caused dense point clusters within homogeneous textures. In the absence of spatial smoothness regularization, the U-Net created fractured decision boundaries that overfit to isolated coordinate clicks, degrading held-out scene mIoU by 31.8%. Identified that point supervision requires spatial continuity priors or contrastive boundary regularizers beyond 5% density.',
      productExperience:
        'Remote sensing analysts inspect raw aerial tiles, view simulated sparse point overlays, and evaluate global semantic land-cover segmentations across buildings, roads, water, forests, and barren terrain.',
      shippedArtifacts: [
        'Open-Source PyTorch Repository on GitHub',
        'Modular Point-Supervision Sampler & pCE Loss Module',
        'Tiled Sliding Inference Engine for Gigapixel Imagery',
        'Empirical Benchmarking Suite with CSV Metric Logging'
      ],
      outcomes: [
        'Delivered 0.1747 mIoU and 0.2670 Mean Dice with only 5% point annotation coverage.',
        'Demonstrated >95% human annotation time reduction compared to dense polygon delineation.'
      ]
    },
    architectureDiagram: {
      layers: [
        {
          name: '1. Tile Ingestion & Point Sampling',
          items: ['LoveDA Urban Tile Loading', 'Random Point Sampler (1%, 5%, 10%, 25%)', 'ignore_index=-1 Zero-Gradient Mask']
        },
        {
          name: '2. Deep U-Net Architecture',
          items: ['DoubleConv Encoder (64-256)', '512-dim Bottleneck', 'Skip-Connected Transposed Conv Decoder']
        },
        {
          name: '3. Loss & Optimization',
          items: ['Partial Cross-Entropy (pCE)', 'Adam Optimizer (lr=1e-3)', 'Sparse-Coordinate Reduction']
        },
        {
          name: '4. Tiled Sliding Inference',
          items: ['Non-Overlapping 256x256 Tiling', 'Batch Inference', 'Stitched Full-Scene Semantic Map']
        }
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

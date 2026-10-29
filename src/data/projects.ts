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
    period: 'March 2026 - Present',
    summary:
      'Architected and shipped a unified PropTech & FinTech ecosystem linking tenants, landlords, property managers, and payment processors. Operates 3 live production applications across Apple App Store (UpwardPay iOS), Google Play Store (Upward Pay & Upward PM Android), and web, backed by a high-throughput NestJS monolithic backend with 500+ production REST APIs.',
    metrics: [
      '3 Production Apps Shipped (Apple App Store & Google Play)',
      '500+ Production REST APIs & Central Server Architecture',
      'Multi-Platform Sync: Internal Apps & External PM Platforms',
      'Structured Relational PostgreSQL Database'
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
        label: 'UpwardPay',
        url: 'https://apps.apple.com/us/app/upwardpay/id6807898152',
        type: 'appstore'
      },
      {
        label: 'Upward Pay',
        url: 'https://play.google.com/store/apps/details?id=com.goodtenants.upward',
        type: 'playstore'
      },
      {
        label: 'Upward PM',
        url: 'https://play.google.com/store/apps/details?id=com.goodtenants.upward.pm',
        type: 'playstore'
      },
      {
        label: 'Web Portal',
        url: 'https://upward.goodtenants.io',
        type: 'live'
      }
    ],
    caseStudy: {
      overview:
        'The Upward Ecosystem modernizes rental property operations. It replaces manual rent collection, paper tenancy records, and delayed maintenance with connected mobile apps (UpwardPay on the Apple App Store, and Upward Pay & Upward PM on the Google Play Store) backed by a central backend.',
      problem:
        'The core engineering challenge was orchestrating a multi-platform ecosystem powered by a single central server. The backend had to handle simultaneous requests from our in-house Property Manager app and Tenant app, while syncing with external property management platforms that integrate directly with the tenant app. This required careful database design, preparing both apps to meet strict Google Play standards, and successfully navigating Apple App Store guidelines to deploy UpwardPay live.',
      architecture: [
        'Modular NestJS server written in TypeScript, organized into clean modules for Identity, Leases, Rent Payments, and Maintenance Requests.',
        'PostgreSQL database with indexed keys for fast query performance and strict transaction boundaries for safe payments.',
        'Cross-platform mobile apps built with Next.js and Capacitor, sharing 90%+ code across iOS and Android.',
        'Platform Bridge: Engineered the backend to handle mobile app requests while providing secure APIs for external property management platforms.'
      ],
      engineeringHighlights: [
        'Took complete ownership of launching UpwardPay live on the Apple App Store, managing the submission process, Apple guidelines, and TestFlight reviews.',
        'Shipped and maintained both Upward Pay and Upward PM on the Google Play Store, managing release tracks and app updates.',
        'Engineered 500+ secure REST APIs across key domains with token authorization, role-based access, and input validation.',
        'Implemented safe database locking to prevent duplicate rent payments and ensure data consistency during peak periods.',
        'Built automated CI/CD workflows using GitHub Actions to run tests, build Docker containers, and verify deployments.'
      ],
      complexityAndFailures:
        'Coordinating concurrent updates between the tenant app, property manager app, and external property management platforms was the primary challenge. I implemented clean data boundaries and transaction locks in PostgreSQL, preventing conflicting updates and ensuring data stays accurate across all platforms.',
      productExperience:
        'Tenants easily view leases, pay rent securely, and submit maintenance requests. Property managers track occupancy, receive rent alerts, sign digital documents, and access clear financial summaries.',
      shippedArtifacts: [
        'UpwardPay iOS Application (Apple App Store)',
        'Upward Pay Android Application (Google Play Store)',
        'Upward PM Android Application (Google Play Store)',
        'Upward Web Administration & Landlord Portal (upward.goodtenants.io)',
        'NestJS Core API Services & Webhook Cluster',
        'Automated GitHub Actions Deployment Pipeline'
      ],
      outcomes: [
        'Seamless synchronization between internal mobile apps and external property management platforms.',
        'Reduced property onboarding and tenant verification from days to minutes.',
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
          items: ['NestJS REST API (500+ Endpoints)', 'Role-Based Access & Auth Guards', 'Data Validation & Serialization', 'Rate Limiting']
        },
        {
          name: 'Payment & Third-Party Integration',
          items: ['Payment Processing', 'Paystack & Flutterwave Integrations', 'KYC & Identity Verification']
        },
        {
          name: 'Data & Infrastructure',
          items: ['PostgreSQL Database', 'NestJS Monolith Engine', 'AWS Cloud', 'GitHub Actions CI/CD']
        }
      ]
    }
  },
  {
    id: 'tweakrr',
    slug: 'tweakrr',
    title: 'Tweakrr',
    subtitle: 'AI Research Assistant & Academic Citation Platform',
    category: 'AI & Document Intelligence',
    role: 'AI Engineer & Backend Engineer',
    status: 'Live Platform',
    featured: true,
    priority: 2,
    period: 'March 2025 - May 2025',
    summary:
      'Built an intelligent research platform that helps students and researchers cite sources accurately. The system scans papers, flags statements that lack proof, and finds relevant academic sources to insert proper citations automatically.',
    metrics: [
      'Background Document Parsing & Analysis Engine',
      'Automated Citation & Bibliography Generation (APA, IEEE, Harvard)',
      'Fast Semantic Search across Academic Papers',
      'Scalable FastAPI and MongoDB Backend'
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
        'Tweakrr helps researchers and students write with confidence. It reads papers in PDF and Word formats, finds statements that need supporting evidence, and recommends relevant academic papers with formatted citations.',
      problem:
        'Researchers spend hours searching for papers to back up claims and formatting bibliographies by hand. Most existing tools only check for plagiarism rather than checking whether claims actually have solid references.',
      architecture: [
        'FastAPI backend that handles document uploads, text extraction, and section analysis.',
        'Text processing pipeline that identifies individual claims and converts text into search embeddings.',
        'Fast vector search to find matching research papers in seconds.',
        'MongoDB database storing manuscripts, citations, and user bibliographies.'
      ],
      engineeringHighlights: [
        'Built a background job queue for non-blocking document processing with live progress updates.',
        'Created automated citation formatting for APA, MLA, IEEE, and Chicago styles.',
        'Implemented semantic filtering to ensure recommended sources directly support the specific claim.'
      ],
      complexityAndFailures:
        'Academic papers often have multi-column layouts, tables, and formulas that confuse standard text extractors. I built fallback parsers to cleanly preserve sentence structure and claim boundaries.',
      productExperience:
        'Users upload their papers and see an intuitive split-screen view highlighting unsupported statements, with one-click buttons to add verified citations.',
      shippedArtifacts: [
        'FastAPI Document & Text Processing Backend',
        'Citation Generator & Reference List Engine',
        'Academic Search Pipeline & Database'
      ],
      outcomes: [
        'Cut paper citation and referencing time by over 70%.',
        'Processed complex multi-page academic papers quickly and accurately.'
      ]
    }
  },
  {
    id: 'asera',
    slug: 'asera',
    title: 'Asera Core & Autonomous Agent Engine',
    subtitle: 'Supervisor Agent Framework & Long-Document Context Engine',
    category: 'Agentic AI & Distributed Systems',
    role: 'AI Systems Architect & Framework Creator',
    status: 'Production Framework & Live',
    featured: false,
    priority: 2,
    period: '2024 - 2025',
    summary:
      'Built Asera Core from scratch: an open-source framework for running teams of AI agents orchestrated across frontier LLMs (Claude, OpenAI, DeepSeek, Gemini). Instead of stuffing entire books or massive PDFs directly into an AI prompt, which causes slowness and mistakes, it lets the agent write Python code to inspect, slice, and query documents piece by piece. This powers the live Asera study platform.',
    metrics: [
      'Zero-Prompt-Bloat Context Engine (96.7% Token Reduction)',
      'Smart Coordinator Routing to Specialized Worker Agents',
      'Parallel Worker Execution for Fast Responses',
      'Session Memory Saved to PostgreSQL'
    ],
    tags: [
      'Python',
      'Frontier LLMs',
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
        'Asera Core is a lightweight Python framework designed to run teams of AI agents without the bloat and slow speeds of existing tools. It pairs frontier LLMs (Claude, OpenAI, DeepSeek, Gemini) with a model-agnostic supervisor agent that breaks down tasks and delegates them to specialist agents, while handling huge documents cleanly through Python code execution.',
      problem:
        'Existing AI agent frameworks are often bulky and difficult to customize. Furthermore, when working with long documents like 200-page textbooks or slide decks, pasting everything directly into the prompt makes the AI slow, expensive, and prone to hallucinations.',
      architecture: [
        'Flexible Supervisor Agent: Automatically detects available worker agents and routes tasks to them in parallel.',
        'Recursive Context Engine: Keeps long documents in a secure Python environment. The AI writes small Python scripts to filter and extract only the relevant paragraphs before answering.',
        'Simple Tool Decorators: Any standard Python function can become an agent tool with a single @agent.tool decorator.',
        'Dual-Tier Memory: Fast temporary memory for active conversations, backed by PostgreSQL for long-term session storage.',
        'Real-Time Streaming: Streams responses and intermediate agent thoughts live to the user interface.'
      ],
      engineeringHighlights: [
        'Implemented recursive document processing, cutting prompt token usage on 150-page files by 96.7% while boosting reasoning accuracy.',
        'Built automated tool discovery with zero unnecessary boilerplate code.',
        'Ran worker agents in parallel using Python asyncio, reducing multi-agent delegation time from 8.2s down to 1.8s.'
      ],
      complexityAndFailures:
        'Running code in an automated loop can risk infinite loops. I added strict safety caps: a maximum of 8 turns, loop limits, output size guards, and deterministic completion markers.',
      productExperience:
        'Developers define worker agents with simple Python functions, connect them to a supervisor agent, and query the system with live streaming responses and persistent conversation memory.',
      shippedArtifacts: [
        'Open-Source asera-core Python Framework on GitHub',
        'Modular Context Subsystem with Python Sandbox',
        'Full Test Suite with Pytest Coverage',
        'Live Asera Web Application (asera-study.vercel.app)'
      ],
      outcomes: [
        'Achieved 94.8% extraction accuracy across large 380,000-token documents.',
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
          items: ['Dynamic Schema Synthesizer', 'Frontier LLM Encoders (Claude, GPT-4o, DeepSeek, Gemini)', 'Recursive llm_query Sub-Dispatcher']
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
    period: 'Dec 2024 - March 2025',
    summary:
      'Trained computer vision models on digitized biopsy slides to classify breast cancer subtypes at University College Hospital (IAMRAT). Connected model predictions to clinical explanation tools and built full-stack web apps for pathologists to review results.',
    metrics: [
      'Automated Subtype Classification across Biopsy Images',
      'Clinical Decision-Support & Diagnostic Explanations with Multimodal LLMs',
      'Full-Stack FastAPI & React Research Annotation Portal'
    ],
    tags: [
      'Path-Foundation',
      'PyTorch',
      'TensorFlow',
      'Multimodal LLMs',
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
      }
    ],
    caseStudy: {
      overview:
        'Conducted clinical AI research at the Institute of Advanced Medical Research and Training (IAMRAT), University College Hospital Ibadan. Researched deep learning methods to assist pathologists in accurately classifying breast cancer subtypes from histological tissue slides and generating interpretable clinical reports.',
      problem:
        'Diagnosing cancer from biopsy slides is time-consuming and prone to human variation. In addition, standard deep learning models act as black boxes, making it difficult for doctors to trust their predictions without clear explanations.',
      architecture: [
        'Deep neural networks (ResNet and MobileNet) fine-tuned on cellular patch images from digitized biopsy slides.',
        'Feature extraction pipeline capturing cell shape, division patterns, and tissue organization.',
        'Diagnostic Explanation Generator: Connected model predictions to structured clinical summaries for doctors.',
        'Full-stack research portal built with FastAPI and React for sample labeling and trial tracking.'
      ],
      engineeringHighlights: [
        'Developed data pipelines to extract image tiles from gigapixel slides, normalize stains, and filter out visual artifacts.',
        'Combined computer vision predictions with multimodal LLMs to generate structured, readable clinical diagnostic explanations.',
        'Built responsive FastAPI backend services for real-time model inference and researcher annotation.'
      ],
      complexityAndFailures:
        'Differences in lab staining techniques caused false cellular boundaries. I added stain normalization algorithms to calibrate colors across slide cohorts before running predictions.',
      productExperience:
        'Pathologists upload tissue biopsy slides, see highlighted areas of interest, and review subtype classifications accompanied by clear clinical reasoning.',
      shippedArtifacts: [
        'Trained Histopathology Classification Models & Checkpoints',
        'Stain Normalization & Whole-Slide Tile Extraction Pipeline',
        'FastAPI & React Pathologist Annotation Portal'
      ],
      outcomes: [
        'Established strong diagnostic baseline accuracy across whole slide tissue datasets.',
        'Successfully connected clinical pathology workflows with clear, interpretable AI predictions.'
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
    period: 'Sept 2024 - Nov 2024',
    summary:
      'Built a deep learning system for the RSNA Lumbar Spine Degenerative Disease Challenge. Analyzes 3D MRI scans to detect 3 degenerative spine conditions across 5 disc levels simultaneously. Standardized medical imaging coordinates and applied specialized pooling, improving severe condition detection to 83.6% recall.',
    metrics: [
      '25 Simultaneous Diagnostic Targets (L1/L2 to L5/S1)',
      '0.441 Multi-Class Log Loss (Down from 0.684 baseline)',
      '83.6% Recall on Severe Spine Conditions',
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
        'Developed a deep learning pipeline for the RSNA 2024 Lumbar Spine Challenge. The system evaluates 3 degenerative conditions across 5 disc levels (L1/L2 through L5/S1) for 25 simultaneous predictions directly from raw multi-sequence MRI scans.',
      problem:
        'Lumbar spine MRI exams contain hundreds of individual slices across different pulse sequences. Manually grading 25 conditions per patient is exhausting for radiologists. Standard computer vision models often misorder slices or miss small, localized nerve compressions.',
      architecture: [
        'Coordinate Reordering: Reconstructed true physical slice order using 3D patient coordinates from DICOM metadata.',
        'Contrast Normalization: Standardized pixel intensities using calibrated DICOM lookup tables.',
        'Volumetric Sampling: Extracted 16 evenly spaced slices across each MRI series to balance detail with GPU memory limits.',
        'Generalized Mean (GeM) Pooling: Focused neural network attention on small, localized nerve compressions rather than healthy surrounding tissue.',
        'Multi-Task Heads: Built parallel output heads to predict severity levels across all 5 disc levels at once.',
        'Weighted Loss Function: Penalized missed severe cases heavily to ensure critical diagnoses are never overlooked.'
      ],
      engineeringHighlights: [
        'Prevented patient data leakage using strict 5-fold cross-validation grouped by patient ID.',
        'Achieved state-of-the-art 0.441 weighted log loss and 83.6% severe condition recall, substantially beating single-slice baselines.',
        'Implemented automatic mixed precision training and test-time augmentation for maximum accuracy.'
      ],
      complexityAndFailures:
        'Standard image pooling washed out tiny spinal narrowing lesions that occupied less than 5% of the slice area. Implementing Generalized Mean (GeM) pooling accentuated high-activation compression zones, improving severe condition recall by 6.6%.',
      productExperience:
        'A clean modular pipeline that accepts raw MRI scans, sorts slices into correct 3D order, normalizes contrast, and outputs clear severity predictions for medical triage.',
      shippedArtifacts: [
        'Complete Open-Source PyTorch Repository on GitHub',
        'Modular DICOM Preprocessing & 3D Normalization Utilities',
        'Multi-Task Spine Classifier Architecture Checkpoints',
        'First-Principles Technical Research Monograph'
      ],
      outcomes: [
        'Achieved 0.441 weighted multi-class log loss across all 25 target conditions.',
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
    subtitle: 'Weakly-Supervised Aerial Segmentation via Partial Cross-Entropy',
    category: 'Computer Vision & Deep Learning',
    role: 'Computer Vision Researcher & ML Engineer',
    status: 'Research & Benchmark',
    featured: false,
    priority: 4,
    period: 'October 2026',
    summary:
      'Researched aerial satellite image segmentation using single point clicks instead of drawing tedious polygon borders. Using an end-to-end PyTorch U-Net, discovered that labeling just 5% of pixels achieved the best real-world generalization, reducing manual labeling time by over 95%.',
    metrics: [
      '5% Density Optimal Generalization (+54.1% mIoU Gain)',
      'Partial Cross-Entropy (pCE) for Sparse Point Labels',
      'End-to-End PyTorch U-Net with Tiled Full-Scene Inference',
      'Empirical Scaling Discovery on LoveDA Urban Benchmark'
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
        'A point-supervised image segmentation framework for high-resolution aerial imagery on the LoveDA Urban dataset (1,156 scenes, 7 land-cover categories). The study examines how sparse point clicks (1%, 5%, 10%, 25% pixel coverage) allow neural networks to segment full satellite scenes.',
      problem:
        'Drawing exact polygon outlines on aerial satellite images is slow and expensive, costing dozens of hours per tile. While clicking single points cuts annotation time by 95%, training neural networks on sparse points can lead to poor boundaries and overfitting.',
      architecture: [
        'Partial Cross-Entropy: Calculates training loss only on the specific labeled points, ignoring unlabeled pixels completely during backpropagation.',
        'PyTorch U-Net: Encoder-decoder convolutional network with skip connections to preserve both high-level context and fine spatial details.',
        'Point Sampler: Selects random coordinate points across land-cover classes to simulate fast human labeling.',
        'Tiled Sliding Window: Slices large satellite images into 256x256 tiles for batch processing, stitching the predictions back into full-resolution maps.'
      ],
      engineeringHighlights: [
        'Discovered that 5% point density achieved peak accuracy (0.1747 mIoU), whereas adding more points up to 25% actually hurt performance due to point-memorization.',
        'Observed clear recognition of thin roads and water bodies emerging at just 5% supervision.',
        'Built unit test suites in Pytest validating numerical consistency between partial and standard cross-entropy.'
      ],
      complexityAndFailures:
        'Higher point densities (10% and 25%) caused points to bunch together, leading the model to overfit to individual clicks. Identified that point-supervised models require spatial smoothness constraints when scaling beyond 5% density.',
      productExperience:
        'Remote sensing analysts can view raw aerial tiles, see sparse point overlays, and evaluate automated segmentation masks across buildings, roads, water, forests, and ground.',
      shippedArtifacts: [
        'Open-Source PyTorch Repository on GitHub',
        'Modular Point Sampler & Partial Cross-Entropy Loss Module',
        'Tiled Sliding Inference Engine for Large Aerial Scenes',
        'Benchmarking Suite with CSV Metric Logging'
      ],
      outcomes: [
        'Delivered 0.1747 mIoU and 0.2670 Mean Dice with only 5% point annotation coverage.',
        'Demonstrated a 95% reduction in manual labeling time compared to full polygon outlines.'
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
    subtitle: 'Mobile Vision Kinematics & Sequence-to-Sequence Translation',
    category: 'Computer Vision & Deep Learning',
    role: 'ML & Computer Vision Engineer',
    status: 'Open Source',
    featured: false,
    priority: 3,
    period: 'November 2024',
    summary:
      'Built a deep learning system that translates continuous American Sign Language fingerspelling from smartphone video into text in real time. Extracted hand landmarks with MediaPipe, compressed motion over time, and ran translation through a lightweight Transformer model on mobile devices with sub-30ms latency.',
    metrics: [
      'Fast Sub-30ms Real-Time Mobile TFLite Inference',
      '64x Compute Reduction via Temporal Convolutions',
      'Automatic Dominant-Hand Detection and Normalization',
      'Accurate Character-Level Sequence Translation'
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
        'Translating continuous fingerspelling from camera video is challenging due to rapid hand movements and finger occlusions. This system tracks 3D hand and body keypoints using MediaPipe Holistic and translates gesture sequences into text with a lightweight Transformer model running entirely on-device.',
      problem:
        'Processing raw video directly with heavy 3D networks requires too much compute and drains mobile battery quickly. Differences in signer handedness (left vs. right) also make models harder to train reliably.',
      architecture: [
        'Landmark Filtering: Filters out face and lower-body landmarks, tracking only 48 keypoints across the hands and upper body.',
        'Hand Normalization: Automatically detects the signing hand and normalizes coordinates so the model handles both left- and right-handed signers seamlessly.',
        'Motion Downsampling: Compresses video frames over time using 1D convolutions to speed up processing without losing gesture information.',
        'Sequence-to-Sequence Transformer: Translates compressed motion sequences into letters and words.',
        'On-Device TFLite Model: Packages preprocessing, hand reflection, and text generation into a single 39 MB file running in under 30ms on mobile.'
      ],
      engineeringHighlights: [
        'Built a modular Python pipeline for landmark extraction, training data preparation, and model export.',
        'Created automatic left/right hand normalization to avoid training separate models.',
        'Packaged the complete model into a self-contained mobile binary requiring no external cloud server.'
      ],
      complexityAndFailures:
        'High video frame rates created redundant data and slowed down translation. I aligned the convolution filters with the typical duration of a fingerspelled letter, reducing computation by 64x while keeping recognition sharp.',
      productExperience:
        'Signers point a standard smartphone camera at their hand while signing and receive instant live text on their screen without requiring internet access.',
      shippedArtifacts: [
        'Standalone model.tflite Mobile Binary (~39 MB)',
        'Landmark Extraction & Training Pipeline',
        'End-to-End Transformer Model Architecture',
        'Offline Video Evaluation CLI Tools'
      ],
      outcomes: [
        'Sub-30ms translation speed directly on mobile hardware.',
        '64x reduction in model computational overhead.',
        'Published as full open-source code on GitHub.'
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

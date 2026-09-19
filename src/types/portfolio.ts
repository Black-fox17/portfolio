export interface ProjectLink {
  label: string;
  url: string;
  type: 'live' | 'github' | 'playstore' | 'case-study';
}

export interface CaseStudy {
  overview: string;
  problem: string;
  architecture: string[];
  engineeringHighlights: string[];
  complexityAndFailures: string;
  productExperience: string;
  shippedArtifacts: string[];
  outcomes: string[];
}

export interface Project {
  id: string;
  slug: string;
  title: string;
  subtitle: string;
  category: 'PropTech / FinTech' | 'AI & Document Intelligence' | 'Agentic AI / EdTech' | 'Computer Vision / Edge ML';
  role: string;
  status: 'In Production' | 'Live Platform' | 'Research Prototype';
  featured: boolean;
  priority: number;
  period: string;
  summary: string;
  metrics: string[];
  tags: string[];
  links: ProjectLink[];
  caseStudy: CaseStudy;
  architectureDiagram?: {
    layers: {
      name: string;
      items: string[];
    }[];
  };
}

export interface ExperienceItem {
  id: string;
  company: string;
  role: string;
  location: string;
  type: 'Hybrid' | 'Remote' | 'On-site';
  period: string;
  current: boolean;
  summary: string;
  highlights: string[];
  technologies: string[];
  domain: string;
}

export interface SkillCategory {
  title: string;
  description: string;
  skills: {
    name: string;
    focus?: string;
  }[];
}

export interface ArticleItem {
  id: string;
  title: string;
  date: string;
  readTime: string;
  category: string;
  excerpt: string;
  slug: string;
  link?: string;
  isExternal?: boolean;
}

export interface OpenSourceReproduction {
  id: string;
  title: string;
  description: string;
  category: 'Transformers & NLP' | 'Optimization' | 'Vision & Multimodal';
  technologies: string[];
  githubUrl: string;
}

export interface LeadershipItem {
  metric: string;
  label: string;
  role: string;
  organization: string;
  description: string;
}

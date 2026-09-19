import React from 'react';
import { Search, Compass, Cpu, Rocket, LineChart } from 'lucide-react';

interface Principle {
  step: string;
  title: string;
  tagline: string;
  description: string;
  icon: React.ElementType;
  points: string[];
}

const principles: Principle[] = [
  {
    step: '01',
    title: 'Understand the Problem & System Boundaries',
    tagline: 'Start from core domain constraints rather than framework hype.',
    description:
      'Before writing code, I analyze domain workflows, data invariants, and failure costs. I evaluate whether machine learning is genuinely required or if deterministic algorithms offer lower latency and higher reliability.',
    icon: Search,
    points: [
      'Model real-world user workflows and domain constraints',
      'Benchmark deterministic vs. ML-based tradeoffs',
      'Define clear interface boundaries and data contracts'
    ]
  },
  {
    step: '02',
    title: 'Design for Failure & Asynchrony',
    tagline: 'Architect around gateway degradations, network retries, and ledger integrity.',
    description:
      'I treat network latency and external third-party outages as inevitable. Systems are designed with idempotency keys, circuit-breaker failovers, and transactional consistency to protect financial ledgers and state machines.',
    icon: Compass,
    points: [
      'ACID transaction boundaries & relational indexing',
      'Dual-gateway fallback & circuit breakers',
      'Asynchronous worker queues for long-running AI tasks'
    ]
  },
  {
    step: '03',
    title: 'Bridge AI & Usable Product Engineering',
    tagline: 'Turn raw embeddings and APIs into intuitive, low-latency interfaces.',
    description:
      'An algorithm is only valuable if users can operate it without friction. I bridge backend services with cross-platform web and mobile frontends, optimizing perceived latency through streaming responses and optimistic updates.',
    icon: Cpu,
    points: [
      'Next.js SSR & Capacitor mobile integration',
      'Streaming LLM token generation & progressive loading',
      'Sub-50ms optimistic UI states'
    ]
  },
  {
    step: '04',
    title: 'Ship with Rigorous Release Controls',
    tagline: 'Automate build artifacts, branch controls, and store deployment.',
    description:
      'I own production environments across web and Android. Automated CI/CD pipelines validate types, run test suites, containerize microservices, and manage Google Play Console internal and production rollout tracks.',
    icon: Rocket,
    points: [
      'GitHub Actions CI/CD with Docker multi-stage builds',
      'Google Play Console staged rollouts & release notes',
      'Zero-downtime database migrations'
    ]
  },
  {
    step: '05',
    title: 'Observe, Profile & Iterate',
    tagline: 'Monitor telemetry, error rates, and memory allocations in production.',
    description:
      'Deployment is the start of observation. I inspect query performance, cache hit ratios, token generation costs, and user analytics to eliminate bottlenecks and continuously improve the system.',
    icon: LineChart,
    points: [
      'Sub-second query tuning and index optimization',
      'KV cache memory profiling for LLM inference',
      'Continuous user feedback loops'
    ]
  }
];

export const HowIBuild: React.FC = () => {
  return (
    <section id="philosophy" className="border-b border-border/60 bg-paper">
      <div className="section-container">
        {/* Section Header */}
        <div className="mb-12">
          <div className="section-heading">
            <span className="w-2 h-2 rounded-full bg-ink" />
            <span>Chapter 04 — Engineering Philosophy</span>
          </div>
          <h2 className="section-title">
            How I Build: Systems Thinking & Execution
          </h2>
          <p className="section-desc">
            A disciplined, first-principles approach to software engineering that scales from conceptual mathematical modeling to resilient production deployment.
          </p>
        </div>

        {/* 5 Principles Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {principles.map((item, index) => {
            const Icon = item.icon;
            const isFullWidthOnLg = index === 4; // Fifth item can span or remain cleanly styled

            return (
              <div
                key={item.step}
                className={`editorial-card p-6 sm:p-7 flex flex-col justify-between space-y-6 ${
                  isFullWidthOnLg ? 'md:col-span-2 lg:col-span-1' : ''
                }`}
              >
                <div className="space-y-3.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-ink-faint px-2 py-0.5 rounded bg-paper-100 border border-border-subtle">
                      PHASE {item.step}
                    </span>
                    <Icon className="w-4 h-4 text-ink-muted" />
                  </div>

                  <h3 className="text-lg font-bold text-ink font-editorial leading-snug">
                    {item.title}
                  </h3>

                  <p className="text-xs font-medium text-ink-faint italic">
                    "{item.tagline}"
                  </p>

                  <p className="text-xs sm:text-sm text-ink-muted leading-relaxed">
                    {item.description}
                  </p>
                </div>

                <div className="pt-4 border-t border-border-subtle space-y-1.5">
                  {item.points.map((pt, i) => (
                    <div key={i} className="text-xs text-ink-light flex items-start gap-2">
                      <span className="text-ink-faint font-mono">→</span>
                      <span>{pt}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
export default HowIBuild;

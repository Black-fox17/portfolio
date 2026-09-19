import React from 'react';
import { leadershipData } from '../../data/leadership';
import { Users, GraduationCap, Mic, Award } from 'lucide-react';

const icons = [Users, GraduationCap, Award, Mic];

export const Leadership: React.FC = () => {
  return (
    <section id="leadership" className="border-b border-border/60 bg-paper-50">
      <div className="section-container">
        {/* Section Header */}
        <div className="mb-12">
          <div className="section-heading">
            <span className="w-2 h-2 rounded-full bg-ink" />
            <span>Chapter 07 — Leadership, Mentorship & Community</span>
          </div>
          <h2 className="section-title">
            Mentoring, Technical Leadership & Speaking
          </h2>
          <p className="section-desc">
            Technical leadership is rooted in clear communication and multiplying engineering talent. Contributing through AI cohorts, university labs, and conference workshops.
          </p>
        </div>

        {/* 4-Item Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {leadershipData.map((item, idx) => {
            const Icon = icons[idx] || Users;

            return (
              <div
                key={item.organization}
                className="editorial-card p-6 flex flex-col justify-between space-y-4 bg-white"
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-ink-faint">
                    <span className="text-2xl font-bold font-editorial text-ink">
                      {item.metric}
                    </span>
                    <Icon className="w-4 h-4 text-ink-muted" />
                  </div>

                  <h3 className="text-xs font-mono font-semibold uppercase text-ink-faint tracking-wider">
                    {item.label}
                  </h3>

                  <div className="space-y-1">
                    <p className="text-xs font-semibold text-ink">
                      {item.organization}
                    </p>
                    <p className="text-xs text-ink-muted leading-relaxed">
                      {item.description}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
export default Leadership;

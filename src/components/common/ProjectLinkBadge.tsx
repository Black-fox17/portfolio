import React from 'react';
import { ArrowUpRight, Globe, Github, BookOpen } from 'lucide-react';
import { ProjectLink } from '../../types/portfolio';
import { AppleIcon, AndroidIcon } from './BrandIcons';

interface ProjectLinkBadgeProps {
  link: ProjectLink;
  variant?: 'light' | 'dark';
  className?: string;
}

export const ProjectLinkBadge: React.FC<ProjectLinkBadgeProps> = ({
  link,
  variant = 'light',
  className = '',
}) => {
  const isAppStore = link.type === 'appstore' || link.url.includes('apple.com');
  const isPlayStore =
    link.type === 'playstore' ||
    link.type === 'android' ||
    link.url.includes('play.google.com');
  const isGithub = link.type === 'github' || link.url.includes('github.com');
  const isArticle = link.type === 'article' || link.url.startsWith('/blog');

  const getTitle = () => {
    if (isAppStore) return `${link.label} on Apple App Store (iOS)`;
    if (isPlayStore) return `${link.label} on Google Play Store (Android)`;
    if (isGithub) return `${link.label} on GitHub`;
    if (isArticle) return `Read ${link.label}`;
    return `Open ${link.label}`;
  };

  const renderIcon = () => {
    if (isAppStore) {
      return (
        <AppleIcon
          className={`w-3.5 h-3.5 shrink-0 transition-transform group-hover/link:scale-110 ${
            variant === 'dark' ? 'text-paper-50' : 'text-ink'
          }`}
        />
      );
    }
    if (isPlayStore) {
      return (
        <AndroidIcon
          className="w-3.5 h-3.5 shrink-0 text-[#3DDC84] transition-transform group-hover/link:scale-110"
        />
      );
    }
    if (isGithub) {
      return (
        <Github
          className={`w-3.5 h-3.5 shrink-0 ${
            variant === 'dark' ? 'text-paper-50' : 'text-ink-muted'
          }`}
        />
      );
    }
    if (isArticle) {
      return (
        <BookOpen
          className={`w-3.5 h-3.5 shrink-0 ${
            variant === 'dark' ? 'text-paper-50' : 'text-accent-cobalt'
          }`}
        />
      );
    }
    return (
      <Globe
        className={`w-3.5 h-3.5 shrink-0 ${
          variant === 'dark' ? 'text-paper-50' : 'text-accent-cobalt'
        }`}
      />
    );
  };

  const baseClasses =
    variant === 'dark'
      ? 'inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-ink text-paper-50 hover:bg-ink-light text-xs font-mono font-medium transition-all shadow-subtle group/link'
      : 'inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-paper-100 hover:bg-paper-200 text-ink text-xs font-mono font-medium border border-border transition-all hover:border-border-strong group/link';

  const isExternal = link.url.startsWith('http');

  return (
    <a
      href={link.url}
      target={isExternal ? '_blank' : '_self'}
      rel={isExternal ? 'noopener noreferrer' : undefined}
      title={getTitle()}
      className={`${baseClasses} ${className}`}
    >
      {renderIcon()}
      <span className="tracking-tight">{link.label}</span>
      <ArrowUpRight
        className={`w-3 h-3 transition-transform group-hover/link:translate-x-0.5 group-hover/link:-translate-y-0.5 ${
          variant === 'dark' ? 'text-paper-50/70' : 'text-ink-faint'
        }`}
      />
    </a>
  );
};

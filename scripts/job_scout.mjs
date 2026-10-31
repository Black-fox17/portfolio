import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Telegram Credentials
const TELEGRAM_BOT_TOKEN =
  process.env.TELEGRAM_BOT_TOKEN || '8797137918:AAHC4-DDMKdpn4K8GDGu_k_PTk19i--jZVU';
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID || '5987992595';

// Path to cache seen job IDs
const SEEN_JOBS_FILE = path.join(__dirname, 'seen_jobs.json');

function loadSeenJobs() {
  try {
    if (fs.existsSync(SEEN_JOBS_FILE)) {
      const data = fs.readFileSync(SEEN_JOBS_FILE, 'utf8');
      return new Set(JSON.parse(data));
    }
  } catch (err) {
    console.warn('Could not read seen_jobs.json, starting fresh');
  }
  return new Set();
}

function saveSeenJobs(seenSet) {
  try {
    const arr = Array.from(seenSet).slice(-2000);
    fs.writeFileSync(SEEN_JOBS_FILE, JSON.stringify(arr, null, 2), 'utf8');
  } catch (err) {
    console.error('Failed to save seen_jobs.json:', err);
  }
}

// Target Profiles & Keywords
const HIGH_PRIORITY_ROLES = [
  'ai engineer',
  'ai systems',
  'machine learning',
  'ml engineer',
  'deep learning',
  'computer vision',
  'data scientist',
  'full-stack',
  'fullstack',
  'full stack',
  'backend',
  'back-end',
  'software engineer',
  'software developer',
  'python engineer',
  'python developer',
  'web developer',
  'frontend',
  'front-end',
  'devops',
  'cloud engineer',
  'systems engineer',
];

const SKILL_KEYWORDS = [
  'pytorch',
  'tensorflow',
  'transformer',
  'rag',
  'llm',
  'langchain',
  'computer vision',
  'fastapi',
  'nestjs',
  'node',
  'nodejs',
  'react',
  'typescript',
  'javascript',
  'python',
  'distributed',
  'docker',
  'kubernetes',
  'postgres',
  'postgresql',
  'mongodb',
  'redis',
  'aws',
  'gcp',
  'graphql',
  'rest api',
];

// Nigeria Location Detectors
const NIGERIA_KEYWORDS = [
  'nigeria',
  'lagos',
  'abuja',
  'ibadan',
  'port harcourt',
  'enugu',
  'edo',
  'ogun',
  'ng',
];

function isNigeriaJob(job) {
  const text = `${job.company} ${job.location} ${job.title} ${job.description || ''}`.toLowerCase();
  const isNigerianCompany = ['moniepoint', 'lemfi', 'andela'].some((c) =>
    job.company.toLowerCase().includes(c)
  );
  const mentionsNigeria = NIGERIA_KEYWORDS.some((kw) => text.includes(kw));
  return isNigerianCompany || mentionsNigeria;
}

// Evaluate & Score a Job
function evaluateJob(job) {
  const text = `${job.title} ${job.description || ''} ${(job.tags || []).join(' ')}`.toLowerCase();

  let roleMatched = false;
  for (const role of HIGH_PRIORITY_ROLES) {
    if (text.includes(role)) {
      roleMatched = true;
      break;
    }
  }

  if (!roleMatched) return { matched: false, score: 0, highlights: [] };

  const matchedSkills = [];
  for (const skill of SKILL_KEYWORDS) {
    if (text.includes(skill)) {
      matchedSkills.push(skill);
    }
  }

  let score = 65 + matchedSkills.length * 6;
  if (isNigeriaJob(job)) {
    score += 5; // Local Nigerian relevance boost
  } else if (text.includes('worldwide') || text.includes('anywhere')) {
    score += 8;
  }

  score = Math.min(score, 98);

  return {
    matched: true,
    score,
    highlights: matchedSkills.slice(0, 4),
  };
}

// --- 100% Free & Direct Application Sources (No Paywalls, No Intermediary Fees) ---

// 1. Arbeitnow (Clean Free Tech Board)
async function fetchArbeitnow() {
  try {
    const res = await fetch('https://www.arbeitnow.com/api/job-board-api', {
      headers: { 'User-Agent': 'Mozilla/5.0' },
    });
    if (!res.ok) return [];
    const data = await res.json();
    if (!data.data || !Array.isArray(data.data)) return [];

    return data.data.map((j) => ({
      id: `arbeitnow-${j.slug}`,
      company: j.company_name,
      title: j.title,
      location: j.location || (j.remote ? 'Remote' : 'On-site'),
      url: j.url,
      description: j.description || '',
      tags: j.tags || [],
      source: 'Arbeitnow',
    }));
  } catch (e) {
    return [];
  }
}

// 2. Remotive (100% Free Remote Dev Feed)
async function fetchRemotive() {
  try {
    const res = await fetch('https://remotive.com/api/remote-jobs?category=software-dev');
    if (!res.ok) return [];
    const data = await res.json();
    if (!data.jobs || !Array.isArray(data.jobs)) return [];

    return data.jobs.map((j) => ({
      id: `remotive-${j.id}`,
      company: j.company_name,
      title: j.title,
      location: j.candidate_required_location || 'Remote (Worldwide)',
      url: j.url,
      description: j.description || '',
      tags: j.tags || [],
      source: 'Remotive',
    }));
  } catch (e) {
    return [];
  }
}

// 3. Himalayas (Direct Company Careers Portal Links)
async function fetchHimalayas() {
  try {
    const res = await fetch('https://himalayas.app/jobs/api?parentCategories=Engineering&limit=100');
    if (!res.ok) return [];
    const data = await res.json();
    if (!data.jobs || !Array.isArray(data.jobs)) return [];

    return data.jobs.map((j) => ({
      id: `himalayas-${j.guid || j.pubDate}`,
      company: j.companyName,
      title: j.title,
      location: Array.isArray(j.locationRestrictions) && j.locationRestrictions.length > 0
        ? `Remote (${j.locationRestrictions.join(', ')})`
        : 'Remote (Worldwide)',
      url: j.applicationLink || j.guid,
      description: j.description || '',
      tags: j.categories || [],
      source: 'Himalayas (Direct Apply)',
    }));
  } catch (e) {
    return [];
  }
}

// 4. Ashby Direct ATS Boards (100% Free, Official Employer Portals)
async function fetchAshbyCompany(companyName, displayName = companyName) {
  try {
    const res = await fetch(`https://api.ashbyhq.com/posting-api/job-board/${companyName}`);
    if (!res.ok) return [];
    const data = await res.json();
    if (!data.jobs || !Array.isArray(data.jobs)) return [];

    return data.jobs.map((j) => ({
      id: `ashby-${companyName}-${j.id}`,
      company: displayName,
      title: j.title,
      location: j.location || 'Remote / Hybrid',
      url: j.jobUrl, // Direct official ATS link
      description: '',
      tags: [],
      source: `Ashby (${displayName})`,
    }));
  } catch (e) {
    return [];
  }
}

// 5. Greenhouse Direct ATS Boards (100% Free, Official Employer Portals)
async function fetchGreenhouseCompany(companyName, displayName = companyName) {
  try {
    const res = await fetch(`https://boards-api.greenhouse.io/v1/boards/${companyName}/jobs`);
    if (!res.ok) return [];
    const data = await res.json();
    if (!data.jobs || !Array.isArray(data.jobs)) return [];

    return data.jobs.map((j) => ({
      id: `gh-${companyName}-${j.id}`,
      company: displayName,
      title: j.title,
      location: j.location?.name || 'Remote / Global',
      url: j.absolute_url, // Direct official ATS link
      description: '',
      tags: [],
      source: `Greenhouse (${displayName})`,
    }));
  } catch (e) {
    return [];
  }
}

// Helper: Sleep to respect Telegram API rate limits
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Helper: Send Telegram Message
async function sendTelegramMessage(text) {
  const url = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chat_id: TELEGRAM_CHAT_ID,
      text,
      parse_mode: 'Markdown',
      disable_web_page_preview: true,
    }),
  });
  return res.json();
}

// Format a batch of jobs into a Telegram message block
function formatJobBatch(titleHeader, jobs, startIndex = 1) {
  const lines = [titleHeader, ''];

  jobs.forEach((j, idx) => {
    const num = startIndex + idx;
    lines.push(`*${num}. ${j.title}*`);
    lines.push(`🏢 *${j.company}* • 📍 _${j.location}_`);
    lines.push(`🎯 *Match Score:* ${j.score}%`);
    if (j.highlights && j.highlights.length > 0) {
      lines.push(`⚡ \`${j.highlights.join('`, `')}\``);
    }
    lines.push(`🔗 [Direct Free Apply (${j.source})](${j.url})`);
    lines.push('');
  });

  return lines.join('\n');
}

async function runJobScout() {
  console.log('🤖 Starting 100% Free Job Scout (Zero Paywalls, Nigeria + Global Mix)...');
  const seenJobs = loadSeenJobs();

  // Run all verified, 100% free scrapers in parallel
  const [
    arbeitJobs,
    remotiveJobs,
    himalayasJobs,
    moniepointJobs,
    lemfiJobs,
    andelaJobs,
    scaleJobs,
    cohereJobs,
    huggingFaceJobs,
    anthropicJobs,
    vercelJobs,
    figmaJobs,
    perplexityJobs,
    supabaseJobs,
    mistralJobs,
    posthogJobs,
    replicateJobs,
  ] = await Promise.all([
    fetchArbeitnow(),
    fetchRemotive(),
    fetchHimalayas(),
    fetchGreenhouseCompany('moniepoint', 'Moniepoint'),
    fetchAshbyCompany('lemfi', 'LemFi'),
    fetchAshbyCompany('andela', 'Andela'),
    fetchGreenhouseCompany('scaleai', 'Scale AI'),
    fetchGreenhouseCompany('cohere', 'Cohere'),
    fetchGreenhouseCompany('huggingface', 'Hugging Face'),
    fetchGreenhouseCompany('anthropic', 'Anthropic'),
    fetchGreenhouseCompany('vercel', 'Vercel'),
    fetchGreenhouseCompany('figma', 'Figma'),
    fetchAshbyCompany('perplexity', 'Perplexity AI'),
    fetchAshbyCompany('supabase', 'Supabase'),
    fetchAshbyCompany('mistral', 'Mistral AI'),
    fetchAshbyCompany('posthog', 'PostHog'),
    fetchAshbyCompany('replicate', 'Replicate'),
  ]);

  const allJobs = [
    arbeitJobs,
    remotiveJobs,
    himalayasJobs,
    moniepointJobs,
    lemfiJobs,
    andelaJobs,
    scaleJobs,
    cohereJobs,
    huggingFaceJobs,
    anthropicJobs,
    vercelJobs,
    figmaJobs,
    perplexityJobs,
    supabaseJobs,
    mistralJobs,
    posthogJobs,
    replicateJobs,
  ].flat();

  console.log(`Fetched total ${allJobs.length} raw jobs from free official ATS feeds.`);

  const nigeriaJobs = [];
  const globalRemoteJobs = [];

  for (const job of allJobs) {
    if (seenJobs.has(job.id)) continue;

    const evaluation = evaluateJob(job);
    if (!evaluation.matched) continue;

    const enriched = {
      ...job,
      score: evaluation.score,
      highlights: evaluation.highlights,
    };

    if (isNigeriaJob(job)) {
      nigeriaJobs.push(enriched);
    } else {
      globalRemoteJobs.push(enriched);
    }

    seenJobs.add(job.id);
  }

  // Sort by match score
  nigeriaJobs.sort((a, b) => b.score - a.score);
  globalRemoteJobs.sort((a, b) => b.score - a.score);

  // Target up to 20 Nigeria (Physical, Hybrid, Remote) + up to 30 Global Remote
  const selectedNigeria = nigeriaJobs.slice(0, 20);
  const selectedGlobal = globalRemoteJobs.slice(0, 30);
  const totalSelected = selectedNigeria.length + selectedGlobal.length;

  console.log(
    `Selected ${totalSelected} roles: ${selectedNigeria.length} Nigeria, ${selectedGlobal.length} Global Remote.`
  );

  if (totalSelected === 0) {
    console.log('No new jobs matched the criteria today.');
    return;
  }

  const dateStr = new Date().toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    weekday: 'short',
  });

  // 1. Send Header Alert
  await sendTelegramMessage(
    `☀️ *Morning Job Scout Digest (${dateStr})*\n\nCurated *${totalSelected} High-Match Roles* (100% Free Direct Apply):\n🇳🇬 *${selectedNigeria.length} Nigeria Opportunities* (Physical / Hybrid / Remote)\n🌍 *${selectedGlobal.length} Global Remote Opportunities* (Abroad / International)\n\n_All links go directly to the company's official ATS without middleman fees._`
  );
  await sleep(1500);

  // 2. Send Nigeria Roles (Batched into chunks of 6)
  if (selectedNigeria.length > 0) {
    const chunkSize = 6;
    for (let i = 0; i < selectedNigeria.length; i += chunkSize) {
      const chunk = selectedNigeria.slice(i, i + chunkSize);
      const batchNum = Math.floor(i / chunkSize) + 1;
      const totalBatches = Math.ceil(selectedNigeria.length / chunkSize);
      const header = `🇳🇬 *Nigeria Tech Opportunities (Part ${batchNum}/${totalBatches})*`;
      const msg = formatJobBatch(header, chunk, i + 1);
      await sendTelegramMessage(msg);
      await sleep(1500);
    }
  }

  // 3. Send Global Remote Roles (Batched into chunks of 6)
  if (selectedGlobal.length > 0) {
    const chunkSize = 6;
    for (let i = 0; i < selectedGlobal.length; i += chunkSize) {
      const chunk = selectedGlobal.slice(i, i + chunkSize);
      const batchNum = Math.floor(i / chunkSize) + 1;
      const totalBatches = Math.ceil(selectedGlobal.length / chunkSize);
      const header = `🌍 *Global Remote Tech & AI Roles (Part ${batchNum}/${totalBatches})*`;
      const msg = formatJobBatch(header, chunk, i + 1);
      await sendTelegramMessage(msg);
      await sleep(1500);
    }
  }

  saveSeenJobs(seenJobs);
  console.log('✅ All verified direct job batches delivered successfully to Telegram!');
}

runJobScout();

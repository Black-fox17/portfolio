export default async function handler(req, res) {
  // CORS configuration
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
  );

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const body = req.body || {};
    const {
      pathname = '/',
      referrer = '',
      screen = '',
      language = '',
      isNewSession = true,
    } = body;

    const userAgent = req.headers['user-agent'] || '';

    // Ignore web crawlers & health check bots from flooding Telegram
    const isBot = /bot|spider|crawl|slurp|lighthouse|preview|headless/i.test(userAgent);
    if (isBot) {
      return res.status(200).json({ status: 'ignored_bot' });
    }

    // Vercel Geolocation Headers (Provided automatically by Vercel Edge/Serverless)
    const city = req.headers['x-vercel-ip-city']
      ? decodeURIComponent(req.headers['x-vercel-ip-city'])
      : null;
    const country = req.headers['x-vercel-ip-country'] || null;
    const region = req.headers['x-vercel-ip-country-region'] || null;

    // IP Extraction
    const rawIp = req.headers['x-forwarded-for'] || req.socket?.remoteAddress || '';
    const clientIp = rawIp ? rawIp.split(',')[0].trim() : '';

    // Parse Device
    let device = 'Desktop';
    if (/mobile/i.test(userAgent)) device = 'Mobile';
    if (/tablet|ipad/i.test(userAgent)) device = 'Tablet';
    if (/iphone/i.test(userAgent)) device = 'iPhone';
    if (/android/i.test(userAgent)) device = 'Android';

    if (/macintosh|mac os x/i.test(userAgent)) device += ' (macOS)';
    else if (/windows/i.test(userAgent)) device += ' (Windows)';
    else if (/linux/i.test(userAgent)) device += ' (Linux)';

    // Parse Browser
    let browser = 'Browser';
    if (/edg\//i.test(userAgent)) browser = 'Edge';
    else if (/chrome|crios/i.test(userAgent)) browser = 'Chrome';
    else if (/firefox|fxios/i.test(userAgent)) browser = 'Firefox';
    else if (/safari/i.test(userAgent)) browser = 'Safari';

    // Format Location
    const locationParts = [city, region, country].filter(Boolean);
    const locationStr = locationParts.length > 0 ? locationParts.join(', ') : 'Unknown Location';

    // Format Referrer
    const refStr = referrer && referrer.trim() !== '' ? referrer : 'Direct / Bookmark';

    const timestamp = new Date().toLocaleString('en-US', {
      timeZone: 'UTC',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    });

    const header = isNewSession
      ? '🚀 *New Visitor on ayeleru.space!*'
      : '📄 *Page View on ayeleru.space*';

    const message = [
      header,
      '',
      `📍 *Location:* ${locationStr}`,
      `📄 *Page:* \`${pathname}\``,
      `🔗 *Referrer:* ${refStr}`,
      `📱 *Device:* ${device} • ${browser}`,
      screen ? `📐 *Screen:* ${screen}` : null,
      language ? `🌐 *Language:* ${language}` : null,
      clientIp ? `🛰️ *IP:* \`${clientIp}\`` : null,
      `⏰ *Time (UTC):* ${timestamp}`,
    ]
      .filter(Boolean)
      .join('\n');

    const botToken =
      process.env.TELEGRAM_BOT_TOKEN || '8797137918:AAHC4-DDMKdpn4K8GDGu_k_PTk19i--jZVU';
    const chatId = process.env.TELEGRAM_CHAT_ID || '5987992595';

    await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text: message,
        parse_mode: 'Markdown',
        disable_web_page_preview: true,
      }),
    });

    return res.status(200).json({ success: true });
  } catch (err) {
    console.error('Error sending Telegram alert:', err);
    return res.status(500).json({ error: 'Failed to dispatch alert' });
  }
}

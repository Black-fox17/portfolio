export const trackPageView = async (pathname: string = window.location.pathname) => {
  // Do not send alerts when browsing on localhost in development
  if (
    typeof window === 'undefined' ||
    window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1'
  ) {
    return;
  }

  try {
    const sessionKey = 'ayeleru_session_tracked';
    const isNewSession = !sessionStorage.getItem(sessionKey);

    const lastPageTracked = sessionStorage.getItem('ayeleru_last_page');
    if (lastPageTracked === pathname) {
      return;
    }

    sessionStorage.setItem(sessionKey, 'true');
    sessionStorage.setItem('ayeleru_last_page', pathname);

    const payload = {
      pathname,
      referrer: document.referrer || '',
      screen: `${window.screen.width}x${window.screen.height}`,
      language: navigator.language || '',
      isNewSession,
    };

    // Dispatch to Vercel Serverless Function (Telegram Bot)
    fetch('/api/track', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    }).catch((err) => {
      console.warn('Telegram tracking ping failed:', err);
    });
  } catch (error) {
    // Fail silently so visitors never experience latency
  }
};

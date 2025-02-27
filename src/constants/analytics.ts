// analytics.js
export const trackPageView = async () => {
    try {
      await fetch('https://portfolio-wahz.onrender.com/', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    } catch (error) {
      console.error('Failed to track visit:', error);
    }
  };

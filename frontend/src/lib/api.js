const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  'https://vivo777-astram-api.hf.space';

async function fetchWithTimeout(url, options = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(timer);
  }
}

async function fetchApi(path, options = {}, retries = 2) {
  const url = `${API_BASE}${path}`;
  let lastError;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetchWithTimeout(
        url,
        {
          headers: { 'Content-Type': 'application/json', ...options.headers },
          ...options,
        },
        attempt === 0 ? 120000 : 180000
      );

      if (res.status === 503 && attempt < retries) {
        // HF Space is waking up — wait and retry
        await new Promise((r) => setTimeout(r, 5000 * (attempt + 1)));
        continue;
      }

      if (!res.ok) {
        const error = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(error.detail || `API error: ${res.status}`);
      }
      return res.json();
    } catch (err) {
      lastError = err;
      if (err.name === 'AbortError') {
        lastError = new Error(
          'Request timed out. The backend may be starting up — please try again in 30 seconds.'
        );
      }
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, 5000 * (attempt + 1)));
        continue;
      }
    }
  }
  throw lastError;
}

export async function predict(eventData) {
  return fetchApi('/predict', {
    method: 'POST',
    body: JSON.stringify(eventData),
  });
}

export async function sendChatMessage(message) {
  return fetchApi('/chatbot', {
    method: 'POST',
    body: JSON.stringify({ message }),
  });
}

export async function getHotspots() {
  return fetchApi('/hotspots');
}

export async function getAnalytics() {
  return fetchApi('/analytics');
}

export async function getMetrics() {
  return fetchApi('/metrics');
}

export async function getHealth() {
  return fetchApi('/health');
}

export async function getDrift() {
  return fetchApi('/drift');
}

export async function getMonitoringPredictions() {
  return fetchApi('/monitoring/predictions');
}

export async function getRetrainStatus() {
  return fetchApi('/monitoring/retrain-status');
}


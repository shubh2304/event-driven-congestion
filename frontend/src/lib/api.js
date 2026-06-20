const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchApi(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }
  return res.json();
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

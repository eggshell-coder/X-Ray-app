const STORAGE_KEY = 'CXR_GNN_API_BASE';

export function getApiBase() {
  const custom = localStorage.getItem(STORAGE_KEY);
  if (custom && custom.trim() !== '') {
    return custom.trim().replace(/\/+$/, '');
  }
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/+$/, '');
  }
  return '';
}

export function setApiBase(url) {
  if (!url || url.trim() === '') {
    localStorage.removeItem(STORAGE_KEY);
  } else {
    localStorage.setItem(STORAGE_KEY, url.trim().replace(/\/+$/, ''));
  }
}

export function getEndpoint(path) {
  const base = getApiBase();
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return base ? `${base}${cleanPath}` : cleanPath;
}

export function getTunnelHeaders() {
  return {
    'Bypass-Tunnel-Remainder': 'true',
    'ngrok-skip-browser-warning': 'true',
  };
}

const STORAGE_KEY = 'CXR_GNN_API_BASE';

export function getApiBase() {
  const custom = typeof window !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
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
  // Tunnel-only headers create an unnecessary CORS preflight on Railway/Render.
  const base = getApiBase().toLowerCase();
  if (!base.includes('ngrok') && !base.includes('loca.lt')) return {};
  return {
    'Bypass-Tunnel-Remainder': 'true',
    'ngrok-skip-browser-warning': 'true',
  };
}

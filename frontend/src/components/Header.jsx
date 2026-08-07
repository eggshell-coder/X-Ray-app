import React, { useEffect, useState } from 'react';
import { Activity, Network } from 'lucide-react';

export default function Header() {
  const [health, setHealth] = useState({ status: 'checking', device: '', classes: [] });

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.status === 'ok') {
          setHealth({ status: 'ok', device: data.device, classes: data.classes });
        } else {
          setHealth({ status: 'bad', device: '', classes: [] });
        }
      } catch (err) {
        setHealth({ status: 'bad', device: '', classes: [] });
      }
    }
    checkHealth();
  }, []);

  return (
    <header>
      <div className="brand">
        <div className="brand-mark">
          <Network size={22} color="#38BDF8" />
        </div>
        <div className="brand-text">
          <h1>CXR·GNN</h1>
          <p>graph-attention chest X-ray classifier &amp; explainable AI</p>
        </div>
      </div>

      <div className="status-pill">
        <span className={`status-dot ${health.status === 'ok' ? 'ok' : 'bad'}`} />
        <span>
          {health.status === 'ok'
            ? `model ready · ${health.device}`
            : health.status === 'checking'
            ? 'checking backend…'
            : 'backend unreachable'}
        </span>
      </div>
    </header>
  );
}

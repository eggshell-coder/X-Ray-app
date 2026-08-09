import React, { useEffect, useState } from 'react';
import { Network, Server, Settings, Check, X } from 'lucide-react';
import { getApiBase, setApiBase, getEndpoint, getTunnelHeaders } from '../apiConfig';

export default function Header() {
  const [health, setHealth] = useState({ status: 'checking', device: '', classes: [] });
  const [showConfig, setShowConfig] = useState(false);
  const [inputUrl, setInputUrl] = useState('');

  const checkHealth = async () => {
    setHealth({ status: 'checking', device: '', classes: [] });
    try {
      const res = await fetch(getEndpoint('/api/health'), {
        headers: getTunnelHeaders(),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setHealth({ status: 'ok', device: data.device, classes: data.classes });
      } else {
        setHealth({ status: 'bad', device: '', classes: [] });
      }
    } catch (err) {
      setHealth({ status: 'bad', device: '', classes: [] });
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const handleOpenConfig = () => {
    setInputUrl(getApiBase());
    setShowConfig(!showConfig);
  };

  const handleSaveConfig = () => {
    setApiBase(inputUrl);
    setShowConfig(false);
    checkHealth();
  };

  const activeBase = getApiBase();

  return (
    <header>
      <div className="brand">
        <div className="brand-mark">
          <Network size={22} color="#38BDF8" />
        </div>
        <div className="brand-text">
          <h1>Adaptive CNN–GATv2 Fusion</h1>
          <p>Uncertainty-Aware Conformal Prediction</p>
        </div>
      </div>

      <div className="header-actions">
        <div className="status-pill">
          <span className={`status-dot ${health.status === 'ok' ? 'ok' : 'bad'}`} />
          <span>
            {health.status === 'ok'
              ? `model ready · ${health.device}${activeBase ? ' (tunnel)' : ''}`
              : health.status === 'checking'
              ? 'checking backend…'
              : 'backend unreachable'}
          </span>
          <button className="btn-icon" onClick={handleOpenConfig} title="Configure Server / Tunnel URL">
            <Settings size={13} />
          </button>
        </div>
      </div>

      {showConfig && (
        <div className="config-modal-backdrop" onClick={() => setShowConfig(false)}>
          <div className="config-modal" onClick={(e) => e.stopPropagation()}>
            <div className="config-header">
              <Server size={16} color="var(--cyan)" />
              <span>Backend Server / Tunnel URL</span>
              <button className="btn-close" onClick={() => setShowConfig(false)}>
                <X size={14} />
              </button>
            </div>
            <p className="config-desc">
              Enter your local backend Tunnel URL (e.g. <code>https://xxxx.ngrok-free.app</code> or <code>https://xxxx.loca.lt</code>) if running PyTorch locally. Leave blank for default server.
            </p>
            <input
              type="text"
              className="config-input"
              placeholder="https://xxxx.ngrok-free.app (or leave empty)"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
            />
            <div className="config-btns">
              <button className="btn-primary-sm" onClick={handleSaveConfig}>
                <Check size={14} />
                Save &amp; Test Connection
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}

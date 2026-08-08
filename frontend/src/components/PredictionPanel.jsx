import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert } from 'lucide-react';
import ConfidenceGauge from './ConfidenceGauge';

export default function PredictionPanel({ results }) {
  if (!results) {
    return (
      <div className="empty-results">
        <div>Awaiting an image.</div>
        <div style={{ fontSize: '11px', color: 'var(--muted)' }}>
          Results, GNN explainability heatmaps<br />and per-class confidence will appear here.
        </div>
      </div>
    );
  }

  if (results.status === 'rejected') {
    return (
      <div className="prediction-results ood-rejected-card">
        <div className="ood-header">
          <ShieldAlert size={32} color="var(--red)" />
          <div>
            <div className="ood-title">Non-medical image</div>
            <div className="ood-subtitle">Please upload a chest X-ray image.</div>
          </div>
        </div>

        <div className="ood-detail-box">
          This is a non-medical image, not a chest X-ray.
        </div>

        <div className="disclaimer" style={{ marginTop: 24 }}>
          No disease prediction was generated.
        </div>
      </div>
    );
  }

  const top = results.probabilities[0];
  const confPct = (results.confidence * 100).toFixed(1);
  const isHighConf = results.certainty_status === 'High Confidence';

  return (
    <div className="prediction-results">
      <div className="result-top">
        <div className="label">{top.label}</div>
        <div className="conf">{confPct}%</div>
      </div>

      <ConfidenceGauge value={results.confidence * 100} />

      <div className="badge-row">
        <span className="badge">{results.n_superpixels} regions analyzed</span>
        <span className="badge">GATv2 Architecture</span>
        <span className={`badge ${isHighConf ? 'status-ok' : 'status-warn'}`}>
          {isHighConf ? (
            <ShieldCheck size={12} style={{ display: 'inline', marginRight: 4 }} />
          ) : (
            <AlertTriangle size={12} style={{ display: 'inline', marginRight: 4 }} />
          )}
          {results.certainty_status || 'Analyzed'}
        </span>
      </div>

      {results.probabilities.map((p, i) => {
        const pct = (p.probability * 100).toFixed(1);
        return (
          <div className="bar-row" key={p.label}>
            <div className="bar-head">
              <span className="bname">{p.label}</span>
              <span className="bval">{pct}%</span>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${i === 0 ? 'top' : ''}`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}

      <div className="disclaimer">
        Model output only — not a clinical diagnosis. Always confirm findings with a qualified radiologist.
      </div>
    </div>
  );
}

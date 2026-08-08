import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert, ImageOff } from 'lucide-react';
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
    const reasonLabels = {
      multi_colored_photo: 'Multi-Colored Non-Medical Photo',
      non_xray_histogram: 'Non-Medical Histogram Profile',
      synthetic_edge_pattern: 'Synthetic Vector / Text Document',
      invalid_aspect_ratio: 'Invalid Image Aspect Ratio',
      degenerate_slic: 'Superpixel Segmentation Failed',
      feature_ood: 'Out-Of-Distribution (OOD) Features',
      recognised_non_medical_image: 'Recognized Non-Medical Image',
      untrusted_raster_source: 'Unverified Raster Image',
      unverified_dicom_metadata: 'Unverified DICOM Metadata',
      validator_unavailable: 'Input Validator Unavailable',
    };

    return (
      <div className="prediction-results ood-rejected-card">
        <div className="ood-header">
          <ShieldAlert size={32} color="var(--red)" />
          <div>
            <div className="ood-title">Prediction Withheld</div>
            <div className="ood-subtitle">Non-Chest X-Ray / Out-Of-Distribution Detected</div>
          </div>
        </div>

        <div className="badge-row" style={{ marginTop: 16 }}>
          <span className="badge status-warn">
            <ImageOff size={12} style={{ display: 'inline', marginRight: 4 }} />
            Reason: {reasonLabels[results.reason] || results.reason}
          </span>
        </div>

        <div className="ood-detail-box">
          {results.detail}
        </div>

        <div className="disclaimer" style={{ marginTop: 24 }}>
          No disease probabilities were generated. The independent chest-X-ray gate rejected this upload before the disease model could run.
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

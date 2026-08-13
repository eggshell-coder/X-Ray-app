import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert, Stethoscope, Crosshair, FileText } from 'lucide-react';
import ConfidenceGauge from './ConfidenceGauge';

function parseExplanation(text) {
  if (!text) return null;
  const get = (label) => {
    const match = text.match(new RegExp(`^${label}:\\s*(.+)$`, 'mi'));
    return match ? match[1].trim() : '';
  };
  const detected = get('Detected');
  const detectedMatch = detected.match(/^(.+?)\\s*\\(Confidence:\s*([\\d.]+)%\\)/i);
  return {
    prediction: detectedMatch ? detectedMatch[1].trim() : '',
    confidence: detectedMatch ? detectedMatch[2] : '',
    sign: get('Key sign'),
    region: get('Region to inspect'),
    why: get('Why'),
  };
}

export default function PredictionPanel({ results, onExplain, explanation, isExplaining }) {
  if (!results) {
    return (
      <div className="empty-results">
        <div>Awaiting an image.</div>
        <div style={{ fontSize: '11px', color: 'var(--muted)' }}>
          Results and per-class confidence will appear here.
        </div>
      </div>
    );
  }

  if (results.status === 'rejected') {
    const needsReview = ['feature_ood', 'degenerate_slic'].includes(results.reason);

    return (
      <div className="prediction-results ood-rejected-card">
        <div className="ood-header">
          <ShieldAlert size={32} color="var(--red)" />
          <div>
            <div className="ood-title">
              {needsReview ? 'Review needed' : 'Non-medical image'}
            </div>
            <div className="ood-subtitle">
              {needsReview
                ? 'The X-ray could not be classified confidently.'
                : 'Please upload a chest X-ray image.'}
            </div>
          </div>
        </div>

        <div className="ood-detail-box">
          {needsReview
            ? 'This image may be a chest X-ray, but the confidence is below the safe prediction threshold. Please review it or upload a clearer X-ray.'
            : 'This is a non-medical image, not a chest X-ray.'}
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
      <button className="btn-ghost" style={{ width: '100%', marginTop: 16 }} onClick={onExplain} disabled={isExplaining}>
        {isExplaining ? 'Preparing explanation…' : 'Explain this result'}
      </button>
      {explanation && (() => {
        const card = parseExplanation(explanation);
        if (!card) return null;
        return (
          <div className="ai-explanation-card">
            <div className="ai-explanation-label">AI explanation</div>
            <div className="ai-explanation-heading">
              <span>{card.prediction || results.prediction}</span>
              <span className="ai-confidence-badge">{card.confidence || (results.confidence * 100).toFixed(1)}% confidence</span>
            </div>
            <div className="ai-explanation-divider" />
            {card.sign && <div className="ai-explanation-row"><Stethoscope size={15} /><span className="ai-row-label">Key sign</span><span>{card.sign}</span></div>}
            {card.region && <div className="ai-explanation-row"><Crosshair size={15} /><span className="ai-row-label">Region</span><span>{card.region}</span></div>}
            {card.why && <div className="ai-explanation-row"><FileText size={15} /><span className="ai-row-label">Why</span><span>{card.why}</span></div>}
          </div>
        );
      })()}
    </div>
  );
}

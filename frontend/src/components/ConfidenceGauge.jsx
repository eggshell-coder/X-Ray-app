import React, { useEffect, useRef } from 'react';

export default function ConfidenceGauge({ value = 0 }) {
  const circumference = Math.PI * 90; // semicircle with r=90
  const filled = circumference * (value / 100);
  const color =
    value >= 80 ? 'var(--green)' :
    value >= 50 ? 'var(--amber)' :
    'var(--red)';

  const glowColor =
    value >= 80 ? 'rgba(52, 211, 153, 0.35)' :
    value >= 50 ? 'rgba(251, 191, 36, 0.35)' :
    'rgba(248, 113, 113, 0.35)';

  const label =
    value >= 80 ? 'High' :
    value >= 50 ? 'Moderate' :
    'Low';

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 200 120" className="gauge-svg">
        {/* Background arc */}
        <path
          d="M 10 110 A 90 90 0 0 1 190 110"
          fill="none"
          stroke="var(--panel-2)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <path
          d="M 10 110 A 90 90 0 0 1 190 110"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
          className="gauge-fill"
          style={{
            filter: `drop-shadow(0 0 6px ${glowColor})`,
            '--target-dash': `${filled} ${circumference}`,
          }}
        />
        {/* Center text */}
        <text x="100" y="95" textAnchor="middle" className="gauge-pct" fill={color}>
          {value.toFixed(1)}%
        </text>
        <text x="100" y="112" textAnchor="middle" className="gauge-label" fill="var(--muted)">
          {label} Confidence
        </text>
      </svg>
    </div>
  );
}

import React, { useState } from 'react';
import { Eye, Layers, Flame, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

export default function MedicalImageViewer({ originalUrl, visualizations }) {
  const [activeView, setActiveView] = useState('original');
  const [opacity, setOpacity] = useState(0.85);
  const [zoom, setZoom] = useState(1);

  const hasVisualizations = visualizations && visualizations.superpixels;

  const getDisplayedImage = () => {
    if (activeView === 'superpixels' && visualizations?.superpixels) {
      return visualizations.superpixels;
    }
    if (activeView === 'attention' && visualizations?.attention_heatmap) {
      return visualizations.attention_heatmap;
    }
    return originalUrl;
  };

  return (
    <div className="medical-viewer">
      {hasVisualizations && (
        <div className="viewport-toolbar">
          <div className="view-tabs">
            <button
              className={`view-btn ${activeView === 'original' ? 'active' : ''}`}
              onClick={() => setActiveView('original')}
            >
              <Eye size={12} style={{ display: 'inline', marginRight: 4 }} />
              Original
            </button>
            <button
              className={`view-btn ${activeView === 'superpixels' ? 'active' : ''}`}
              onClick={() => setActiveView('superpixels')}
            >
              <Layers size={12} style={{ display: 'inline', marginRight: 4 }} />
              Superpixel Graph
            </button>
            <button
              className={`view-btn ${activeView === 'attention' ? 'active' : ''}`}
              onClick={() => setActiveView('attention')}
            >
              <Flame size={12} style={{ display: 'inline', marginRight: 4 }} />
              GATv2 Attention
            </button>
          </div>

          {activeView === 'attention' && (
            <div className="opacity-slider-wrap">
              <span>Heatmap Opacity:</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
              />
              <span>{Math.round(opacity * 100)}%</span>
            </div>
          )}

          <div className="view-tabs">
            <button className="view-btn" onClick={() => setZoom((z) => Math.min(z + 0.25, 2.5))} title="Zoom In">
              <ZoomIn size={13} />
            </button>
            <button className="view-btn" onClick={() => setZoom((z) => Math.max(z - 0.25, 0.75))} title="Zoom Out">
              <ZoomOut size={13} />
            </button>
            <button className="view-btn" onClick={() => setZoom(1)} title="Reset Zoom">
              <RotateCcw size={13} />
            </button>
          </div>
        </div>
      )}

      <div className="preview-wrap" style={{ overflow: 'hidden' }}>
        <img
          src={getDisplayedImage()}
          alt="X-ray Viewport"
          style={{
            transform: `scale(${zoom})`,
            opacity: activeView === 'attention' ? opacity : 1,
            transition: 'transform 0.15s ease, opacity 0.15s ease',
          }}
        />
        <div className="frame-corner fc-tl" />
        <div className="frame-corner fc-tr" />
        <div className="frame-corner fc-bl" />
        <div className="frame-corner fc-br" />
      </div>
    </div>
  );
}

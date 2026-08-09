import React, { useState } from 'react';
import { Eye, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

export default function MedicalImageViewer({ originalUrl, visualizations }) {
  const [activeView, setActiveView] = useState('original');
  const [zoom, setZoom] = useState(1);

  const getDisplayedImage = () => {
    return originalUrl;
  };

  return (
    <div className="medical-viewer">
      {(
        <div className="viewport-toolbar">
          <div className="view-tabs">
            <button
              className={`view-btn ${activeView === 'original' ? 'active' : ''}`}
              onClick={() => setActiveView('original')}
            >
              <Eye size={12} style={{ display: 'inline', marginRight: 4 }} />
              Original
            </button>
          </div>

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
            transition: 'transform 0.15s ease',
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

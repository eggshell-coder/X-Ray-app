import React, { useState } from 'react';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import MedicalImageViewer from './components/MedicalImageViewer';
import PredictionPanel from './components/PredictionPanel';
import { Sparkles, Trash2 } from 'lucide-react';
import { getEndpoint, getTunnelHeaders } from './apiConfig';

const MAX_CLIENT_UPLOAD_BYTES = 1.8 * 1024 * 1024;
const MAX_CLIENT_IMAGE_DIMENSION = 2400;

function isDicomFile(file) {
  return file.type === 'application/dicom' || /\.dcm$/i.test(file.name || '');
}

async function prepareUploadFile(file) {
  // DICOM contains medical metadata and must not be converted through a
  // browser canvas. The backend validates its size and content directly.
  if (isDicomFile(file) || file.size <= MAX_CLIENT_UPLOAD_BYTES) {
    return file;
  }

  const objectUrl = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = 'async';
    image.src = objectUrl;
    await image.decode();

    const scale = Math.min(1, MAX_CLIENT_IMAGE_DIMENSION / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext('2d', { alpha: false });
    context.drawImage(image, 0, 0, canvas.width, canvas.height);

    let quality = 0.86;
    let compressed = null;
    do {
      compressed = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
      quality -= 0.08;
    } while (compressed && compressed.size > MAX_CLIENT_UPLOAD_BYTES && quality >= 0.50);

    if (!compressed || compressed.size >= file.size) {
      return file;
    }

    const filename = (file.name || 'xray').replace(/\.[^.]+$/, '') + '.jpg';
    return new File([compressed], filename, { type: 'image/jpeg', lastModified: Date.now() });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelected = (file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setError(null);

      const uploadFile = await prepareUploadFile(selectedFile);
      const formData = new FormData();
      formData.append('file', uploadFile, uploadFile.name);

    try {
      const res = await fetch(getEndpoint('/api/predict'), {
        method: 'POST',
        headers: getTunnelHeaders(),
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }

      setResults(data);
    } catch (err) {
      const isFetchError = err instanceof TypeError && /fetch/i.test(err.message || '');
      setError(
        isFetchError
          ? 'Cannot connect to the backend. Please check the deployed API URL.'
          : (err.message || 'An error occurred during analysis')
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="wrap">
      <Header />

      <div className="grid">
        {/* Left Column: Image Upload & Viewport */}
        <div className="panel">
          <p className="panel-label">
            <span>01 — Input Image &amp; Visualizer</span>
            {selectedFile && <span style={{ color: 'var(--cyan)' }}>{selectedFile.name}</span>}
          </p>

          {!results ? (
            <UploadZone
              onFileSelected={handleFileSelected}
              previewUrl={previewUrl}
              isAnalyzing={isAnalyzing}
            />
          ) : (
            <MedicalImageViewer
              originalUrl={previewUrl}
              visualizations={results.visualizations}
            />
          )}

          <div className="analyze-row">
            <button
              className="btn-primary"
              onClick={handleAnalyze}
              disabled={!selectedFile || isAnalyzing}
            >
              <Sparkles size={16} />
              {isAnalyzing ? 'Analyzing…' : 'Analyze Image'}
            </button>

            {selectedFile && (
              <button className="btn-ghost" onClick={handleClear}>
                <Trash2 size={16} />
                Clear
              </button>
            )}
          </div>

          {error && (
            <div style={{ color: 'var(--red)', marginTop: 12, fontSize: 13, textAlign: 'center' }}>
              {error}
            </div>
          )}
        </div>

        {/* Right Column: Classification Results & Report */}
        <div className="panel">
          <p className="panel-label">02 — Classification Result</p>
          <PredictionPanel results={results} />
        </div>
      </div>

      <footer>
        For research &amp; educational use only — not a diagnostic device. Model output is not a substitute for clinical judgment.
      </footer>
    </div>
  );
}

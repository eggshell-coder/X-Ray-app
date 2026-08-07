import React, { useState } from 'react';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import MedicalImageViewer from './components/MedicalImageViewer';
import PredictionPanel from './components/PredictionPanel';
import { Sparkles, Trash2 } from 'lucide-react';

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

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }

      setResults(data);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis');
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
              {isAnalyzing ? 'Segmenting & Analyzing…' : 'Analyze Image'}
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

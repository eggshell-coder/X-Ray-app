import React, { useRef, useState } from 'react';
import { Upload, FileCode } from 'lucide-react';

export default function UploadZone({ onFileSelected, previewUrl, isAnalyzing }) {
  const [isDrag, setIsDrag] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDrag(true);
  };

  const handleDragLeave = () => {
    setIsDrag(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDrag(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelected(e.target.files[0]);
    }
  };

  return (
    <div
      className={`dropzone ${isDrag ? 'drag' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        accept="image/png, image/jpeg, image/bmp, image/tiff, .dcm"
        onChange={handleChange}
      />

      {!previewUrl ? (
        <div className="dz-empty">
          <Upload className="icon" size={36} />
          <div className="t1">Drop a chest X-ray, or click to browse</div>
          <div className="t2">PNG · JPG · BMP · TIFF · DICOM (.dcm), up to 25MB</div>
        </div>
      ) : (
        <div className="preview-wrap">
          <img src={previewUrl} alt="X-ray input preview" />
          <div className="frame-corner fc-tl" />
          <div className="frame-corner fc-tr" />
          <div className="frame-corner fc-bl" />
          <div className="frame-corner fc-br" />
          <div className={`scan-line ${isAnalyzing ? 'active' : ''}`} />
          <div className="change-file">click to replace</div>
        </div>
      )}
    </div>
  );
}

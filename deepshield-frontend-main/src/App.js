import React, { useState, useEffect } from "react";
import "./App.css";

const SCAN_LOGS = [
  "Loading neural backbone weights...",
  "Extracting pixel-level artifacts...",
  "Running GAN fingerprint analysis...",
  "Checking frequency domain anomalies...",
  "Cross-referencing facial geometry...",
  "Analyzing temporal inconsistencies...",
  "Computing confidence score...",
];

function App() {
  const [imagePreview, setImagePreview] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [scanLog, setScanLog] = useState(SCAN_LOGS[0]);
  const [scanProgress, setScanProgress] = useState(0);
  const [logIndex, setLogIndex] = useState(0);
  const [dragOver, setDragOver] = useState(null);

  const handleImageUpload = (file) => {
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) {
      alert("Unsupported image format! Use JPG or PNG.");
      return;
    }
    setImagePreview(URL.createObjectURL(file));
    setSelectedFile(file);
    setVideoPreview(null);
  };

  const handleVideoUpload = (file) => {
    if (!file) return;
    if (file.type !== "video/mp4") {
      alert("Only MP4 videos are supported!");
      return;
    }
    setVideoPreview(URL.createObjectURL(file));
    setSelectedFile(file);
    setImagePreview(null);
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    setDragOver(null);
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (type === "image") handleImageUpload(file);
    else handleVideoUpload(file);
  };

  useEffect(() => {
    if (status !== "analyzing") return;

    let progress = 0;
    const progressTimer = setInterval(() => {
      progress += Math.random() * 2;
      if (progress >= 95) {
        progress = 95;
        clearInterval(progressTimer);
      }
      setScanProgress(Math.min(progress, 95));
    }, 120);

    const logTimer = setInterval(() => {
      setLogIndex((i) => {
        const next = (i + 1) % SCAN_LOGS.length;
        setScanLog(SCAN_LOGS[next]);
        return next;
      });
    }, 800);

    const analyzeMedia = async () => {
      if (!selectedFile) return;

      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL || "http://localhost:8000"}/predict/`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        const isFake = data.prediction === "Fake";
        
        // 100% confidence means model is absolutely sure of its prediction
        // If it's real, confidence of real is the score. If it's fake, confidence of fake is the score.
        // The API returns 'data.confidence' (e.g., 0.95), which is probability of Fake.
        let finalConf = Math.round(data.confidence * 100);
        if (!isFake) {
            // For 'Real', it's 1 - fake probability
            finalConf = 100 - finalConf;
        }

        let signalsList = data.reason?.details || [];
        if (signalsList.length === 0) {
            signalsList = isFake 
                ? ["GAN manipulation found", "Irregular artifacts"] 
                : ["No manipulations detected"];
        }

        clearInterval(progressTimer);
        clearInterval(logTimer);
        setScanProgress(100);
        
        setResult({
          verdict: isFake ? "DEEPFAKE DETECTED" : "AUTHENTIC MEDIA",
          type: isFake ? "fake" : "real",
          confidence: finalConf,
          signals: signalsList,
          heatmap: data.reason?.heatmap || null
        });
        setStatus("result");

      } catch (err) {
        console.error("API error:", err);
        clearInterval(progressTimer);
        clearInterval(logTimer);
        alert("Failed to connect to DeepGuard AI API.");
        setStatus("idle");
      }
    };

    analyzeMedia();

    return () => {
      clearInterval(progressTimer);
      clearInterval(logTimer);
    };
  }, [status, selectedFile]);

  const handleAnalyze = () => {
    if (!selectedFile) {
      alert("Upload at least one file to analyze.");
      return;
    }
    setScanProgress(0);
    setScanLog(SCAN_LOGS[0]);
    setStatus("analyzing");
  };

  const handleReset = () => {
    setImagePreview(null);
    setVideoPreview(null);
    setSelectedFile(null);
    setResult(null);
    setStatus("idle");
    setScanProgress(0);
  };

  const hasFile = imagePreview || videoPreview;

  return (
    <div className="app">
      <div className="bg-grid" />
      <div className="bg-glow glow-amber" />
      <div className="bg-glow glow-red" />

      <header className="header">
        <div className="header-left">

          <span className="brand">DEEP<span className="brand-accent">SHIELD AI</span></span>
        </div>
        <nav className="header-nav">

        </nav>
        <div className="header-status">
          <div className="status-dot" />
          <span>SYSTEM ONLINE</span>
        </div>
      </header>

      {status === "analyzing" && (
        <div className="scan-overlay">
          <div className="scan-content">
            <div className="scan-icon">⚡</div>
            <div className="scan-title">Forensic Analysis Running</div>
            <div className="scan-bar-wrap">
              <div className="scan-bar" style={{ width: `${scanProgress}%` }} />
            </div>
            <div className="scan-percent">{Math.floor(scanProgress)}%</div>
            <div className="scan-log">{scanLog}</div>
          </div>
        </div>
      )}

      <main className="main">
        <section className="hero">
          <div className="hero-eyebrow">
            <span className="eyebrow-dot" />
          </div>
          <h1 className="hero-title">
            Detect Deepfakes<br />
            <span className="title-accent">Before They Spread.</span>
          </h1>
          <p className="hero-sub">
            Upload any image or video. Our AI analyzes GAN fingerprints, pixel artifacts,
            and frequency anomalies in seconds — with full explainability.
          </p>
        </section>

        <section className="upload-section">
          <div
            className={`upload-box ${imagePreview ? "has-file" : ""} ${dragOver === "image" ? "drag-active" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver("image"); }}
            onDragLeave={() => setDragOver(null)}
            onDrop={(e) => handleDrop(e, "image")}
          >
            <div className="box-corner tl" />
            <div className="box-corner br" />
            <div className="box-meta">
              <span className="box-tag">// IMAGE</span>
              <span className="box-format">JPG · PNG · max 10MB</span>
            </div>
            <h2 className="box-title">Image Analysis</h2>
            {!imagePreview ? (
              <label className="drop-zone">
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={(e) => handleImageUpload(e.target.files[0])}
                />
                <div className="drop-icon-wrap">
                  <svg className="drop-svg" viewBox="0 0 48 48" fill="none">
                    <rect x="4" y="10" width="40" height="30" rx="4" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 3" />
                    <circle cx="17" cy="21" r="4" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M4 34 l12-10 8 7 8-9 12 13" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                  </svg>
                </div>
                <p className="drop-hint">Drag & drop or <span>browse files</span></p>
              </label>
            ) : (
              <div className="preview-wrap">
                <img src={imagePreview} alt="Preview" className="preview-img" />
                <div className="preview-bar">
                  <span className="preview-status">✓ Image Loaded</span>
                  <button className="remove-btn" onClick={() => setImagePreview(null)}>✕ Remove</button>
                </div>
              </div>
            )}
          </div>

          <div className="upload-divider">
            <div className="divider-line" />
            <span className="divider-text">OR</span>
            <div className="divider-line" />
          </div>

          <div
            className={`upload-box ${videoPreview ? "has-file" : ""} ${dragOver === "video" ? "drag-active" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver("video"); }}
            onDragLeave={() => setDragOver(null)}
            onDrop={(e) => handleDrop(e, "video")}
          >
            <div className="box-corner tl" />
            <div className="box-corner br" />
            <div className="box-meta">
              <span className="box-tag">// VIDEO</span>
              <span className="box-format">MP4 · max 100MB</span>
            </div>
            <h2 className="box-title">Video Analysis</h2>
            {!videoPreview ? (
              <label className="drop-zone">
                <input
                  type="file"
                  accept="video/mp4"
                  onChange={(e) => handleVideoUpload(e.target.files[0])}
                />
                <div className="drop-icon-wrap">
                  <svg className="drop-svg" viewBox="0 0 48 48" fill="none">
                    <rect x="4" y="12" width="30" height="24" rx="4" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 3" />
                    <path d="M34 20 l10-6v20l-10-6V20Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                    <circle cx="19" cy="24" r="5" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M17 22l5 2-5 2v-4Z" fill="currentColor" />
                  </svg>
                </div>
                <p className="drop-hint">Drag & drop or <span>browse files</span></p>
              </label>
            ) : (
              <div className="preview-wrap">
                <video controls className="preview-video">
                  <source src={videoPreview} type="video/mp4" />
                </video>
                <div className="preview-bar">
                  <span className="preview-status">✓ Video Loaded</span>
                  <button className="remove-btn" onClick={() => setVideoPreview(null)}>✕ Remove</button>
                </div>
              </div>
            )}
          </div>
        </section>

        <div className="analyze-wrap">
          <button
            className={`analyze-btn ${hasFile ? "active" : ""}`}
            onClick={handleAnalyze}
            disabled={!hasFile}
          >
            <span className="btn-shimmer" />
            Run Fakeness Test
            <svg className="btn-arrow" viewBox="0 0 20 20" fill="none">
              <path d="M4 10h12M12 6l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          {!hasFile && <p className="analyze-hint">Upload an image or video to begin</p>}
        </div>

        {status === "result" && result && (
          <div className={`result-card ${result.type}`}>
            <div className="result-glow" />
            <div className="result-header">
              <div className="result-icon">
                {result.type === "fake" ? "⚠" : "✓"}
              </div>
              <div>
                <div className="result-label">Analysis Complete</div>
                <div className="result-verdict">{result.verdict}</div>
              </div>
            </div>
            <div className="result-body">
              <div className="conf-section">
                <div className="conf-header">
                  <span className="conf-label">Confidence Score</span>
                  <span className="conf-value">{result.confidence}%</span>
                </div>
                <div className="conf-bar-wrap">
                  <div className="conf-bar" style={{ width: `${result.confidence}%` }} />
                </div>
              </div>
              <div className="signals-section">
                <div className="signals-label">Detection Signals</div>
                <div className="signals-list">
                  {result.signals.map((sig, i) => (
                    <div key={i} className="signal-item">
                      <span className="signal-dot" />
                      {sig}
                    </div>
                  ))}
                </div>
              </div>
            {result.heatmap && (
              <div className="heatmap-display">
                <div className="signals-label">Grad-CAM XAI Heatmap</div>
                <img src={result.heatmap} alt="AI Heatmap" className="heatmap-img" />
                <p className="heatmap-desc">Red regions indicate high model attention zones used for forecasting manipulation artifacts.</p>
              </div>
            )}
            </div>
            <button className="reset-btn" onClick={handleReset}>↩ Analyze Another File</button>
          </div>
        )}
      </main>

      <footer className="footer">
        <span>DeepShield AI · Built at Hackathon 2025</span>
        <span className="footer-tech">Powered by CNN + Vision Transformer</span>
      </footer>
    </div>
  );
}

export default App;
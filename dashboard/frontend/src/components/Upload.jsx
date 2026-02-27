import { useState, useRef } from "react";

const API = "/api";

export default function Upload({ onUploaded, loaded, fileNames }) {
  const [uploading, setUploading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);
  const timerRef = useRef(null);

  function startTimer() {
    setElapsed(0);
    timerRef.current = setInterval(() => {
      setElapsed((e) => e + 1);
    }, 1000);
  }

  function stopTimer() {
    clearInterval(timerRef.current);
    timerRef.current = null;
  }

  async function handleUpload(body, isFormData = false) {
    setUploading(true);
    setError(null);
    startTimer();

    try {
      const opts = { method: "POST" };
      if (isFormData) {
        opts.body = body;
      } else {
        opts.headers = { "Content-Type": "application/json" };
        opts.body = JSON.stringify(body);
      }

      const res = await fetch(`${API}/upload`, opts);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      onUploaded(data);
    } catch (err) {
      setError(err.message);
    } finally {
      stopTimer();
      setUploading(false);
    }
  }

  function handleFiles() {
    const files = fileRef.current?.files;
    if (!files || files.length === 0) return;
    const form = new FormData();
    for (const f of files) {
      form.append("file", f);
    }
    handleUpload(form, true);
  }

  return (
    <section className="upload-section">
      <div className="upload-header">Document</div>
      <div className="upload-controls">
        <label className="file-label">
          <input
            ref={fileRef}
            type="file"
            accept=".txt"
            multiple
            onChange={handleFiles}
            disabled={uploading}
          />
          Upload .txt files
        </label>
        <span className="upload-or">or</span>
        <button
          onClick={() => handleUpload({ use_sample: true })}
          disabled={uploading}
          className="upload-btn upload-btn-outline"
        >
          Use Sample (Cinderella)
        </button>
      </div>

      {uploading && (
        <div className="upload-status">
          <span className="spinner" />
          Building RAPTOR tree... ({elapsed}s)
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {loaded && !uploading && (
        <div className="upload-success">
          Indexed {fileNames.length} file{fileNames.length !== 1 ? "s" : ""}
          {fileNames.length > 0 && (
            <span className="file-list">
              {fileNames.map((n, i) => (
                <span key={i} className="file-tag">{n}</span>
              ))}
            </span>
          )}
        </div>
      )}
    </section>
  );
}

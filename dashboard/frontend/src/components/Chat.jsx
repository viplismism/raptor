import { useState } from "react";
import Markdown from "react-markdown";

const API = "/api";

export default function Chat({ loaded, suggestions }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  async function submitQuestion(q) {
    if (!q.trim() || !loaded || loading) return;

    setQuestion("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Query failed");
      setHistory((prev) => [data, ...prev]);
    } catch (err) {
      setHistory((prev) => [{ question: q.trim(), error: err.message }, ...prev]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    submitQuestion(question);
  }

  // Only show suggestions that haven't been asked yet
  const asked = new Set(history.map((h) => h.question));
  const visibleSuggestions = suggestions.filter((s) => !asked.has(s));

  return (
    <section className="chat-section">
      <form className="query-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={loaded ? "Ask a question about the document..." : "Upload a document first"}
          disabled={!loaded || loading}
          className="query-input"
        />
        <button
          type="submit"
          disabled={!loaded || loading || !question.trim()}
          className="btn-send"
        >
          {loading ? "Querying..." : "Send"}
        </button>
      </form>

      {loaded && visibleSuggestions.length > 0 && !loading && (
        <div className="suggestions">
          <span className="suggestions-label">Try asking:</span>
          <div className="suggestions-list">
            {visibleSuggestions.map((s, i) => (
              <button
                key={i}
                className="suggestion-chip"
                onClick={() => submitQuestion(s)}
                disabled={loading}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-bar">
          <span className="spinner" /> Querying both pipelines...
        </div>
      )}

      <div className="history">
        {history.length === 0 && !loading && (
          <div className="empty-state">
            <p>
              {loaded
                ? "Ask a question to compare RAPTOR vs flat RAG retrieval."
                : "Upload a document to get started."}
            </p>
          </div>
        )}
        {history.map((item, i) => (
          <ResultCard key={i} item={item} />
        ))}
      </div>
    </section>
  );
}

function ResultCard({ item }) {
  const [contextModal, setContextModal] = useState(null);

  if (item.error) {
    return (
      <div className="result-card error-card">
        <div className="result-question">{item.question}</div>
        <div className="error">{item.error}</div>
      </div>
    );
  }

  return (
    <>
      <div className="result-card">
        <div className="result-question">{item.question}</div>
        <div className="result-columns">
          <ResultColumn
            label="Flat RAG"
            accent="flat"
            data={item.flat}
            onShowContext={() => setContextModal({ label: "Flat RAG", text: item.flat.context })}
          />
          <ResultColumn
            label="RAPTOR"
            accent="raptor"
            data={item.raptor}
            onShowContext={() => setContextModal({ label: "RAPTOR", text: item.raptor.context })}
          />
        </div>
      </div>

      {contextModal && (
        <div className="context-modal-overlay" onClick={() => setContextModal(null)}>
          <div className="context-modal" onClick={(e) => e.stopPropagation()}>
            <div className="context-modal-header">
              <span className="context-modal-title">
                Retrieved Context — {contextModal.label}
              </span>
              <button className="tree-modal-close" onClick={() => setContextModal(null)}>
                &#x2715;
              </button>
            </div>
            <pre className="context-modal-body">{contextModal.text}</pre>
          </div>
        </div>
      )}
    </>
  );
}

function ResultColumn({ label, accent, data, onShowContext }) {
  return (
    <div className={`result-col ${accent}`}>
      <div className="col-header">
        <span className="col-label">{label}</span>
      </div>

      <div className="col-meta">
        <span className="timing-badge">Retrieval {data.retrieve_time}s</span>
        <span className="timing-badge">QA {data.qa_time}s</span>
        {data.layers_hit && (
          <span className="timing-badge layers-badge">
            Layers [{data.layers_hit.join(", ")}]
          </span>
        )}
      </div>

      <div className="context-section">
        <button className="context-toggle" onClick={onShowContext}>
          View retrieved context
        </button>
      </div>

      <div className="answer">
        <div className="answer-label">Answer</div>
        <Markdown>{data.answer}</Markdown>
      </div>
    </div>
  );
}

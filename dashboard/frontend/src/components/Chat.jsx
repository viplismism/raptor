import { useState, useRef } from "react";
import Markdown from "react-markdown";

const API = "/api";

export default function Chat({ loaded, suggestions }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [streaming, setStreaming] = useState(null); // current streaming result
  const abortRef = useRef(null);

  async function submitQuestion(q) {
    if (!q.trim() || !loaded || loading) return;

    const trimmed = q.trim();
    setQuestion("");
    setLoading(true);
    setStreaming({ question: trimmed, flat: {}, raptor: {} });

    try {
      const controller = new AbortController();
      abortRef.current = controller;

      const res = await fetch(`${API}/query?stream=1`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Query failed");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let result = { question: trimmed, flat: { answer: "" }, raptor: { answer: "" } };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (!line.trim()) continue;
          const msg = JSON.parse(line);

          if (msg.type === "retrieval") {
            result.flat = { ...result.flat, ...msg.flat, answer: "" };
            result.raptor = { ...result.raptor, ...msg.raptor, answer: "" };
            setStreaming({ ...result });
          } else if (msg.type === "flat_token") {
            result.flat.answer += msg.token;
            setStreaming({ ...result });
          } else if (msg.type === "flat_done") {
            result.flat.qa_time = msg.qa_time;
            setStreaming({ ...result });
          } else if (msg.type === "raptor_token") {
            result.raptor.answer += msg.token;
            setStreaming({ ...result });
          } else if (msg.type === "raptor_done") {
            result.raptor.qa_time = msg.qa_time;
            setStreaming({ ...result });
          }
        }
      }

      setHistory((prev) => [result, ...prev]);
    } catch (err) {
      if (err.name !== "AbortError") {
        setHistory((prev) => [{ question: trimmed, error: err.message }, ...prev]);
      }
    } finally {
      setLoading(false);
      setStreaming(null);
      abortRef.current = null;
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    submitQuestion(question);
  }

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

      {streaming && (
        <StreamingCard item={streaming} />
      )}

      <div className="history">
        {history.length === 0 && !loading && !streaming && (
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

function StreamingCard({ item }) {
  const hasFlat = item.flat?.answer;
  const hasRaptor = item.raptor?.answer;

  return (
    <div className="result-card streaming">
      <div className="result-question">{item.question}</div>
      <div className="result-columns">
        <div className="result-col flat">
          <div className="col-header">
            <span className="col-label">Flat RAG</span>
          </div>
          {item.flat?.retrieve_time != null && (
            <div className="col-meta">
              <span className="timing-badge">Retrieval {item.flat.retrieve_time}s</span>
              {item.flat.rerank_time != null && (
                <span className="timing-badge">Rerank {item.flat.rerank_time}s</span>
              )}
              {item.flat.qa_time != null && (
                <span className="timing-badge">QA {item.flat.qa_time}s</span>
              )}
            </div>
          )}
          <div className="answer">
            <div className="answer-label">{hasFlat ? "Answer" : "Generating..."}</div>
            {hasFlat ? <Markdown>{item.flat.answer}</Markdown> : <span className="spinner" />}
          </div>
        </div>
        <div className="result-col raptor">
          <div className="col-header">
            <span className="col-label">RAPTOR</span>
          </div>
          {item.raptor?.retrieve_time != null && (
            <div className="col-meta">
              <span className="timing-badge">Retrieval {item.raptor.retrieve_time}s</span>
              {item.raptor.qa_time != null && (
                <span className="timing-badge">QA {item.raptor.qa_time}s</span>
              )}
              {item.raptor.layers_hit && (
                <span className="timing-badge layers-badge">
                  Layers [{item.raptor.layers_hit.join(", ")}]
                </span>
              )}
            </div>
          )}
          <div className="answer">
            <div className="answer-label">{hasRaptor ? "Answer" : "Waiting..."}</div>
            {hasRaptor ? <Markdown>{item.raptor.answer}</Markdown> : <span className="spinner" />}
          </div>
        </div>
      </div>
    </div>
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
        {data.rerank_time != null && (
          <span className="timing-badge">Rerank {data.rerank_time}s</span>
        )}
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

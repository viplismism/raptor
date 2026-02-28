"""
RAPTOR Dashboard — Flask API server.

Endpoints:
    POST /api/upload   – Upload files (.txt, .pdf, .docx) or raw text
    POST /api/query    – Query both pipelines side-by-side (supports streaming)
    GET  /api/status   – Check whether a document is loaded
    GET  /api/tree     – Return full tree structure for visualization
"""

import json
import logging
import os
import pickle
import sys
import time

import anthropic
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from raptor.models.embedding_models import SBertEmbeddingModel
from raptor.models.anthropic_models import ClaudeQAModel
from raptor.models.base import BaseSummarizationModel
from raptor.raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.retrieval.flat_retriever import FaissRetriever, FaissRetrieverConfig

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("dashboard")

app = Flask(__name__)
CORS(app)

# ── Global session state ─────────────────────────────────────────────────────
session = {
    "ra": None,
    "faiss_ret": None,
    "tree_info": None,
    "build_times": None,
    "file_names": [],
    "file_texts_norm": [],
    "suggested_queries": [],
    "reranker": None,
}

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "demo", "sample.txt")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".raptor_cache")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

_anthropic_client = anthropic.Anthropic()


# ── Cross-encoder reranker ───────────────────────────────────────────────────

def get_reranker():
    """Lazy-load cross-encoder model."""
    if session["reranker"] is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker...")
        session["reranker"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker loaded.")
    return session["reranker"]


def rerank(question: str, chunks: list, top_k: int = 5) -> list:
    """Rerank text chunks by relevance to the question using cross-encoder."""
    if not chunks:
        return chunks
    reranker = get_reranker()
    pairs = [[question, c] for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


# ── File parsing ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    import io
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract text from a file based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes)
    else:
        return file_bytes.decode("utf-8")


# ── Clean summarization model ────────────────────────────────────────────────

class CleanSummarizationModel(BaseSummarizationModel):
    """Summarization model that produces clean prose without markdown formatting."""

    def __init__(self, model: str = CLAUDE_MODEL):
        self.model = model
        self.client = _anthropic_client

    def summarize(self, context: str, max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=(
                "Write a clear, concise summary of the following text. "
                "Include all key details and facts. "
                "Write in plain prose — do NOT use markdown, headers, bullet points, "
                "or any formatting. Just write flowing paragraphs."
            ),
            messages=[{"role": "user", "content": context}],
        )
        return response.content[0].text


QA_SYSTEM = (
    "Answer the user's question using only the information provided below. "
    "Answer directly and concisely — do NOT say 'based on the context' or "
    "'according to the provided text'. Just answer the question as if you "
    "know the answer naturally."
)


def answer_direct(context: str, question: str) -> str:
    """Answer a question directly without referencing the context."""
    response = _anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=QA_SYSTEM,
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    return response.content[0].text.strip()


def answer_stream(context: str, question: str):
    """Stream answer tokens using Anthropic streaming API."""
    with _anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=QA_SYSTEM,
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def judge_answers(question: str, flat_ctx: str, flat_answer: str, raptor_ctx: str, raptor_answer: str) -> dict:
    """Compare both answers side-by-side and score each on relevance, completeness, correctness (0-10)."""
    try:
        response = _anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            system=(
                "You are comparing two QA systems answering the same question. "
                "System A uses flat chunk retrieval. System B uses hierarchical tree retrieval.\n\n"
                "Score EACH system on three dimensions (0-10). Be discriminating — "
                "if one answer is more complete or accurate, its score should be noticeably higher.\n"
                "- relevance: does it address the question directly?\n"
                "- completeness: does it cover all important aspects?\n"
                "- correctness: is it factually accurate given its retrieved context?\n\n"
                'Output ONLY JSON:\n'
                '{"flat": {"relevance": 7, "completeness": 5, "correctness": 6}, '
                '"raptor": {"relevance": 9, "completeness": 8, "correctness": 9}}'
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"--- System A (Flat RAG) ---\n"
                    f"Context:\n{flat_ctx[:2000]}\n\n"
                    f"Answer: {flat_answer}\n\n"
                    f"--- System B (RAPTOR) ---\n"
                    f"Context:\n{raptor_ctx[:2000]}\n\n"
                    f"Answer: {raptor_answer}"
                ),
            }],
        )
        return json.loads(response.content[0].text.strip())
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return None


def generate_suggestions(file_names: list, file_texts: list) -> list:
    """Generate suggested questions covering all uploaded files."""
    per_file = 500
    excerpts = []
    for name, text in zip(file_names, file_texts):
        snippet = text[:per_file].strip()
        excerpts.append(f"[{name}]\n{snippet}")
    combined_preview = "\n\n".join(excerpts)

    n_questions = min(len(file_names) * 2, 6)
    try:
        response = _anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=(
                f"You are given excerpts from {len(file_names)} different documents. "
                f"Suggest exactly {n_questions} interesting questions a reader might ask — "
                "make sure to cover ALL documents, not just one. "
                "Output ONLY the questions, one per line, no numbering, no bullets, no other text."
            ),
            messages=[{"role": "user", "content": combined_preview}],
        )
        lines = [l.strip() for l in response.content[0].text.strip().split("\n") if l.strip()]
        return lines[:n_questions]
    except Exception as e:
        logger.warning(f"Failed to generate suggestions: {e}")
        return []


def find_node_source(node_text: str, file_names: list, file_texts_normalized: list) -> str:
    """Figure out which file a node's text came from by substring matching."""
    normalized = " ".join(node_text.split())
    for length in [200, 100, 50]:
        probe = normalized[:length].strip()
        if not probe:
            continue
        for name, norm_text in zip(file_names, file_texts_normalized):
            if probe in norm_text:
                return name
    if len(normalized) > 40:
        mid = len(normalized) // 2
        probe = normalized[mid - 20 : mid + 20]
        for name, norm_text in zip(file_names, file_texts_normalized):
            if probe in norm_text:
                return name
    return file_names[0] if len(file_names) == 1 else None


# ── Session persistence ──────────────────────────────────────────────────────

def save_session():
    """Save the RAPTOR tree and metadata to disk for persistence across restarts."""
    if session["ra"] is None:
        return
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Save the tree (pickleable) separately from unpickleable objects
    tree_path = os.path.join(SAVE_DIR, "tree.pkl")
    meta_path = os.path.join(SAVE_DIR, "meta.pkl")
    session["ra"].save(tree_path)
    meta = {
        "tree_info": session["tree_info"],
        "build_times": session["build_times"],
        "file_names": session["file_names"],
        "file_texts_norm": session["file_texts_norm"],
        "suggested_queries": session["suggested_queries"],
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    logger.info(f"Session saved to {SAVE_DIR}")


def load_session() -> bool:
    """Load a previously saved session. Returns True if successful."""
    tree_path = os.path.join(SAVE_DIR, "tree.pkl")
    meta_path = os.path.join(SAVE_DIR, "meta.pkl")
    if not os.path.exists(tree_path) or not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # Rebuild RA and FAISS from saved tree
        emb = SBertEmbeddingModel()
        qa = ClaudeQAModel()
        summarizer = CleanSummarizationModel()
        ra_config = RetrievalAugmentationConfig(
            embedding_model=emb, qa_model=qa, summarization_model=summarizer,
        )
        ra = RetrievalAugmentation(config=ra_config, tree=tree_path)

        emb_model_name = list(ra.tree_builder.embedding_models.keys())[0]
        faiss_config = FaissRetrieverConfig(
            embedding_model=emb, question_embedding_model=emb,
            use_top_k=True, top_k=5, embedding_model_string=emb_model_name,
        )
        faiss_ret = FaissRetriever(faiss_config)
        faiss_ret.build_from_leaf_nodes(list(ra.tree.leaf_nodes.values()))

        session["ra"] = ra
        session["faiss_ret"] = faiss_ret
        for key in meta:
            session[key] = meta[key]

        logger.info(f"Session restored from {SAVE_DIR}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load session: {e}")
        return False


# Try restoring previous session on startup
load_session()


def build_index(combined_text: str, file_names: list, file_texts: list) -> dict:
    """Build both RAPTOR tree and flat FAISS index from text."""
    emb = SBertEmbeddingModel()
    qa = ClaudeQAModel()
    summarizer = CleanSummarizationModel()

    # Build RAPTOR tree
    t0 = time.time()
    ra_config = RetrievalAugmentationConfig(
        embedding_model=emb,
        qa_model=qa,
        summarization_model=summarizer,
    )
    ra = RetrievalAugmentation(config=ra_config)
    ra.add_documents(combined_text, force=True)
    raptor_time = time.time() - t0

    tree = ra.tree

    # Build flat FAISS index from same leaf nodes
    t0 = time.time()
    emb_model_name = list(ra.tree_builder.embedding_models.keys())[0]
    faiss_config = FaissRetrieverConfig(
        embedding_model=emb,
        question_embedding_model=emb,
        use_top_k=True,
        top_k=5,
        embedding_model_string=emb_model_name,
    )
    faiss_ret = FaissRetriever(faiss_config)
    faiss_ret.build_from_leaf_nodes(list(tree.leaf_nodes.values()))
    flat_time = time.time() - t0

    # Generate suggested queries covering all files
    suggestions = generate_suggestions(file_names, file_texts)

    tree_info = {
        "num_layers": tree.num_layers,
        "total_nodes": len(tree.all_nodes),
        "leaf_nodes": len(tree.leaf_nodes),
        "root_nodes": len(tree.root_nodes),
    }
    build_times = {
        "raptor": round(raptor_time, 2),
        "flat": round(flat_time, 2),
    }

    session["ra"] = ra
    session["faiss_ret"] = faiss_ret
    session["tree_info"] = tree_info
    session["build_times"] = build_times
    session["file_names"] = file_names
    session["file_texts_norm"] = [" ".join(t.split()) for t in file_texts]
    session["suggested_queries"] = suggestions

    # Persist session to disk
    save_session()

    return {
        "tree_info": tree_info,
        "build_times": build_times,
        "file_names": file_names,
        "suggested_queries": suggestions,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept files (.txt, .pdf, .docx), raw text, or 'use_sample' flag."""
    try:
        text_parts = []
        file_names = []

        if request.is_json:
            data = request.get_json()
            if data.get("use_sample"):
                if not os.path.exists(SAMPLE_PATH):
                    return jsonify({"error": "Sample file not found"}), 404
                with open(SAMPLE_PATH, "r") as f:
                    text_parts.append(f.read())
                file_names.append("sample.txt")
            elif data.get("text"):
                text_parts.append(data["text"])
                file_names.append("pasted-text")
        else:
            files = request.files.getlist("file")
            if not files or all(f.filename == "" for f in files):
                return jsonify({"error": "No files selected"}), 400

            for f in files:
                if f.filename == "":
                    continue
                raw = f.read()
                text = extract_text(f.filename, raw)
                if text.strip():
                    text_parts.append(text)
                    file_names.append(f.filename)

        if not text_parts:
            return jsonify({"error": "No text provided."}), 400

        combined = "\n\n".join(text_parts)

        logger.info(
            f"Building index for {len(file_names)} file(s): {file_names} "
            f"({len(combined)} chars total)"
        )
        result = build_index(combined, file_names, text_parts)
        logger.info(f"Index built: {result['tree_info']}")

        return jsonify({"status": "ok", **result})

    except Exception as e:
        logger.exception("Upload failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query():
    """Query both RAPTOR and flat pipelines. Supports streaming via ?stream=1."""
    if session["ra"] is None:
        return jsonify({"error": "No document loaded. Upload a document first."}), 400

    data = request.get_json()
    if not data or not data.get("question"):
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"]
    use_stream = request.args.get("stream") == "1"
    ra = session["ra"]
    faiss_ret = session["faiss_ret"]

    try:
        # RAPTOR retrieval + reranking
        t0 = time.time()
        raptor_ctx, raptor_layers = ra.retrieve(
            question, top_k=5, max_tokens=3500,
            collapse_tree=True, return_layer_information=True,
        )
        raptor_retrieve_time = time.time() - t0
        layers_hit = sorted(set(l["layer_number"] for l in raptor_layers))

        # Flat retrieval
        t0 = time.time()
        flat_ctx = faiss_ret.retrieve(question)
        flat_retrieve_time = time.time() - t0

        # Cross-encoder reranking on flat results
        t0_rerank = time.time()
        flat_chunks = [c for c in flat_ctx.split("\n\n") if c.strip()]
        if len(flat_chunks) > 1:
            flat_chunks = rerank(question, flat_chunks, top_k=5)
            flat_ctx = "\n\n".join(flat_chunks)
        rerank_time = time.time() - t0_rerank

        if use_stream:
            # Streaming mode: send retrieval info, then stream both answers in parallel
            def generate():
                yield json.dumps({
                    "type": "retrieval",
                    "question": question,
                    "raptor": {
                        "context": raptor_ctx,
                        "retrieve_time": round(raptor_retrieve_time, 3),
                        "layers_hit": layers_hit,
                    },
                    "flat": {
                        "context": flat_ctx,
                        "retrieve_time": round(flat_retrieve_time, 3),
                        "rerank_time": round(rerank_time, 3),
                    },
                }) + "\n"

                # Stream both answers in parallel using threads
                import queue
                from threading import Thread

                q = queue.Queue()

                answers = {"flat": "", "raptor": ""}

                def stream_flat():
                    t0 = time.time()
                    for token in answer_stream(flat_ctx, question):
                        answers["flat"] += token
                        q.put(json.dumps({"type": "flat_token", "token": token}) + "\n")
                    qa_time = time.time() - t0
                    q.put(json.dumps({"type": "flat_done", "qa_time": round(qa_time, 3)}) + "\n")

                def stream_raptor():
                    t0 = time.time()
                    for token in answer_stream(raptor_ctx, question):
                        answers["raptor"] += token
                        q.put(json.dumps({"type": "raptor_token", "token": token}) + "\n")
                    qa_time = time.time() - t0
                    q.put(json.dumps({"type": "raptor_done", "qa_time": round(qa_time, 3)}) + "\n")

                t_flat = Thread(target=stream_flat)
                t_raptor = Thread(target=stream_raptor)
                t_flat.start()
                t_raptor.start()

                done_count = 0
                while done_count < 2:
                    try:
                        msg = q.get(timeout=0.05)
                        yield msg
                        if '"flat_done"' in msg or '"raptor_done"' in msg:
                            done_count += 1
                    except queue.Empty:
                        continue

                t_flat.join()
                t_raptor.join()

                # Score both answers comparatively with LLM judge
                result = judge_answers(
                    question, flat_ctx, answers["flat"], raptor_ctx, answers["raptor"]
                )

                yield json.dumps({
                    "type": "scores",
                    "flat": result.get("flat") if result else None,
                    "raptor": result.get("raptor") if result else None,
                }) + "\n"

            return Response(
                stream_with_context(generate()),
                mimetype="application/x-ndjson",
            )

        # Non-streaming mode
        t0 = time.time()
        raptor_answer = answer_direct(raptor_ctx, question)
        raptor_qa_time = time.time() - t0

        t0 = time.time()
        flat_answer = answer_direct(flat_ctx, question)
        flat_qa_time = time.time() - t0

        return jsonify({
            "question": question,
            "raptor": {
                "answer": raptor_answer,
                "context": raptor_ctx,
                "retrieve_time": round(raptor_retrieve_time, 3),
                "qa_time": round(raptor_qa_time, 3),
                "layers_hit": layers_hit,
            },
            "flat": {
                "answer": flat_answer,
                "context": flat_ctx,
                "retrieve_time": round(flat_retrieve_time, 3),
                "rerank_time": round(rerank_time, 3),
                "qa_time": round(flat_qa_time, 3),
            },
        })

    except Exception as e:
        logger.exception("Query failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Return whether a document is loaded and tree info if so."""
    loaded = session["ra"] is not None
    resp = {"loaded": loaded}
    if loaded:
        resp["tree_info"] = session["tree_info"]
        resp["build_times"] = session["build_times"]
        resp["file_names"] = session["file_names"]
        resp["suggested_queries"] = session["suggested_queries"]
    return jsonify(resp)


@app.route("/api/tree", methods=["GET"])
def tree_data():
    """Return the full tree structure for visualization."""
    if session["ra"] is None:
        return jsonify({"error": "No document loaded"}), 400

    tree = session["ra"].tree
    file_names = session["file_names"]
    file_texts_norm = session.get("file_texts_norm", [])
    layers = {}

    leaf_sources = {}
    for node in tree.leaf_nodes.values():
        source = find_node_source(node.text, file_names, file_texts_norm)
        leaf_sources[node.index] = source

    def get_node_source(node_index):
        if node_index in leaf_sources:
            return leaf_sources[node_index]
        node = tree.all_nodes.get(node_index)
        if node and node.children:
            child_sources = set()
            for child_idx in node.children:
                s = get_node_source(child_idx)
                if s:
                    child_sources.add(s)
            if len(child_sources) == 1:
                return child_sources.pop()
            elif child_sources:
                return ", ".join(sorted(child_sources))
        return None

    for layer_num, nodes in tree.layer_to_nodes.items():
        layers[layer_num] = []
        for node in nodes:
            source = get_node_source(node.index)
            layers[layer_num].append({
                "index": node.index,
                "text": node.text,
                "children": sorted(node.children) if node.children else [],
                "source": source,
            })

    return jsonify({
        "num_layers": tree.num_layers,
        "total_nodes": len(tree.all_nodes),
        "layers": layers,
    })


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, port=5000)

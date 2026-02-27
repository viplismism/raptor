"""
RAPTOR Dashboard — Flask API server.

Endpoints:
    POST /api/upload   – Upload .txt file(s) or raw text, build RAPTOR tree + flat FAISS index
    POST /api/query    – Query both pipelines side-by-side
    GET  /api/status   – Check whether a document is loaded
    GET  /api/tree     – Return full tree structure for visualization
"""

import logging
import os
import sys
import time

import anthropic
from flask import Flask, jsonify, request
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
    "file_texts_norm": [],    # whitespace-normalized text per file (for node→file mapping)
    "suggested_queries": [],
}

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "demo", "sample.txt")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

_anthropic_client = anthropic.Anthropic()


# ── Clean summarization model (no markdown headers) ──────────────────────────

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
            messages=[
                {"role": "user", "content": context}
            ],
        )
        return response.content[0].text


def answer_direct(context: str, question: str) -> str:
    """Answer a question directly without referencing the context."""
    response = _anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=(
            "Answer the user's question using only the information provided below. "
            "Answer directly and concisely — do NOT say 'based on the context' or "
            "'according to the provided text'. Just answer the question as if you "
            "know the answer naturally."
        ),
        messages=[
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ],
    )
    return response.content[0].text.strip()


def generate_suggestions(file_names: list, file_texts: list) -> list:
    """Generate suggested questions covering all uploaded files."""
    # Sample ~500 chars from each file so all files are represented
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
            messages=[
                {"role": "user", "content": combined_preview}
            ],
        )
        lines = [l.strip() for l in response.content[0].text.strip().split("\n") if l.strip()]
        return lines[:n_questions]
    except Exception as e:
        logger.warning(f"Failed to generate suggestions: {e}")
        return []


def find_node_source(node_text: str, file_names: list, file_texts_normalized: list) -> str:
    """Figure out which file a node's text came from by substring matching."""
    # Normalize whitespace in the probe to match normalized file texts
    normalized = " ".join(node_text.split())
    # Try progressively shorter probes until we find a match
    for length in [200, 100, 50]:
        probe = normalized[:length].strip()
        if not probe:
            continue
        for name, norm_text in zip(file_names, file_texts_normalized):
            if probe in norm_text:
                return name
    # Last resort: try a mid-section of the text
    if len(normalized) > 40:
        mid = len(normalized) // 2
        probe = normalized[mid - 20 : mid + 20]
        for name, norm_text in zip(file_names, file_texts_normalized):
            if probe in norm_text:
                return name
    return file_names[0] if len(file_names) == 1 else None


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

    return {
        "tree_info": tree_info,
        "build_times": build_times,
        "file_names": file_names,
        "suggested_queries": suggestions,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept .txt file(s), raw text in JSON body, or 'use_sample' flag."""
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
            # Handle multiple file uploads
            files = request.files.getlist("file")
            if not files or all(f.filename == "" for f in files):
                return jsonify({"error": "No files selected"}), 400

            for f in files:
                if f.filename == "":
                    continue
                content = f.read().decode("utf-8")
                text_parts.append(content)
                file_names.append(f.filename)

        if not text_parts:
            return jsonify({"error": "No text provided."}), 400

        # Combine texts (plain, no markers — source mapping happens post-build)
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
    """Query both RAPTOR and flat pipelines."""
    if session["ra"] is None:
        return jsonify({"error": "No document loaded. Upload a document first."}), 400

    data = request.get_json()
    if not data or not data.get("question"):
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"]
    ra = session["ra"]
    faiss_ret = session["faiss_ret"]

    try:
        # RAPTOR retrieval + QA
        t0 = time.time()
        raptor_ctx, raptor_layers = ra.retrieve(
            question, top_k=5, max_tokens=3500,
            collapse_tree=True, return_layer_information=True,
        )
        raptor_retrieve_time = time.time() - t0

        t0 = time.time()
        raptor_answer = answer_direct(raptor_ctx, question)
        raptor_qa_time = time.time() - t0

        layers_hit = sorted(set(l["layer_number"] for l in raptor_layers))

        # Flat retrieval + QA
        t0 = time.time()
        flat_ctx = faiss_ret.retrieve(question)
        flat_retrieve_time = time.time() - t0

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

    # Build source mapping for leaf nodes
    leaf_sources = {}
    for node in tree.leaf_nodes.values():
        source = find_node_source(node.text, file_names, file_texts_norm)
        leaf_sources[node.index] = source

    # For summary nodes, derive source from children
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

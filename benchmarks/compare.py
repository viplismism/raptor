#!/usr/bin/env python3
"""
RAPTOR vs Flat RAG — side-by-side comparison script.

Ingests a document, builds both a RAPTOR tree and a standard flat FAISS index
from the same chunks, then runs each question against both pipelines and prints
the retrieved context + final answer so you can see the difference.

Usage:
    python benchmarks/compare.py                          # uses defaults (demo/sample.txt)
    python benchmarks/compare.py --document my_doc.txt
    python benchmarks/compare.py --questions "Who is X?" "What happened to Y?"
    python benchmarks/compare.py --embedding sbert        # use SBert instead of OpenAI
    python benchmarks/compare.py --output results.json    # save JSON results
"""

import argparse
import json
import logging
import textwrap
import time
import sys
import os

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("compare")

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DOC = os.path.join(os.path.dirname(__file__), "..", "demo", "sample.txt")
DEFAULT_QUESTIONS = [
    "How did Cinderella reach her happy ending?",
    "What role did the birds play in the story?",
    "How were the stepsisters punished at the end?",
]
CONTEXT_PREVIEW_CHARS = 500  # how many chars of retrieved context to show


# ── Helpers ──────────────────────────────────────────────────────────────────
def truncate(text: str, limit: int = CONTEXT_PREVIEW_CHARS) -> str:
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def divider(char="=", width=80):
    print(char * width)


def section(title: str):
    print()
    divider()
    print(f"  {title}")
    divider()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison: RAPTOR tree retrieval vs flat FAISS RAG"
    )
    parser.add_argument(
        "--document",
        default=DEFAULT_DOC,
        help="Path to the document text file (default: demo/sample.txt)",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="Questions to run (default: 3 built-in Cinderella questions)",
    )
    parser.add_argument(
        "--embedding",
        choices=["openai", "sbert"],
        default="openai",
        help="Embedding model (default: openai)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks/nodes to retrieve (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3500,
        help="Max context tokens for RAPTOR retrieval (default: 3500)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save full results as JSON",
    )
    args = parser.parse_args()

    questions = args.questions or DEFAULT_QUESTIONS

    # ── Lazy imports ─────────────────────────────────────────────────────────
    from raptor.models.anthropic_models import ClaudeQAModel
    from raptor.models.embedding_models import OpenAIEmbeddingModel, SBertEmbeddingModel
    from raptor.raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    from raptor.retrieval.flat_retriever import FaissRetriever, FaissRetrieverConfig

    # ── Choose embedding model ───────────────────────────────────────────────
    if args.embedding == "sbert":
        emb = SBertEmbeddingModel()
        emb_label = "SBert"
    else:
        emb = OpenAIEmbeddingModel()
        emb_label = "OpenAI"

    qa = ClaudeQAModel()

    # ── Load document ────────────────────────────────────────────────────────
    with open(args.document, "r") as f:
        document = f.read()

    section("RAPTOR vs Flat RAG Comparison")
    print(f"  Document   : {args.document} ({len(document)} chars)")
    print(f"  Embedding  : {emb_label}")
    print(f"  Top-K      : {args.top_k}")
    print(f"  Questions  : {len(questions)}")

    # ── 1. Build RAPTOR tree ─────────────────────────────────────────────────
    section("STEP 1 — Building RAPTOR tree")
    t0 = time.time()
    ra_config = RetrievalAugmentationConfig(
        embedding_model=emb,
        qa_model=qa,
    )
    ra = RetrievalAugmentation(config=ra_config)
    ra.add_documents(document, force=True)
    raptor_build_time = time.time() - t0

    tree = ra.tree
    print(f"  Tree built in {raptor_build_time:.1f}s")
    print(f"  Layers     : {tree.num_layers}")
    print(f"  Total nodes: {len(tree.all_nodes)}")
    print(f"  Leaf nodes : {len(tree.leaf_nodes)}")
    print(f"  Root nodes : {len(tree.root_nodes)}")

    # ── 2. Build flat FAISS index from the *same* leaf chunks ────────────────
    section("STEP 2 — Building flat FAISS index (same chunks)")
    t0 = time.time()

    emb_model_name = list(ra.tree_builder.embedding_models.keys())[0]
    faiss_config = FaissRetrieverConfig(
        embedding_model=emb,
        question_embedding_model=emb,
        use_top_k=True,
        top_k=args.top_k,
        embedding_model_string=emb_model_name,
    )
    faiss_ret = FaissRetriever(faiss_config)
    leaf_nodes = list(tree.leaf_nodes.values())
    faiss_ret.build_from_leaf_nodes(leaf_nodes)
    flat_build_time = time.time() - t0

    print(f"  Index built in {flat_build_time:.1f}s")
    print(f"  Indexed {len(leaf_nodes)} chunks")

    # ── 3. Run queries ───────────────────────────────────────────────────────
    results = []

    for i, question in enumerate(questions, 1):
        section(f"QUESTION {i}/{len(questions)}")
        print(f"  {question}")

        # ── RAPTOR: retrieve context + answer ──
        t0 = time.time()
        raptor_ctx, raptor_layers = ra.retrieve(
            question,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            collapse_tree=True,
            return_layer_information=True,
        )
        raptor_retrieve_time = time.time() - t0

        t0 = time.time()
        raptor_answer = qa.answer_question(raptor_ctx, question)
        raptor_qa_time = time.time() - t0

        # ── Flat: retrieve context + answer ──
        t0 = time.time()
        flat_ctx = faiss_ret.retrieve(question)
        flat_retrieve_time = time.time() - t0

        t0 = time.time()
        flat_answer = qa.answer_question(flat_ctx, question)
        flat_qa_time = time.time() - t0

        # ── Display ──
        print()
        print("  --- FLAT RAG (chunk-level only) ---")
        print(f"  Retrieved in {flat_retrieve_time:.2f}s, answered in {flat_qa_time:.2f}s")
        print(f"  Context preview:")
        for line in textwrap.wrap(truncate(flat_ctx), width=76):
            print(f"    {line}")
        print(f"  Answer: {flat_answer}")

        print()
        print("  --- RAPTOR (hierarchical tree) ---")
        print(f"  Retrieved in {raptor_retrieve_time:.2f}s, answered in {raptor_qa_time:.2f}s")
        layers_used = sorted(set(l["layer_number"] for l in raptor_layers))
        print(f"  Layers hit: {layers_used} ({len(raptor_layers)} nodes selected)")
        print(f"  Context preview:")
        for line in textwrap.wrap(truncate(raptor_ctx), width=76):
            print(f"    {line}")
        print(f"  Answer: {raptor_answer}")

        results.append({
            "question": question,
            "flat": {
                "context": flat_ctx,
                "answer": flat_answer,
                "retrieve_time": flat_retrieve_time,
                "qa_time": flat_qa_time,
            },
            "raptor": {
                "context": raptor_ctx,
                "answer": raptor_answer,
                "retrieve_time": raptor_retrieve_time,
                "qa_time": raptor_qa_time,
                "layers_hit": layers_used,
                "nodes_selected": len(raptor_layers),
            },
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    section("SUMMARY")
    print(f"  Build time  — RAPTOR: {raptor_build_time:.1f}s | Flat: {flat_build_time:.1f}s")

    avg_flat_r = sum(r["flat"]["retrieve_time"] for r in results) / len(results)
    avg_flat_q = sum(r["flat"]["qa_time"] for r in results) / len(results)
    avg_rapt_r = sum(r["raptor"]["retrieve_time"] for r in results) / len(results)
    avg_rapt_q = sum(r["raptor"]["qa_time"] for r in results) / len(results)

    print(f"  Avg retrieve — RAPTOR: {avg_rapt_r:.2f}s | Flat: {avg_flat_r:.2f}s")
    print(f"  Avg QA       — RAPTOR: {avg_rapt_q:.2f}s | Flat: {avg_flat_q:.2f}s")
    print()
    print("  Key insight: RAPTOR retrieves from *multiple tree layers* (summaries +")
    print("  leaf chunks), giving the LLM broader context. Flat RAG only retrieves")
    print("  from the raw chunks, so it can miss high-level narrative connections.")
    print()

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.output:
        output_data = {
            "document": args.document,
            "embedding": emb_label,
            "top_k": args.top_k,
            "build_times": {
                "raptor": raptor_build_time,
                "flat": flat_build_time,
            },
            "tree_info": {
                "num_layers": tree.num_layers,
                "total_nodes": len(tree.all_nodes),
                "leaf_nodes": len(tree.leaf_nodes),
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Full results saved to {args.output}")

    divider()


if __name__ == "__main__":
    main()

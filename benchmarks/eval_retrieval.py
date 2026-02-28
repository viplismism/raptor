#!/usr/bin/env python3
"""
Retrieval quality evaluation: RAPTOR vs Flat RAG.

Measures whether the retrieved context actually contains the information
needed to answer the question, using an LLM judge.

Usage:
    python benchmarks/eval_retrieval.py
    python benchmarks/eval_retrieval.py --document demo/sample.txt
    python benchmarks/eval_retrieval.py --output benchmarks/eval_results.json
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("eval")

# ── Test cases: question + expected answer keywords ──────────────────────────

CINDERELLA_QA = [
    {
        "question": "How did Cinderella reach her happy ending?",
        "key_facts": ["bird", "hazel tree", "golden dress", "slipper", "prince", "festival"],
    },
    {
        "question": "What role did the birds play in the story?",
        "key_facts": ["white bird", "wishes", "dress", "slippers", "pigeons", "lentils", "ashes"],
    },
    {
        "question": "How were the stepsisters punished at the end?",
        "key_facts": ["pigeons", "eyes", "pecked", "blind", "wedding"],
    },
    {
        "question": "What did Cinderella's father bring her from his journey?",
        "key_facts": ["branch", "hazel", "hat", "knocked"],
    },
    {
        "question": "How did the prince try to find Cinderella after the festival?",
        "key_facts": ["slipper", "golden shoe", "pitch", "staircase", "pigeon-house"],
    },
]


def score_context(context: str, key_facts: list) -> float:
    """Score how many key facts are present in the retrieved context (0-1)."""
    context_lower = context.lower()
    found = sum(1 for fact in key_facts if fact.lower() in context_lower)
    return found / len(key_facts)


def llm_judge(question: str, context: str, answer: str, client, model: str) -> dict:
    """Use LLM to judge answer quality: relevance, completeness, correctness."""
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=(
            "You are evaluating a QA system. Given a question, the retrieved context, "
            "and the answer produced, rate the answer on three dimensions:\n"
            "- relevance (0-10): does the answer address the question?\n"
            "- completeness (0-10): does the answer cover all important aspects?\n"
            "- correctness (0-10): is the answer factually correct given the context?\n\n"
            "Output ONLY a JSON object like: {\"relevance\": 8, \"completeness\": 7, \"correctness\": 9}\n"
            "No other text."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved context (first 500 chars): {context[:500]}\n\n"
                f"Answer: {answer}"
            ),
        }],
    )
    try:
        return json.loads(response.content[0].text.strip())
    except json.JSONDecodeError:
        return {"relevance": 0, "completeness": 0, "correctness": 0}


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAPTOR vs Flat retrieval quality")
    parser.add_argument("--document", default=os.path.join(os.path.dirname(__file__), "..", "demo", "sample.txt"))
    parser.add_argument("--embedding", choices=["openai", "sbert"], default="sbert")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import anthropic
    from raptor.models.embedding_models import SBertEmbeddingModel, OpenAIEmbeddingModel
    from raptor.models.anthropic_models import ClaudeQAModel
    from raptor.raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    from raptor.retrieval.flat_retriever import FaissRetriever, FaissRetrieverConfig

    client = anthropic.Anthropic()

    with open(args.document, "r") as f:
        document = f.read()

    emb = SBertEmbeddingModel() if args.embedding == "sbert" else OpenAIEmbeddingModel()
    qa = ClaudeQAModel()

    # ── Build RAPTOR tree ────────────────────────────────────────────────
    print("=" * 70)
    print("  RAPTOR vs Flat RAG — Retrieval Quality Evaluation")
    print("=" * 70)
    print(f"  Document: {args.document} ({len(document)} chars)")
    print(f"  Embedding: {args.embedding}")
    print(f"  Questions: {len(CINDERELLA_QA)}")
    print()

    print("Building RAPTOR tree...")
    t0 = time.time()
    ra_config = RetrievalAugmentationConfig(embedding_model=emb, qa_model=qa)
    ra = RetrievalAugmentation(config=ra_config)
    ra.add_documents(document, force=True)
    raptor_build = time.time() - t0
    print(f"  Tree built in {raptor_build:.1f}s — {ra.tree.num_layers} layers, {len(ra.tree.all_nodes)} nodes")

    print("Building flat FAISS index...")
    t0 = time.time()
    emb_name = list(ra.tree_builder.embedding_models.keys())[0]
    faiss_config = FaissRetrieverConfig(
        embedding_model=emb, question_embedding_model=emb,
        use_top_k=True, top_k=5, embedding_model_string=emb_name,
    )
    faiss_ret = FaissRetriever(faiss_config)
    faiss_ret.build_from_leaf_nodes(list(ra.tree.leaf_nodes.values()))
    flat_build = time.time() - t0
    print(f"  Index built in {flat_build:.1f}s")
    print()

    # ── Run evaluation ───────────────────────────────────────────────────
    results = []
    raptor_scores = {"context": [], "relevance": [], "completeness": [], "correctness": []}
    flat_scores = {"context": [], "relevance": [], "completeness": [], "correctness": []}

    for i, qa_pair in enumerate(CINDERELLA_QA, 1):
        question = qa_pair["question"]
        key_facts = qa_pair["key_facts"]
        print(f"  Q{i}: {question}")

        # RAPTOR retrieval
        t0 = time.time()
        raptor_ctx, raptor_layers = ra.retrieve(
            question, top_k=5, max_tokens=3500,
            collapse_tree=True, return_layer_information=True,
        )
        raptor_retrieve_time = time.time() - t0
        try:
            raptor_answer = qa.answer_question(raptor_ctx, question)
        except Exception as e:
            logger.warning(f"QA failed for RAPTOR: {e}")
            raptor_answer = "(QA failed)"
        raptor_context_score = score_context(raptor_ctx, key_facts)
        try:
            raptor_judge = llm_judge(question, raptor_ctx, raptor_answer, client, "claude-sonnet-4-20250514")
        except Exception as e:
            logger.warning(f"LLM judge failed for RAPTOR: {e}")
            raptor_judge = {"relevance": 0, "completeness": 0, "correctness": 0}
        layers_hit = sorted(set(l["layer_number"] for l in raptor_layers))

        # Flat retrieval
        t0 = time.time()
        flat_ctx = faiss_ret.retrieve(question)
        flat_retrieve_time = time.time() - t0
        try:
            flat_answer = qa.answer_question(flat_ctx, question)
        except Exception as e:
            logger.warning(f"QA failed for flat: {e}")
            flat_answer = "(QA failed)"
        flat_context_score = score_context(flat_ctx, key_facts)
        try:
            flat_judge = llm_judge(question, flat_ctx, flat_answer, client, "claude-sonnet-4-20250514")
        except Exception as e:
            logger.warning(f"LLM judge failed for flat: {e}")
            flat_judge = {"relevance": 0, "completeness": 0, "correctness": 0}

        raptor_scores["context"].append(raptor_context_score)
        flat_scores["context"].append(flat_context_score)
        for k in ["relevance", "completeness", "correctness"]:
            raptor_scores[k].append(raptor_judge.get(k, 0))
            flat_scores[k].append(flat_judge.get(k, 0))

        print(f"      RAPTOR  context={raptor_context_score:.0%}  rel={raptor_judge.get('relevance',0)}  comp={raptor_judge.get('completeness',0)}  corr={raptor_judge.get('correctness',0)}  layers={layers_hit}  {raptor_retrieve_time:.3f}s")
        print(f"      Flat    context={flat_context_score:.0%}  rel={flat_judge.get('relevance',0)}  comp={flat_judge.get('completeness',0)}  corr={flat_judge.get('correctness',0)}  {flat_retrieve_time:.3f}s")
        print()

        results.append({
            "question": question,
            "raptor": {
                "context_score": raptor_context_score,
                "judge": raptor_judge,
                "retrieve_time": raptor_retrieve_time,
                "layers_hit": layers_hit,
                "answer_preview": raptor_answer[:100],
            },
            "flat": {
                "context_score": flat_context_score,
                "judge": flat_judge,
                "retrieve_time": flat_retrieve_time,
                "answer_preview": flat_answer[:100],
            },
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"                    RAPTOR      Flat RAG")
    print(f"  Context coverage  {avg(raptor_scores['context']):.0%}          {avg(flat_scores['context']):.0%}")
    print(f"  Relevance         {avg(raptor_scores['relevance']):.1f}/10      {avg(flat_scores['relevance']):.1f}/10")
    print(f"  Completeness      {avg(raptor_scores['completeness']):.1f}/10      {avg(flat_scores['completeness']):.1f}/10")
    print(f"  Correctness       {avg(raptor_scores['correctness']):.1f}/10      {avg(flat_scores['correctness']):.1f}/10")
    print()

    raptor_avg = (avg(raptor_scores['relevance']) + avg(raptor_scores['completeness']) + avg(raptor_scores['correctness'])) / 3
    flat_avg = (avg(flat_scores['relevance']) + avg(flat_scores['completeness']) + avg(flat_scores['correctness'])) / 3
    print(f"  Overall score     {raptor_avg:.1f}/10      {flat_avg:.1f}/10")

    if raptor_avg > flat_avg:
        print(f"  RAPTOR wins by {raptor_avg - flat_avg:.1f} points")
    elif flat_avg > raptor_avg:
        print(f"  Flat RAG wins by {flat_avg - raptor_avg:.1f} points")
    else:
        print(f"  Tie!")
    print()

    if args.output:
        output_data = {
            "document": args.document,
            "embedding": args.embedding,
            "build_times": {"raptor": raptor_build, "flat": flat_build},
            "results": results,
            "summary": {
                "raptor": {k: avg(v) for k, v in raptor_scores.items()},
                "flat": {k: avg(v) for k, v in flat_scores.items()},
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()

"""
CLI entry point for running RAPTOR vs flat retrieval benchmarks.

Usage:
    python -m benchmarks.run_benchmark \
        --document demo/sample.txt \
        --questions "How did Cinderella reach her happy ending?" "Who helped Cinderella?" \
        --output results.txt
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def main():
    parser = argparse.ArgumentParser(
        description="RAPTOR vs Flat retrieval benchmark"
    )
    parser.add_argument(
        "--document", required=True, help="Path to the document text file"
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        required=True,
        help="Questions to benchmark",
    )
    parser.add_argument(
        "--embedding",
        choices=["openai", "sbert"],
        default="openai",
        help="Embedding model to use (default: openai)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path for results",
    )
    args = parser.parse_args()

    # Lazy imports so --help is fast
    from raptor.benchmark import RaptorBenchmark
    from raptor.models.embedding_models import OpenAIEmbeddingModel, SBertEmbeddingModel

    if args.embedding == "sbert":
        embedding_model = SBertEmbeddingModel()
    else:
        embedding_model = OpenAIEmbeddingModel()

    with open(args.document, "r") as f:
        document = f.read()

    benchmark = RaptorBenchmark(embedding_model=embedding_model)
    report = benchmark.run(document, args.questions, top_k=args.top_k)

    summary = report.summary()
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(summary)

    if args.output:
        with open(args.output, "w") as f:
            f.write(summary)
            f.write("\n\n--- Raw JSON ---\n")
            raw = []
            for r in report.results:
                raw.append({
                    "question": r.question,
                    "raptor_answer": r.raptor_answer,
                    "raptor_time": r.raptor_time,
                    "flat_answer": r.flat_answer,
                    "flat_time": r.flat_time,
                })
            f.write(json.dumps(raw, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()

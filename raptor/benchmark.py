import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models.base import BaseEmbeddingModel, BaseQAModel
from .models.anthropic_models import ClaudeQAModel
from .models.embedding_models import OpenAIEmbeddingModel
from .raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from .retrieval.flat_retriever import FaissRetriever, FaissRetrieverConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    question: str
    raptor_answer: str
    raptor_time: float
    flat_answer: str
    flat_time: float

    def summary(self) -> str:
        return (
            f"Q: {self.question}\n"
            f"  RAPTOR ({self.raptor_time:.2f}s): {self.raptor_answer}\n"
            f"  Flat   ({self.flat_time:.2f}s): {self.flat_answer}\n"
        )


@dataclass
class BenchmarkReport:
    results: List[BenchmarkResult] = field(default_factory=list)
    tree_build_time: float = 0.0
    flat_build_time: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Build times — RAPTOR tree: {self.tree_build_time:.2f}s, "
            f"Flat index: {self.flat_build_time:.2f}s",
            "",
        ]
        for r in self.results:
            lines.append(r.summary())

        avg_raptor = sum(r.raptor_time for r in self.results) / max(len(self.results), 1)
        avg_flat = sum(r.flat_time for r in self.results) / max(len(self.results), 1)
        lines.append(
            f"Average query time — RAPTOR: {avg_raptor:.2f}s, Flat: {avg_flat:.2f}s"
        )
        return "\n".join(lines)


class RaptorBenchmark:
    """Compares RAPTOR tree retrieval vs flat FAISS retrieval on the same data."""

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        qa_model: Optional[BaseQAModel] = None,
    ):
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.qa_model = qa_model or ClaudeQAModel()

    def run(
        self,
        document: str,
        questions: List[str],
        top_k: int = 10,
        max_tokens: int = 3500,
    ) -> BenchmarkReport:
        report = BenchmarkReport()

        # --- Build RAPTOR tree ---
        logger.info("Building RAPTOR tree...")
        t0 = time.time()
        ra_config = RetrievalAugmentationConfig(
            embedding_model=self.embedding_model,
            qa_model=self.qa_model,
        )
        ra = RetrievalAugmentation(config=ra_config)
        ra.add_documents(document, force=True)
        report.tree_build_time = time.time() - t0
        logger.info(f"RAPTOR tree built in {report.tree_build_time:.2f}s")

        # --- Build flat FAISS index from the same leaf nodes ---
        logger.info("Building flat FAISS index...")
        t0 = time.time()
        faiss_config = FaissRetrieverConfig(
            embedding_model=self.embedding_model,
            question_embedding_model=self.embedding_model,
            use_top_k=True,
            top_k=top_k,
        )
        faiss_retriever = FaissRetriever(faiss_config)
        leaf_nodes = list(ra.tree.leaf_nodes.values())
        embedding_model_name = list(ra.tree_builder.embedding_models.keys())[0]
        faiss_config.embedding_model_string = embedding_model_name
        faiss_retriever.embedding_model_string = embedding_model_name
        faiss_retriever.build_from_leaf_nodes(leaf_nodes)
        report.flat_build_time = time.time() - t0
        logger.info(f"Flat index built in {report.flat_build_time:.2f}s")

        # --- Run queries ---
        for question in questions:
            logger.info(f"Querying: {question}")

            # RAPTOR retrieval + QA
            t0 = time.time()
            raptor_answer = ra.answer_question(
                question, top_k=top_k, max_tokens=max_tokens
            )
            raptor_time = time.time() - t0

            # Flat retrieval + QA
            t0 = time.time()
            flat_context = faiss_retriever.retrieve(question)
            flat_answer = self.qa_model.answer_question(flat_context, question)
            flat_time = time.time() - t0

            result = BenchmarkResult(
                question=question,
                raptor_answer=raptor_answer,
                raptor_time=raptor_time,
                flat_answer=flat_answer,
                flat_time=flat_time,
            )
            report.results.append(result)

        return report

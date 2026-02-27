<picture>
  <source media="(prefers-color-scheme: dark)" srcset="raptor_dark.png">
  <img alt="RAPTOR" src="raptor.jpg">
</picture>

## RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

RAPTOR builds a **recursive tree** from documents — clustering and summarizing chunks at multiple levels — then retrieves from the full hierarchy at query time. This gives the LLM both fine-grained details and high-level context that flat RAG misses.

> Based on the [original paper](https://arxiv.org/abs/2401.18059) by Sarthi et al. (ICLR 2024)

---

## Dashboard

Upload documents, ask questions, and see **RAPTOR vs flat RAG** answers side-by-side. Explore the tree structure with an interactive explorer and zoomable graph.

```bash
# Backend
cd dashboard/backend && pip install flask flask-cors && python app.py

# Frontend
cd dashboard/frontend && npm install && npm run dev
```

Open **http://localhost:5173**

| Upload & Stats | Side-by-side Results |
|---|---|
| ![home](docs/screenshots/dashboard-home.png) | ![results](docs/screenshots/query-results.png) |

| Tree Explorer | Graph View |
|---|---|
| ![explorer](docs/screenshots/tree-explorer.png) | ![graph](docs/screenshots/tree-graph.png) |

---

## Quick Start

```bash
git clone https://github.com/viplismism/raptor.git
cd raptor
pip install -e ".[all]"
```

Create a `.env` file (see `.env.example`):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...          # optional, for OpenAI embeddings
```

```python
from raptor import RetrievalAugmentation

RA = RetrievalAugmentation()

with open('demo/sample.txt', 'r') as file:
    RA.add_documents(file.read())

answer = RA.answer_question("How did Cinderella reach her happy ending?")
```

<details>
<summary><strong>Saving and loading trees</strong></summary>

```python
RA.save("demo/cinderella")
RA = RetrievalAugmentation(tree="demo/cinderella")
```
</details>

<details>
<summary><strong>Custom models</strong></summary>

Extend the base classes to plug in any model:

```python
from raptor import (
    BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel,
    RetrievalAugmentation, RetrievalAugmentationConfig,
)

class MySummarizer(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return "your summary"

class MyQA(BaseQAModel):
    def answer_question(self, context, question):
        return "your answer"

class MyEmbedding(BaseEmbeddingModel):
    def create_embedding(self, text):
        return [0.0] * 768

config = RetrievalAugmentationConfig(
    summarization_model=MySummarizer(),
    qa_model=MyQA(),
    embedding_model=MyEmbedding(),
)
RA = RetrievalAugmentation(config=config)
```
</details>

<details>
<summary><strong>Install extras</strong></summary>

| Extra       | What it adds                                  |
|-------------|-----------------------------------------------|
| `openai`    | OpenAI models for embeddings                  |
| `sbert`     | Sentence-Transformers embeddings              |
| `local`     | Local T5/UnifiedQA (torch + transformers)     |
| `faiss`     | FAISS flat retrieval baseline                 |
| `benchmark` | Benchmark CLI                                 |
| `all`       | Everything above                              |

```bash
pip install -e ".[sbert,faiss]"   # pick what you need
```
</details>

<details>
<summary><strong>Benchmarking</strong></summary>

```bash
python -m benchmarks.run_benchmark \
    --document demo/sample.txt \
    --questions "How did Cinderella reach her happy ending?"
```

```python
from raptor import RaptorBenchmark

report = RaptorBenchmark().run(text, ["How did Cinderella reach her happy ending?"])
print(report.summary())
```
</details>

---

## Acknowledgements

Based on the original [RAPTOR implementation](https://github.com/parthsarthi03/raptor) by Parth Sarthi et al. This fork adds modular code structure, a React dashboard, multi-file support, and benchmark tooling.

## Citation

```bibtex
@inproceedings{sarthi2024raptor,
    title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
    author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```

## License

MIT

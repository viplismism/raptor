<picture>
  <source media="(prefers-color-scheme: dark)" srcset="raptor_dark.png">
  <img alt="RAPTOR" src="raptor.jpg">
</picture>

## RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

**RAPTOR** introduces a novel approach to retrieval-augmented language models by constructing a recursive tree structure from documents. This allows for more efficient and context-aware information retrieval across large texts, addressing common limitations in traditional language models.

For detailed methodologies and implementations, refer to the original paper:

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/viplismism/raptor.git
cd raptor
pip install -e ".[all]"
```

### Minimal install (Anthropic + OpenAI embeddings only)

```bash
pip install -e ".[openai]"
```

### Optional extras

| Extra       | What it adds                                  |
|-------------|-----------------------------------------------|
| `openai`    | OpenAI models for embeddings and comparison   |
| `sbert`     | Sentence-Transformers embedding models        |
| `local`     | Local T5/UnifiedQA models (torch + transformers) |
| `faiss`     | FAISS flat retrieval baseline                 |
| `benchmark` | Benchmark CLI (`faiss` + `tqdm`)              |
| `all`       | Everything above                              |

## Configuration

Create a `.env` file in the project root (see `.env.example`):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...          # optional, for OpenAI embeddings
```

## Basic Usage

```python
from raptor import RetrievalAugmentation

# Defaults: Claude for summarization + QA, OpenAI for embeddings
RA = RetrievalAugmentation()

with open('demo/sample.txt', 'r') as file:
    text = file.read()

RA.add_documents(text)
```

### Answering Questions

```python
question = "How did Cinderella reach her happy ending?"
answer = RA.answer_question(question=question)
print("Answer:", answer)
```

### Saving and Loading Trees

```python
RA.save("demo/cinderella")

# Load later
RA = RetrievalAugmentation(tree="demo/cinderella")
answer = RA.answer_question(question=question)
```

## Custom Models

RAPTOR is designed to be flexible. Extend the base classes to use any model:

```python
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel

class MySummarizer(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return "your summary"

class MyQA(BaseQAModel):
    def answer_question(self, context, question):
        return "your answer"

class MyEmbedding(BaseEmbeddingModel):
    def create_embedding(self, text):
        return [0.0] * 768  # your embedding
```

Integrate with RAPTOR:

```python
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

config = RetrievalAugmentationConfig(
    summarization_model=MySummarizer(),
    qa_model=MyQA(),
    embedding_model=MyEmbedding(),
)
RA = RetrievalAugmentation(config=config)
```

## Benchmarking: RAPTOR vs Flat Retrieval

Compare RAPTOR's hierarchical tree retrieval against standard flat FAISS retrieval on the same data:

```bash
python -m benchmarks.run_benchmark \
    --document demo/sample.txt \
    --questions "How did Cinderella reach her happy ending?" "Who helped Cinderella?" \
    --output results.txt
```

Or from Python:

```python
from raptor import RaptorBenchmark

benchmark = RaptorBenchmark()
report = benchmark.run(text, ["How did Cinderella reach her happy ending?"])
print(report.summary())
```

Check out `notebooks/demo.ipynb` for interactive examples.

## Acknowledgements

This project is based on the original [RAPTOR implementation](https://github.com/parthsarthi03/raptor) by Parth Sarthi et al. The core tree-building and retrieval algorithm comes from their work. This fork adds a modular code structure, a React dashboard for visual comparison, multi-file support, and benchmark tooling.

## License

RAPTOR is released under the MIT License. See the LICENSE file in the repository for full details.

## Citation

If RAPTOR assists in your research, please cite the original paper:

```bibtex
@inproceedings{sarthi2024raptor,
    title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
    author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```

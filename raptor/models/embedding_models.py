import logging

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text: str):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text: str):
        return self.model.encode(text)

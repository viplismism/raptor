from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str):
        pass


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        pass


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context: str, question: str) -> str:
        pass

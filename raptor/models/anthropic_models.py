import logging

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..config import get_anthropic_api_key
from .base import BaseQAModel, BaseSummarizationModel

logger = logging.getLogger(__name__)


class ClaudeSummarizationModel(BaseSummarizationModel):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=get_anthropic_api_key())

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context: str, max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Write a summary of the following, including as many "
                        f"key details as possible: {context}"
                    ),
                }
            ],
        )
        return response.content[0].text


class ClaudeQAModel(BaseQAModel):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=get_anthropic_api_key())

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context: str, question: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Given Context: {context}\n\n"
                        f"Give the best full answer to the question: {question}"
                    ),
                }
            ],
        )
        return response.content[0].text.strip()

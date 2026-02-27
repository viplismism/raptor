import logging

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..config import get_openai_api_key
from .base import BaseQAModel, BaseSummarizationModel

logger = logging.getLogger(__name__)


class OpenAIChatSummarizationModel(BaseSummarizationModel):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=get_openai_api_key())

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context: str, max_tokens: int = 500) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Write a summary of the following, including as many "
                        f"key details as possible: {context}"
                    ),
                },
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class OpenAIChatQAModel(BaseQAModel):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=get_openai_api_key())

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context: str, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": (
                        f"Given Context: {context}\n\n"
                        f"Give the best full answer to the question: {question}"
                    ),
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

import openai
import os

from pydantic import BaseModel
from typing import Any, List, Optional


class Chat(BaseModel):
    model: str = "gpt-4"
    temperature: float = 0.0
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")

    def __call__(self, prompt: str) -> Any:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

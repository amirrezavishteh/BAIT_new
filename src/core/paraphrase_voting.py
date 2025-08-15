import os
from collections import Counter
from typing import List

from openai import OpenAI


class Paraphraser:
    """Generate paraphrases of a prompt using an auxiliary LLM."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def paraphrase(self, prompt: str, n: int = 3) -> List[str]:
        """Return ``n`` paraphrased prompts."""
        outputs: List[str] = []
        for _ in range(n):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": f"Paraphrase: {prompt}"}],
                    max_tokens=128,
                )
                outputs.append(resp.choices[0].message.content.strip())
            except Exception:
                break
        return outputs


def vote_outputs(outputs: List[str]) -> Counter:
    """Return a counter of outputs for simple majority voting."""
    return Counter(outputs)

from typing import List

from sentence_transformers import SentenceTransformer, util


class SemanticSimilarity:
    """Utility for computing semantic similarity between texts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def score(self, text_a: str, text_b: str) -> float:
        emb_a = self.model.encode(text_a, convert_to_tensor=True)
        emb_b = self.model.encode(text_b, convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b)
        return float(sim.item())

    def most_similar(self, target: str, candidates: List[str]) -> str:
        """Return the candidate text with the highest similarity to ``target``."""
        scores = [self.score(target, c) for c in candidates]
        return candidates[int(max(range(len(scores)), key=lambda i: scores[i]))]

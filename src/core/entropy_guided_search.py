from typing import Tuple

import torch


def compute_self_entropy(probs: torch.Tensor) -> float:
    """Return the self entropy of ``probs``.

    ``probs`` should be a 1-D tensor representing a probability distribution.
    """
    probs = probs / probs.sum()
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return entropy


def dynamic_top_k(logits: torch.Tensor, base_k: int, entropy: float, max_k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select tokens using a dynamic ``k`` based on entropy.

    A higher entropy value results in exploring more candidates, while a low
    entropy keeps the selection conservative. ``logits`` is a 1-D tensor of raw
    scores for the vocabulary.
    """
    k = base_k + int(entropy)
    k = max(base_k, min(max_k, k))
    topk = torch.topk(logits, k)
    return topk.values, topk.indices

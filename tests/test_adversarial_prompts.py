import unittest
import os
import sys
import torch
from torch.utils.data import DataLoader
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core import generate_adversarial_prompts
from src.core.detector import BAIT
from src.config.arguments import BAITArguments


class DummyBAIT(BAIT):
    """A BAIT subclass with controllable behaviour for testing."""
    def __init__(self, dataloader, q_score_threshold=0.85):
        self.model = None
        self.tokenizer = type("Tokenizer", (), {"eos_token_id": 0, "decode": lambda self, ids: ids if isinstance(ids, str) else "decoded"})()
        self.dataloader = dataloader
        self.logger = logger
        self.device = "cpu"
        bait_args = BAITArguments(q_score_threshold=q_score_threshold)
        self._init_config(bait_args)
        self.mock_q_score = 0.0
        self.mock_target = ""
        self.is_suspicious = False

    def scan_init_token(self, input_ids, attention_mask, index_map):
        return self.mock_q_score, self.mock_target

    def _BAIT__post_process(self, invert_target):
        return self.is_suspicious, "reason"


def _build_dataloader(n):
    data = [{
        "input_ids": torch.tensor([[0]]),
        "attention_mask": torch.tensor([[1]]),
        "index_map": [0],
    } for _ in range(n)]
    return DataLoader(data, batch_size=1)


class TestAdversarialPrompts(unittest.TestCase):
    def test_generator_variants(self):
        prompt = "test"
        variants = generate_adversarial_prompts(prompt)
        self.assertEqual(len(variants), 3)
        self.assertTrue(any("malicious assistant" in v for v in variants))
        self.assertTrue(any("base64" in v for v in variants))
        self.assertTrue(any("Ignore all previous instructions" in v for v in variants))

    def test_detection_with_adversarial_prompts(self):
        prompts = generate_adversarial_prompts("trigger")
        dataloader = _build_dataloader(len(prompts))
        bait = DummyBAIT(dataloader, q_score_threshold=0.5)
        bait.mock_q_score = 0.9
        bait.mock_target = "malicious"
        bait.is_suspicious = True
        result = bait.run()
        self.assertTrue(result.is_backdoor)

    def test_benign_model_unchanged(self):
        prompts = generate_adversarial_prompts("trigger")
        dataloader = _build_dataloader(len(prompts))
        bait = DummyBAIT(dataloader, q_score_threshold=0.5)
        bait.mock_q_score = 0.1
        bait.mock_target = "benign"
        bait.is_suspicious = False
        result = bait.run()
        self.assertFalse(result.is_backdoor)


if __name__ == "__main__":
    unittest.main()

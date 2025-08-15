import unittest
import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.core.detector import BAIT
from src.config.arguments import BAITArguments


class DummyBAIT(BAIT):
    """A BAIT subclass with controllable behaviour for testing."""
    def __init__(self, dataloader, q_score_threshold=0.85):
        # minimal initialization without external services
        self.model = None
        self.tokenizer = type("Tokenizer", (), {"eos_token_id": 0, "decode": lambda self, ids: ids if isinstance(ids, str) else "decoded"})()
        self.dataloader = dataloader
        self.logger = logger
        self.device = "cpu"
        bait_args = BAITArguments(q_score_threshold=q_score_threshold)
        self._init_config(bait_args)
        # fields set in tests
        self.mock_q_score = 0.0
        self.mock_target = ""
        self.is_suspicious = False

    def scan_init_token(self, input_ids, attention_mask, index_map):
        """Return predefined q-score and target."""
        return self.mock_q_score, self.mock_target

    def _BAIT__post_process(self, invert_target):
        """Override private post-process with predefined result."""
        return self.is_suspicious, "reason"


class TestDetector(unittest.TestCase):
    def _build_dataloader(self):
        data = [{
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.tensor([[1]]),
            "index_map": [0],
        }]
        return DataLoader(data, batch_size=1)

    def test_stable_softmax(self):
        dataloader = self._build_dataloader()
        bait = DummyBAIT(dataloader)
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        probs = bait.stable_softmax(logits, dim=-1)
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-6))

    def test_run_detects_backdoor(self):
        dataloader = self._build_dataloader()
        bait = DummyBAIT(dataloader, q_score_threshold=0.5)
        bait.mock_q_score = 0.9
        bait.mock_target = "malicious"
        bait.is_suspicious = True
        result = bait.run()
        self.assertTrue(result.is_backdoor)
        self.assertEqual(result.best_target.invert_target, "malicious")

    def test_run_handles_safe_model(self):
        dataloader = self._build_dataloader()
        bait = DummyBAIT(dataloader, q_score_threshold=0.5)
        bait.mock_q_score = 0.1
        bait.mock_target = "benign"
        bait.is_suspicious = False
        result = bait.run()
        self.assertFalse(result.is_backdoor)
        self.assertIsNone(result.best_target.invert_target)


if __name__ == "__main__":
    unittest.main()

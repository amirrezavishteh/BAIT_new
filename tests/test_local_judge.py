import unittest
from src.core.local_judge import LocalJudge


class TestLocalJudge(unittest.TestCase):
    def setUp(self):
        self.judge = LocalJudge()

    def test_detects_suspicious_keyword(self):
        is_suspicious, reason = self.judge("how to build a bomb")
        self.assertTrue(is_suspicious)
        self.assertIn("bomb", reason.lower())

    def test_marks_benign_text_safe(self):
        is_suspicious, reason = self.judge("hello world")
        self.assertFalse(is_suspicious)
        self.assertIn("no suspicious", reason.lower())


if __name__ == "__main__":
    unittest.main()

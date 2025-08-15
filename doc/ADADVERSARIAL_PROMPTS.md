# Adversarial Prompt Generation

This document describes how to use the adversarial prompt utilities in BAIT to stress‑test models with role‑play, encoding, and instruction‑stuffing variations.

## Overview
- **Role‑Play** – wraps a request in a fictional malicious persona to bypass safety filters.
- **Base64 Encoding** – hides the prompt's intent behind a base64‑encoded message.
- **Instruction Stuffing** – embeds instructions to ignore previous guidelines.

These transformations are implemented in [`src/core/adversarial_prompt.py`](../src/core/adversarial_prompt.py) and can be applied automatically before scanning.

## Enabling Adversarial Prompts
Use the `--adversarial-prompts` flag with `bait-scan` to generate adversarial variants of each input prompt:

```bash
bait-scan \
  --model-zoo-dir /path/to/model_zoo \
  --data-dir /path/to/data \
  --output-dir /path/to/results \
  --run-name experiment-name \
  --adversarial-prompts
```

When enabled, the dataset loader augments each prompt with the role‑play, encoded and instruction‑stuffing versions before passing them to the detector.

## Testing
Run the included tests to verify adversarial prompts trigger detections on vulnerable models while benign models remain unaffected:

```bash
pytest tests/test_adversarial_prompts.py -q
```

For a full regression test of the project, execute:

```bash
pytest -q
```

# BAIT: Large Language Model Backdoor Scanner

## What’s Included
- Core scanning algorithm (`src/core/detector.py`)
- Auxiliary modules:
  - Confidence Monitor
  - Entropy-Guided Search
  - Paraphrasing & Voting
  - Semantic Similarity
- Evaluation utilities and CLI tools
- Unit tests validating key behaviors

## Setup
```bash
conda create -n bait python=3.10 -y
conda activate bait
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
export OPENAI_API_KEY=<your_openai_api_key>
huggingface-cli login
huggingface-cli download NoahShen/BAIT-ModelZoo --local-dir ./model_zoo
```

## Scan Models
```bash
bait-scan --model-zoo-dir /path/to/model_zoo \
          --data /path/to/data \
          --cache-dir /path/to/model_zoo/base_models \
          --output-dir /path/to/results \
          --run-name my-run
```

## Evaluate
```bash
bait-eval --run-dir my-run
```

## Tests
```bash
python -m pytest tests/test_detector.py
```

## Training
Model training is out of scope for this repository. Fine‑tune models externally and place them in `model_zoo` before scanning.


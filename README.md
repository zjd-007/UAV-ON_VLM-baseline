# VLM Baseline for UAV-ON

This project fine-tunes `microsoft/Phi-3.5-vision-instruct` and Qwen2.5-VL
models as single-frame UAV object-goal navigation baselines. The repository
contains source code, experiment configurations, evaluation utilities, tests,
and small smoke-test samples. Full datasets, downloaded base models, training
checkpoints, logs, and generated evaluation results are intentionally stored
outside Git.

## Repository layout

- `src/vlm_baseline`: reusable navigation, prompting, memory, and safety logic
- `scripts`: data preparation, training, evaluation, and analysis entry points
- `configs`: LLaMA-Factory/DeepSpeed experiment configurations
- `eval`: evaluation implementation and base simulator settings
- `tests`: unit tests
- `data`: dataset registration, evaluation subsets, and small smoke samples

## Installation

Python 3.10 or newer is required. Install the package and its declared runtime
dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Training scripts also expect a compatible LLaMA-Factory installation in the
active environment. GPU-enabled PyTorch, CUDA, AirSim, and model versions must
match the target machine.

## External assets

The default project layout places the full UAV-ON dataset beside this
repository:

```text
Aerial-ObjectNav/
├── UAV-ON_dataset/
└── VLM-baseline/
```

Downloaded model weights should be placed under `models/`, or supplied through
the model path accepted by the relevant script/configuration. These assets are
ignored by Git. Do not replace the small tracked smoke samples with symlinks to
machine-local absolute paths.

## Data preparation

From the repository root:

```bash
./scripts/prepare_data.sh
```

Create a small smoke dataset with:

```bash
LIMIT=100 ./scripts/prepare_smoke_data.sh
```

## Training

The full Phi-3.5 training script defaults to GPUs `3,4,6,7`:

```bash
./scripts/train_phi35_lora.sh
```

Smoke training defaults to GPU `3`:

```bash
./scripts/train_phi35_lora_smoke.sh
```

The scripts use `scripts/llamafactory_phi3v.py` to register the Phi-3.5-Vision
multimodal template before invoking LLaMA-Factory. Review GPU IDs, model paths,
dataset paths, and output paths in the selected configuration before running a
job. Machine-specific orchestration scripts may also expose `ROOT`,
`DATASET_ROOT`, `CONDA_SH`, or `PYTHON_BIN` environment variables.

## Export

```bash
./scripts/export_phi35_lora.sh
```

## Evaluation

```bash
MODEL_PATH=/path/to/merged-or-adapter ./scripts/eval_phi35_uavon.sh
```

The evaluator uses the current RGB frame and target description. It maps model
output to the following commands:

`stop`, `forward 3m`, `turn left 30 degree`, `turn right 30 degree`,
`ascend 3m`, `descend 3m`.

## Tests

```bash
python -m pytest
```

## Version-control policy

Commit code, canonical configurations, tests, small data samples, and compact
result summaries. Keep full datasets, base-model weights, checkpoints,
optimizer states, frame archives, logs, and generated results in dedicated
dataset/model storage. Before making the repository public, add an appropriate
license and verify the redistribution terms for all sample data.

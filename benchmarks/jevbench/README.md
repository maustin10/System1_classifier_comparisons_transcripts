# JevBench comparison

This integration runs any TypeSafe-compatible `/v1/systemone` endpoint through
the 231 public JevBench decisions: 72 original, 48 easy, and 111 hard. It does
not include JevBench's sealed items and must not be presented as an official
leaderboard submission.

The pinned baseline in `results/jevbench/askatt` used JevBench revision
`1bcc55eb6c8cffde2306b3db03ede39b61c6152a`.

## 1. Start an endpoint

AskATT:

```bash
export ASKATT_DEVICE=cpu
export ASKATT_TOOL_LORA_CHECKPOINT=../Training_System1/checkpoints/pilot_2000_lora_r8_last8.pt
.venv/bin/python -m askatt_system1_api
```

Kev-4B on an Apple Silicon Mac with at least 32 GB unified memory:

```bash
git clone https://github.com/jaredpalmer/kev.git ../kev
cd ../kev
uv sync --extra serve
uv run --extra serve python -m kev.serve \
  --run jaredpalmer/kev-4b --host 127.0.0.1 --port 8009
```

Razorback OpenJev on Apple Silicon requires about 16 GB of **free** unified
memory; use a 32 GB or larger Mac:

```bash
git clone https://github.com/razorback16/openjev.git ../razorback-openjev
cd ../razorback-openjev
python3 -m venv .venv
.venv/bin/pip install -e '.[mlx]'
OPENJEV_BACKEND=mlx .venv/bin/python -m openjev
```

The M2 Pro 16 GB feasibility smoke test loaded the model but timed out on its
first decision after 120.13 seconds under severe swap pressure. Do not treat
that record as a quality score. It is preserved at
`results/jevbench/openjev-diffusion/smoke-m2-pro-16gb.jsonl`; use a 32 GB+
Mac or 24 GB+ NVIDIA GPU for the actual suite.

## 2. Run a cohort

Clone JevBench and replace the endpoint, model, output directory, and cohort as
needed:

```bash
git clone https://github.com/fstandhartinger/jevbench.git ../jevbench
cd ../jevbench
python3 -m jevbench.cli run \
  --tasks datasets/public/original.jsonl \
  --adapter typesafe \
  --endpoint http://127.0.0.1:8000 \
  --model askatt-gliclass-modern-large-v3.0 \
  --key-env '' \
  --results ../System1_classifier_comparisons_transcripts/results/jevbench/askatt/original.jsonl \
  --raw-dir ../System1_classifier_comparisons_transcripts/results/jevbench/askatt/raw-original \
  --reserve-usd 0 \
  --cost-basis local_hardware
```

Repeat with `easy.jsonl` and `hard.jsonl`. Use port `8009` and model
`kev-latest` for Kev; use port `8080` and model `openjev-latest` for Razorback
OpenJev.

Before launching all 231 decisions on new hardware, run one smoke case:

```bash
python3 -m jevbench.cli run \
  --tasks datasets/public/original.jsonl \
  --adapter typesafe \
  --endpoint http://127.0.0.1:8080 \
  --model openjev-latest \
  --key-env '' \
  --results /tmp/openjev-smoke.jsonl \
  --raw-dir /tmp/openjev-smoke-raw \
  --reserve-usd 0 \
  --cost-basis local_mlx_4bit \
  --limit 1
```

Proceed only if that request completes without sustained swap pressure. The
quality comparison requires completed `original`, `easy`, and `hard` files; a
timed-out smoke case must never be included as an incorrect prediction.

## 3. Summarize

```bash
python3 scripts/summarize_jevbench_runs.py \
  --jevbench-dir ../jevbench \
  --system askatt=results/jevbench/askatt \
  --system kev4b=results/jevbench/kev4b \
  --system openjev_diffusion=results/jevbench/openjev-diffusion \
  --output results/jevbench/summary.json
```

## Accelerator round

Use an A100/H100 for the two 27B controls and the vLLM DiffusionGemma path.
The same JevBench commands above apply because all servers expose the same
endpoint shape.

Recommended systems:

1. `openjev/openjev` FP8, model alias `openjev`, on an 80 GB accelerator.
2. `Zefan-Cai/Open-Jev-27B-v1.1` using its pinned server implementation.
3. Razorback OpenJev with `nvidia/diffusiongemma-26B-A4B-it-NVFP4` and its
   pinned vLLM container on a GPU with at least 24 GB.

Keep batch size, request order, warm-up, and serial/concurrent measurement modes
identical. Record the GPU, quantization, model and repository revisions, state
tokens, question count, p50/p95 latency, and decisions per second.

## Move the run to an M5 Mac

Use an M5 with at least 32 GB unified memory. Clone or copy this repository,
but create fresh virtual environments on the destination because environments
contain machine-specific paths. The model weights are intentionally excluded
from Git.

Recommended destination layout:

```text
work/
  System1_classifier_comparisons_transcripts/
  jevbench/
  razorback-openjev/
  kev/
```

The reproducible path is to clone the four repositories on the M5 and let each
runtime download its weights. From `work/`:

```bash
git clone https://github.com/maustin10/System1_classifier_comparisons_transcripts.git
git clone https://github.com/fstandhartinger/jevbench.git
git clone https://github.com/razorback16/openjev.git razorback-openjev
git clone https://github.com/jaredpalmer/kev.git
```

Pin JevBench and Razorback to the revisions used for the first feasibility
round before creating their environments:

```bash
git -C jevbench checkout 1bcc55eb6c8cffde2306b3db03ede39b61c6152a
git -C razorback-openjev checkout 6900d153b9052417135729a85137e2d5f782add1
```

Do **not** move `.venv`, `.codex-*`, `node_modules`, raw API caches, or any
temporary `/private/tmp` clones. To avoid downloading weights again, transfer
the complete `~/.cache/huggingface/hub` and `~/.cache/huggingface/xet`
directories with `rsync` while no model process is running; copying only a
single snapshot directory can leave missing content-addressed blobs. Otherwise,
allow the launch commands above to download fresh, validated caches.

Start Razorback first and run the one-case smoke command in section 2. Check
Activity Monitor memory pressure and confirm that the request completes before
running all three cohorts. Repeat with Kev on port 8009. Keep the Razorback and
Kev runs separate so each model has the full unified-memory budget.

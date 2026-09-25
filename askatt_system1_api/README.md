# AskATT System1 API

`Askatt_system1_api` exposes a local GLiClass checkpoint behind the same endpoint and JSON shapes used by the TypeSafe System One API for **Noul** and **Choice** questions.

It is a compatibility layer, not a reimplementation of JEV. Probability calibration, confidence semantics, context limits, and model quality are GLiClass/AskATT behavior.

See the [standalone appendix](../report/appendix-askatt-system1-api.md) for the architecture, compatibility boundary, persona-specific deployment guidance, cost formulas, and test evidence.

## Install

Create a virtual environment that can reuse an existing PyTorch installation:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r askatt_system1_api/requirements.txt
```

Download `knowledgator/gliclass-modern-large-v3.0` into:

```text
models/gliclass-modern-large-v3.0
```

With the Hugging Face CLI:

```bash
.venv/bin/pip install -U huggingface_hub
.venv/bin/hf download knowledgator/gliclass-modern-large-v3.0 \
  --local-dir models/gliclass-modern-large-v3.0
```

Model weights are excluded by `.gitignore`.

## Load the model on a hosted NVIDIA GPU

The model is loaded locally from `ASKATT_GLICLASS_MODEL`; the service does not download weights at request time. `ASKATT_DEVICE=cuda` causes the engine to move the checkpoint onto the NVIDIA GPU when the first inference request arrives.

On the GPU host, first confirm that the driver is visible:

```bash
nvidia-smi
```

Create the environment and install a CUDA-enabled PyTorch build. This example uses the CUDA 12.4 wheel index; select the PyTorch/CUDA combination that matches the host driver and base image:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r askatt_system1_api/requirements.txt
```

Verify that PyTorch sees the accelerator before starting the API:

```bash
.venv/bin/python - <<'PY'
import torch

print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY
```

Download or copy the model onto persistent local storage, then start one API process for the GPU:

```bash
export ASKATT_GLICLASS_MODEL=/opt/askatt/models/gliclass-modern-large-v3.0
export ASKATT_DEVICE=cuda
export ASKATT_LABELS_PER_PASS=64
export ASKATT_CHOICE_INSTRUCTION_MODE=state
export ASKATT_SYSTEM1_API_KEY='read-this-from-a-secret-manager'
export ASKATT_HOST=0.0.0.0
export ASKATT_PORT=8000

cd /opt/askatt/System1_classifier_comparisons_transcripts
/opt/askatt/System1_classifier_comparisons_transcripts/.venv/bin/python -m askatt_system1_api
```

The GLiClass checkpoint is loaded lazily. The first real inference therefore includes model-loading and GPU-initialization time. Send a warm-up request before registering the instance with a load balancer:

```bash
curl -i http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer ${ASKATT_SYSTEM1_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "askatt-gliclass-modern-large-v3.0",
    "state": "The caller reports an unfamiliar charge.",
    "questions": {
      "unknown_charge": {
        "type": "noul",
        "instructions": "Is an unknown charge reported?",
        "criteria": {
          "true": "An unfamiliar or unexplained charge is explicitly reported.",
          "false": "No unfamiliar charge is reported."
        }
      }
    }
  }'
```

Check that the response is HTTP 200, `answers.unknown_charge.noul` is between zero and one, and `X-AskATT-Truncated` is `false`.

## Run

```bash
export ASKATT_GLICLASS_MODEL=models/gliclass-modern-large-v3.0
export ASKATT_DEVICE=cpu          # use cuda on an NVIDIA deployment
export ASKATT_CHOICE_INSTRUCTION_MODE=state
# Optional: export ASKATT_SYSTEM1_API_KEY=local-secret
.venv/bin/python -m askatt_system1_api
```

The compatible endpoint is:

```text
POST http://127.0.0.1:8000/v1/systemone
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## Deployment strategies

| Strategy | Best fit | Operational shape |
|---|---|---|
| Local CPU | Development, contract tests, and small experiments | Run the current Uvicorn process with `ASKATT_DEVICE=cpu` |
| Single GPU VM | First production pilot or a steady departmental workload | One API process and one model copy per GPU, behind an HTTPS load balancer |
| Shared GPU platform | Multiple application teams with complementary demand | Pool traffic across GPU-backed replicas to improve utilization; apply tenant quotas and routing centrally |
| Kubernetes GPU service | Horizontal scale, rolling deployment, and multi-zone operation | One GPU-requesting pod per replica; mount the model read-only and scale replicas, not Uvicorn workers |
| Hybrid local plus hosted fallback | Bursty demand or workloads with different quality/context needs | Route steady validated traffic locally and overflow, failures, or unsupported cases to JEV or another managed service |

### Recommended initial production deployment

The fastest production path is a Linux NVIDIA GPU VM:

1. Attach persistent storage containing the model checkpoint.
2. Install the NVIDIA driver and a compatible CUDA-enabled PyTorch build.
3. Run one `python -m askatt_system1_api` process for each assigned GPU.
4. Store `ASKATT_SYSTEM1_API_KEY` in the platform secret manager.
5. Put TLS, authentication, rate limiting, and request-size controls at an API gateway or load balancer.
6. Warm the model with an inference request before admitting traffic.
7. Monitor latency, queueing, errors, input tokens, forward passes, compiled labels, and truncation.

For a VM managed by `systemd`, an example unit is:

```ini
[Unit]
Description=AskATT System1 API
After=network-online.target

[Service]
Type=simple
User=askatt
Group=askatt
WorkingDirectory=/opt/askatt/System1_classifier_comparisons_transcripts
EnvironmentFile=/etc/askatt-system1.env
ExecStart=/opt/askatt/System1_classifier_comparisons_transcripts/.venv/bin/python -m askatt_system1_api
Restart=always
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Save the unit as `/etc/systemd/system/askatt-system1.service`.

Keep the environment file readable only by the service account:

```bash
sudo install -m 600 /dev/null /etc/askatt-system1.env
sudoedit /etc/askatt-system1.env
```

Example contents:

```text
ASKATT_GLICLASS_MODEL=/opt/askatt/models/gliclass-modern-large-v3.0
ASKATT_DEVICE=cuda
ASKATT_LABELS_PER_PASS=64
ASKATT_CHOICE_INSTRUCTION_MODE=state
ASKATT_SYSTEM1_API_KEY=replace-with-secret-manager-material
ASKATT_HOST=0.0.0.0
ASKATT_PORT=8000
```

Then enable the service:

```bash
sudo chmod 600 /etc/askatt-system1.env
sudo systemctl daemon-reload
sudo systemctl enable --now askatt-system1
```

### Container and Kubernetes deployment

This repository does not currently include a production Dockerfile or Kubernetes manifests. When containerizing it:

- Start from an NVIDIA CUDA/PyTorch runtime compatible with the host driver.
- Install `askatt_system1_api/requirements.txt` and copy the application package.
- Mount the model directory read-only instead of downloading it for every container start.
- Set `ASKATT_DEVICE=cuda` and request exactly one GPU per replica, for example `nvidia.com/gpu: 1` in Kubernetes.
- Run one Uvicorn process per GPU-backed container. Multiple workers create multiple model copies and compete for the same GPU memory.
- Scale horizontally by adding GPU replicas behind a service or inference gateway.
- Use rolling or canary deployment with a pinned model directory/version.

The current `/healthz` endpoint is a process liveness check; it reports the configured model name but does not force the lazy model load. Do not use it alone as production readiness. Add a readiness probe backed by a completed warm-up inference, or keep the pod out of service until the warm-up request succeeds.

### Capacity, latency, and scaling cautions

- The engine uses a process-local lock around inference. Requests in one process are serialized; dynamic batching is not implemented.
- Use one process per GPU and add replicas for concurrency. Do not increase Uvicorn worker count blindly.
- Interactive services should retain GPU headroom and scale against queue depth plus p95/p99 latency, rather than targeting continuous 100% utilization.
- Offline batch jobs can run closer to saturation.
- More than `ASKATT_LABELS_PER_PASS` labels in a scoring group causes another pass and repeats that group's state context.
- Watch `X-AskATT-Forward-Passes`, `X-AskATT-Compiled-Labels`, `X-AskATT-Inference-Ms`, and `X-AskATT-Truncated` in production telemetry.
- The configured GLiClass checkpoint has a much smaller context window than JEV. Treat any `X-AskATT-Truncated: true` response as a policy event, not a normal success.
- Keep a hosted fallback until local quality, capacity, failover, and tail latency have been demonstrated with representative production traffic.

## Mixed Choice and Noul example

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "The customer says an unknown fee appeared and needs billing help.",
    "questions": {
      "unknown_charge": {
        "type": "noul",
        "instructions": "Is an unknown charge reported?",
        "criteria": {
          "true": "An unfamiliar or unexplained charge is explicit",
          "false": "No unfamiliar charge is reported"
        }
      },
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, credits, or unknown charges",
          "technical": "Service failures, devices, or integrations",
          "sales": "Upgrades, pricing, or new products"
        }
      }
    }
  }'
```

The response preserves the JEV field names:

```json
{
  "model": "askatt-gliclass-modern-large-v3.0",
  "answers": {
    "unknown_charge": {"type": "noul", "noul": 0.91},
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.82, "technical": 0.11, "sales": 0.07},
      "confidence": 0.48
    }
  },
  "usage": {"input_tokens": 140, "output_tokens": 0}
}
```

For conventional binary Choice options (`present/absent`, `yes/no`, or `true/false`), one positive GLiClass score is returned with its complement. Other Choice questions append their instructions to the state, score all of that question's option definitions together, and apply a within-question softmax to the log-odds. Noul uses the positive criterion when supplied; the negative criterion remains part of request validation but is not a separate label. `confidence` is normalized distribution concentration, not TypeSafe's proprietary confidence calculation.

## Compatibility and cost behavior

- Accepts string, object, or array state.
- Supports up to 64 questions per request.
- Supports Noul and Choice; Score is intentionally rejected.
- Runs compiled labels in groups of `ASKATT_LABELS_PER_PASS` (default 64).
- For general Choice, the default `ASKATT_CHOICE_INSTRUCTION_MODE=state` appends the question instructions to the state and scores that question's options together. This preserves question meaning, but repeats the state once per Choice question.
- Noul and conventional binary Choice labels still share a pass. More than `ASKATT_LABELS_PER_PASS` labels within any group adds passes.
- The legacy `ASKATT_CHOICE_INSTRUCTION_MODE=discard` is retained only for reproducibility; it is faster but ignores general-Choice instructions.
- Returns inference time, pass count, compiled-label count, and truncation status in `X-AskATT-*` headers.
- Returns the active question-handling strategy in `X-AskATT-Choice-Instruction-Mode`.
- Reports serialized GLiClass input tokens, not JEV billable tokens.
- Does not claim JEV probability calibration or internal architecture compatibility.

## Verified results

The locked test contains 150 transcripts and 27 attributes per transcript, or 4,050 binary decisions. Both Noul at 0.50 and conventional binary Choice reproduced the underlying raw GLiClass result:

| Accuracy | Precision | Recall | F1 | Exact transcript match |
|---:|---:|---:|---:|---:|
| 94.10% | 83.89% | 96.07% | 89.57% | 16.67% |

All 27 labels fit in one forward pass and no rows were truncated. Local sequential CPU timings varied substantially across repeated runs even though the compiled work was identical, so they are retained in the machine-readable result files only as implementation observations. Use a controlled GPU load test for capacity or end-to-end production latency decisions.

The transcript's Choice form is binary (`present/absent`) and intentionally reproduces the same underlying score as Noul. A separate multiclass benchmark tests four true Choice questions per state: department, urgency, sentiment, and next action. On 24 synthetic conversations (96 decisions), the fixed adapter reached **88.54% accuracy** in declared option order and **91.67%** with every option list reversed. Exact four-answer state accuracy was **54.17%**, and **9.38%** of answers changed under option reversal. Department reached 83.33%, urgency 87.50%, sentiment 95.83%, and next action 87.50%. Treat Choice quality as question-specific and require held-out domain validation plus option-order permutation tests.

The fix places each Choice question in the state context while retaining the option-definition labels the checkpoint already understands. This raised accuracy from the legacy **77.08%** to **88.54%**, at the cost of four state-processing passes instead of one for this four-question suite. Simply inserting the question into every label did not work: a fully structured encoding fell to **47.92%**, and a nine-format label sweep peaked at **75.00%**. Fine-tuning on a composite question/option grammar remains the path to recovering cross-question shared-state efficiency without discarding question meaning.

A synthetic cardinality stress test selected an explicit destination correctly at 2, 4, 8, 16, 32, and 65 options. The 65-option cases required two forward passes because they crossed the configured 64-label boundary. This checks mechanics, not real-world semantic accuracy.

On the same 96 semantic Choice decisions, TypeSafe.ai JEV reached 96.88% accuracy, fixed AskATT reached 88.54%, and zero-shot `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` reached 59.38%. JEV was perfect on department and next-action selection. AskATT materially exceeded ModernBERT on department, urgency, and next action, while ModernBERT remained perfectly invariant to option reversal. This is a comparison with pairwise zero-shot ModernBERT, not the fixed-taxonomy trained-head variant. See the walkthrough for paired results, timing boundaries, and limitations.

See [the benchmark walkthrough](../report/askatt-system1-benchmark-results.md) for request/response examples, full metrics, caveats, and interpretation.

Run the tests and benchmark with:

```bash
.venv/bin/python -m pytest -q tests/test_askatt_system1_api.py
.venv/bin/python -m pytest -q tests/test_askatt_choice_benchmark.py
ASKATT_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -q tests/test_askatt_system1_model.py
.venv/bin/python scripts/benchmark_askatt_system1_api.py
.venv/bin/python scripts/benchmark_askatt_choice.py
.venv/bin/python scripts/benchmark_askatt_label_formats.py
.venv/bin/python scripts/benchmark_askatt_question_in_state.py
.venv/bin/python scripts/benchmark_modernbert_choice.py
.venv/bin/python scripts/benchmark_jev_choice.py
```

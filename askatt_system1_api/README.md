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

Model weights are excluded by `.gitignore`.

## Run

```bash
export ASKATT_GLICLASS_MODEL=models/gliclass-modern-large-v3.0
export ASKATT_DEVICE=cpu          # use cuda on an NVIDIA deployment
# Optional: export ASKATT_SYSTEM1_API_KEY=local-secret
.venv/bin/python -m askatt_system1_api
```

The compatible endpoint is:

```text
POST http://127.0.0.1:8000/v1/systemone
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

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

For conventional binary Choice options (`present/absent`, `yes/no`, or `true/false`), one positive GLiClass score is returned with its complement. Other Choice questions apply a within-question softmax to the log-odds of GLiClass option scores. Noul uses the positive criterion when supplied; the negative criterion remains part of request validation but is not a separate label. `confidence` is normalized distribution concentration, not TypeSafe's proprietary confidence calculation.

## Compatibility and cost behavior

- Accepts string, object, or array state.
- Supports up to 64 questions per request.
- Supports Noul and Choice; Score is intentionally rejected.
- Runs compiled labels in groups of `ASKATT_LABELS_PER_PASS` (default 64).
- Each group is one GLiClass shared-state forward pass. More groups repeat the state.
- Returns inference time, pass count, compiled-label count, and truncation status in `X-AskATT-*` headers.
- Reports serialized GLiClass input tokens, not JEV billable tokens.
- Does not claim JEV probability calibration or internal architecture compatibility.

## Verified results

The locked test contains 150 transcripts and 27 attributes per transcript, or 4,050 binary decisions. Both Noul at 0.50 and conventional binary Choice reproduced the underlying raw GLiClass result:

| Accuracy | Precision | Recall | F1 | Exact transcript match |
|---:|---:|---:|---:|---:|
| 94.10% | 83.89% | 96.07% | 89.57% | 16.67% |

All 27 labels fit in one forward pass and no rows were truncated. The local sequential CPU run processed 1.91 states/s for Noul and 1.81 states/s for Choice. These are implementation timings, not H100 or end-to-end production latency measurements.

Run the tests and benchmark with:

```bash
.venv/bin/python -m pytest -q tests/test_askatt_system1_api.py
ASKATT_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -q tests/test_askatt_system1_model.py
.venv/bin/python scripts/benchmark_askatt_system1_api.py
```

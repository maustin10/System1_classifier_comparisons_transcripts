# AskATT System1 hybrid deployment versus TypeSafe.ai JEV

## Deployment policy tested

The AskATT API now keeps two independently loaded GLiClass views:

- **Base encoder:** general semantic Choice and transcript Noul questions.
- **Tool-routing LoRA:** tool selection and no-tool detection only.

The API selects a profile per question. Auto-routing uses strong tool/function wording in the question ID, instructions, or criteria. Applications can override it with `"routing_profile": "base"` or `"routing_profile": "tool_routing"`. The response exposes the selected profile and engine under `askatt_routing`, and the HTTP response reports pass counts in `X-AskATT-Routing-Profiles`.

This prevents the LoRA adapter's semantic-choice regression from affecting unrelated classification traffic while preserving its routing improvement.

## Matched results, including the untouched base baseline

| Benchmark | AskATT base only | AskATT hybrid | JEV | Hybrid lift vs. base | Hybrid gap vs. JEV |
|---|---:|---:|---:|---:|---:|
| Semantic Choice, 96 decisions | 88.54% | 88.54% | **96.88%** | 0.00 pp | -8.33 pp |
| BFCL-derived routing, 30 cases | 56.67% | 83.33% | **96.67%** | **+26.67 pp** | -13.33 pp |
| Positive tool selection | 85.00% | 85.00% | **100.00%** | 0.00 pp | -15.00 pp |
| No-tool recall | 0.00% | 80.00% | **90.00%** | **+80.00 pp** | -10.00 pp |
| Transcript Noul accuracy, validation-calibrated | 96.02% | 96.02% | **99.90%** | 0.00 pp | -3.88 pp |
| Transcript Noul precision, validation-calibrated | 89.20% | 89.20% | **99.63%** | 0.00 pp | -10.43 pp |
| Transcript Noul recall, validation-calibrated | 96.63% | 96.63% | **100.00%** | 0.00 pp | -3.37 pp |
| Transcript Noul F1, validation-calibrated | 92.76% | 92.76% | **99.81%** | 0.00 pp | -7.05 pp |
| Transcript exact-row match | 21.33% | 21.33% | **97.33%** | 0.00 pp | -76.00 pp |

The base-only BFCL result was rerun through the same production API
serialization with `routing_profile: base`. The LoRA specialist did not change
positive tool selection, but it fixed 8 of the 10 no-tool cases, lifting overall
routing accuracy by 26.67 percentage points. Semantic Choice and transcript Noul
are intentionally identical in the base and hybrid columns because the hybrid
policy routes those workloads to the untouched base encoder.

Routing verification from the benchmark artifact:

- 96/96 semantic decisions used `base`.
- 8,100/8,100 transcript attribute decisions used `base`.
- 30/30 BFCL cases used `tool_routing`.
- The separate BFCL baseline sent 30/30 cases to `base`.

The policy therefore worked as designed, but JEV remained materially more accurate on every matched suite.

## JEV provenance

- Semantic Choice was refreshed against the live JEV API on 2026-09-26: 93/96 correct. The reversed-option run was also refreshed; its flip rate was 2.08%.
- BFCL routing was refreshed live on 2026-09-26: 29/30 correct, 100% positive tool selection, and 90% no-tool recall.
- Transcript Noul uses the existing 300-call live JEV validation/test run with the same structured speaker state and strict per-attribute criteria. Thresholds were selected from validation only.

Raw live API responses are retained locally but intentionally hidden from normal repository views when their filenames begin with a dot.

## Latency observation—not an equivalent hardware benchmark

| Workload | AskATT local CPU | JEV hosted API |
|---|---:|---:|
| Semantic Choice | 1.22 s/state | 0.236 s/API call |
| BFCL routing | 0.586 s/case | 0.188 s median/case |
| Transcript Noul | 0.811 s/state | 0.293 s/state |

These numbers include different environments: AskATT ran sequentially on a local Apple CPU; JEV includes hosted inference, network, and response parsing. They demonstrate the current development experience, not H100-equivalent throughput.

## Noul boundary finding

JEV accepts both true and false criteria. AskATT's production default remains `positive`, meaning GLiClass scores the true criterion and returns that independent probability. A tested `paired` mode scores both criteria and normalizes them, but it reduced calibrated transcript F1 to 61.90% because this checkpoint was not trained to treat true and false descriptions as a mutually exclusive pair. Paired mode is therefore experimental and must be explicitly enabled with `ASKATT_NOUL_BOUNDARY_MODE=paired`.

## Operational implications

1. Use the hybrid API when self-hosting economics, privacy, or customization justify the measured quality gap.
2. Keep JEV as the quality reference and hosted fallback, particularly for strict transcript extraction and broad semantic Choice.
3. Require explicit `routing_profile` in production tool schemas when question wording may not contain a strong tool marker.
4. The current implementation loads two full base-model instances. This is simple and safe but approximately doubles model memory. A future adapter hot-swap or merged-adapter pool could reduce memory after concurrency behavior is validated.
5. Do not enable paired Noul boundaries by default without retraining on complementary criteria.

## Reproduce

```bash
.venv/bin/python scripts/benchmark_askatt_hybrid_vs_jev.py

# Add or refresh only the untouched base BFCL comparison
.venv/bin/python scripts/benchmark_askatt_hybrid_vs_jev.py --only-baseline-bfcl
```

Refresh the two smaller live JEV suites using the commands documented in their benchmark scripts. The transcript JEV run requires 300 hosted calls and is intentionally not refreshed by the hybrid runner.

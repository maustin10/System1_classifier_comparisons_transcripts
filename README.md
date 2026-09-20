# System1 classifier comparisons for customer-care transcripts

This repository represents an independent comparison of open encoder classifiers, TypeSafe.ai JEV, and frontier LLMs on the same synthetic customer-care transcript classification task.

**Author: Mark Austin**

The benchmark contains 1,000 synthetic conversations, each labeled for 27 binary customer-care attributes. The fixed split is 700 training, 150 validation, and 150 locked test conversations. All quality numbers below are calculated only on the locked test set: 4,050 binary decisions.

> This is a controlled synthetic benchmark, not a claim of production accuracy. The conversations use explicit, template-driven evidence and do not represent the ambiguity, distribution shift, or annotation disagreement of live calls.

## Executive decision pack

The first decision is whether the taxonomy is fixed. The paired maps use identical state-length, question-count, and utilization grids so the only difference is whether trained ModernBERT heads are eligible.

![Lowest-cost approach when the taxonomy is fixed and trained ModernBERT heads are eligible](charts/executive-fixed-taxonomy-winner-heatmap.png)

When the taxonomy is fixed, trained ModernBERT wins every tested cell at 25% or greater paid H100 utilization. At 10%, JEV wins many longer-state cells because its usage pricing avoids idle GPU cost.

![Lowest-cost runtime-question approach across state length, question count, and GPU utilization](charts/executive-runtime-winner-heatmap.png)

When questions must change at runtime, trained heads are ineligible. Each cell shows the lowest-cost eligible approach and estimated USD per 1,000 states. The evaluated runtime candidates are GLiClass Modern, TypeSafe.ai JEV Noul, ModernBERT pairwise NLI, GPT-5.6 Luna, and GPT-5.6 Sol.

![Cost advantage of the winning runtime-question approach](charts/executive-runtime-winner-confidence.png)

The confidence map shows the winner's percentage cost advantage over the runner-up. Low percentages identify close decisions where deployment simplicity, support, latency, or data-residency requirements may outweigh the modeled savings.

![Executive 2x2 matrix of economics and runtime-question flexibility](charts/executive-decision-matrix.png)

The 2x2 compresses the decision into two executive dimensions: deployment economics and whether questions can change at runtime. It keeps trained ModernBERT visible as the fixed-taxonomy cost leader while separating runtime-flexible GLiClass, JEV, and LLM options.

![Cost per 1,000 transcripts by GPU utilization](charts/executive-cost-vs-gpu-utilization.png)

The bar charts hold the reference workload at **6,000 state tokens × 25 questions** and show how self-hosted encoder cost changes at 10%, 25%, 50%, and 100% paid H100 utilization. JEV remains usage-priced and therefore unchanged across the four panels.

![Operating-model scorecard](charts/executive-operating-scorecard.png)

The named examples are:

- **Fixed trained encoder:** `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` embeddings plus 27 trained logistic heads. It is the lowest-cost high-quality option when the taxonomy is stable, but new questions require training.
- **Shared-state zero-shot encoder:** `knowledgator/gliclass-modern-large-v3.0`. It is the leading self-hosted runtime-label option in this test.
- **Hosted shared-state API:** **TypeSafe.ai JEV Noul**. It provides the strongest measured managed-service balance of quality and modeled cost.
- **Pairwise zero-shot encoder:** `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`. It is a transparent baseline, but repeats the state for every question.
- **Single-call LLMs:** GPT-5.6 Luna and Sol. They score extremely well, but their estimated serving costs are materially higher.

![Utilization decision bands for fixed and runtime taxonomies](charts/executive-utilization-decision-bands.png)

The crossover view holds the workload at **6,000 state tokens × 25 questions**. Self-hosted encoder costs scale as `full-utilization cost / utilization`; JEV remains usage-priced. In this scenario, **JEV becomes cheaper than trained ModernBERT below approximately 13.9% utilization and cheaper than GLiClass Modern below approximately 14.6%**. These are modeled crossovers, not measured H100 benchmarks or invoices.

## Key results

![F1 comparison across all transcript classifiers](charts/f1-comparison.png)

![Accuracy comparison across all transcript classifiers](charts/accuracy-comparison.png)

The principal findings are:

- JEV Noul with validation-refined criteria and validation-selected thresholds reached 99.90% accuracy and 99.81% micro F1, four errors across 4,050 decisions.
- GPT-5.6 Sol produced the highest measured result: 99.93% accuracy and 99.86% F1, three errors.
- GPT-5.6 Luna reached 99.75% accuracy and 99.53% F1.
- A frozen ModernBERT encoder with 27 supervised logistic heads reached 99.16% accuracy and 98.41% F1.
- Per-label threshold calibration improved unchanged zero-shot ModernBERT from 92.96% to 96.42% accuracy without training new weights.
- GLiClass Modern Large v3 became the strongest zero-shot open encoder tested: validation-only thresholds raised it from 94.10% accuracy / 89.57% F1 to 97.90% / 96.00%.
- GLiClass Large v3 reached 96.59% accuracy / 93.67% F1 after calibration. Its 512-token context required two label batches to retain every transcript token; GLiClass Modern Large fit all 27 labels in one pass with no truncation.
- DeBERTa-v3-large-zeroshot-v2.0-c did not beat ModernBERT zero-shot on this dataset and was approximately 2.47 times slower on the same CPU path.

## Estimated cost per 1,000 transcripts with chunking

![Estimated cost per 1,000 transcripts versus state length for ModernBERT, GLiClass, JEV, Sol, and Luna](charts/normalized-cost-vs-state-tokens.png)

**Chunking** splits a state that exceeds a model's native context into overlapping pieces, runs the same classifier on every piece, and combines the chunk-level probabilities into one transcript-level result. For these presence-style attributes, a maximum or calibrated noisy-OR is a plausible aggregator, but it must be validation-tuned because additional chunks can increase false positives.

This sensitivity analysis covers states up to **32k tokens** and uses NVIDIA H100s at **$5/GPU-hour**, 27 questions or labels, near-100% utilization, and 256 overlapping tokens between adjacent chunks. ModernBERT uses approximately 8,178 state tokens per chunk. GLiClass Modern Large uses the same ModernBERT-large H100 throughput proxy and fits approximately 7,895 state tokens plus all 27 labels in one pass. GLiClass Large has a 512-token context and used two label groups, leaving a 357-token state payload and repeating the state twice per chunk. JEV uses approximately 31,890 state tokens per request. At 32k this becomes five ModernBERT chunks, five GLiClass Modern chunks, 315 GLiClass Large chunks, or two JEV requests under the shared 256-token-overlap assumption. Sol and Luna remain one API call because their 1.05M-token contexts exceed this range.

| State tokens | ModernBERT trained | GLiClass Modern | GLiClass Large | ModernBERT zero-shot | JEV Noul | JEV Choice | Luna | Sol |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | $0.0008 | $0.0032 | $0.0063 | $0.0251 | $0.1117 | $0.1287 | $0.3751 | $6.6541 |
| 224 benchmark mean | $0.0018 | $0.0043 | $0.0094 | $0.0525 | $0.1169 | $0.1339 | $0.4000 | $7.1520 |
| 600 | $0.0049 | $0.0073 | $0.0496 | $0.1352 | $0.1327 | $0.1497 | $0.4751 | $8.6541 |
| 8,000 | $0.0652 | $0.0722 | $0.9825 | $1.7643 | $0.4435 | $0.4605 | $1.9551 | $38.2541 |
| 8,192 | $0.0689 | $0.0737 | $1.0077 | $1.8660 | $0.4516 | $0.4685 | $1.9935 | $39.0221 |
| 16,000 | $0.1326 | $0.1419 | $1.9918 | $3.5850 | $0.7795 | $0.7965 | $3.5551 | $70.2541 |
| 32,000 | $0.2694 | $0.2814 | $4.0208 | $7.2858 | $1.5698 | $1.6037 | $6.7551 | $134.2541 |

All values are estimated USD per **1,000 transcripts**. GLiClass Modern is the notable result: it retains arbitrary runtime labels while remaining close to the fixed-head ModernBERT cost curve in this simulation. At the 224-token benchmark mean it is approximately $0.0043 per 1,000 transcripts, versus $0.0018 for trained ModernBERT, $0.0525 for zero-shot ModernBERT NLI, $0.1169 for JEV Noul, $0.4000 for Luna, and $7.1520 for Sol. GLiClass Large is also inexpensive for short transcripts, but its 512-token context and two label groups make long-state chunking costly.

The cost model counts repeated overlap and label/request overhead. With `S` as raw state tokens, `kM` as ModernBERT chunks, `kGM` and `kGL` as the two GLiClass chunk counts, and `kJ` as JEV requests:

```text
kM = max(1, ceil((S - 256) / (8178.11 - 256)))
kGM = max(1, ceil((S - 256) / (7895 - 256)))
kGL = max(1, ceil((S - 256) / (357 - 256)))
kJ = max(1, ceil((S - 256) / (31890 - 256)))

ModernBERT trained:   0.008154 * [S + 256(kM-1) + 2kM] / 1,000
GLiClass Modern:      0.008154 * [S + 256(kGM-1) + 297kGM] / 1,000
GLiClass Large:       0.012669 * {2[S + 256(kGL-1)] + 294kGL} / 1,000
ModernBERT zero-shot: 0.008154 * {27[S + 256(kM-1)] + 375kM} / 1,000
JEV Noul:             0.042 * [S + 256(kJ-1) + 2559.74kJ] / 1,000
JEV Choice:           0.042 * [S + 256(kJ-1) + 2963.67kJ] / 1,000
GPT-5.6 Luna:         [0.20(S + 503.53) + 1.20(212)] / 1,000
GPT-5.6 Sol:          [4.00(S + 503.53) + 20.00(212)] / 1,000
```

Chunks may execute in parallel, but parallelism changes latency rather than total token-compute cost. Zero-shot ModernBERT evaluates all 27 questions against every chunk; trained ModernBERT encodes each chunk once; GLiClass Modern encodes each chunk once with all labels; GLiClass Large encodes each chunk twice for its two label groups; JEV uses one request per state chunk. Aggregation compute is excluded and aggregation quality requires validation.

GLiClass Modern is assigned the 170.3k processed-token/s ModernBERT-large H100 proxy because it uses that backbone. GLiClass Large is assigned 109.6k processed tokens/s by scaling that proxy with the official 32-label A6000 sample-throughput ratio, `28.79 / 44.73`. These are simulations rather than H100 measurements. Throughput sources: [official ModernBERT efficiency comparison](https://huggingface.co/blog/modernbert), [third-party H100 ModernBERT-base observation](https://www.linkedin.com/posts/michael-feil_the-latest-release-of-infinity-httpslnkdin-activity-7280971190632943616-E07N), and the [GLiClass model-card throughput table](https://huggingface.co/knowledgator/gliclass-large-v3.0). The $5/H100-hour price is a scenario assumption.

Sol and Luna use the standardized one-call prompt already used by the API-cost comparison: 728 mean input tokens at the 224.46-token benchmark state and a fixed 212-token binary JSON response. The state-length curves therefore use `S + 503.53` input tokens and assume each added state token contributes one LLM input token. This is approximate because the state axis uses ModernBERT token counts while the LLM estimate used `o200k_base`. Hidden reasoning tokens remain excluded. Current OpenAI API rates are [Sol: $4/M input and $20/M output](https://developers.openai.com/api/docs/models/gpt-5.6-sol) and [Luna: $0.20/M input and $1.20/M output](https://developers.openai.com/api/docs/models).

Context-limit sources: [ModernBERT documentation](https://huggingface.co/docs/transformers/en/model_doc/modernbert) and [JEV models and limits](https://docs.typesafe.ai/models).

## Generic transcript workload views

The original chart above preserves the measured 27-question workload. The following projections make the comparison easier to reuse for other transcript-classification problems by varying both transcript duration and taxonomy size.

### Cost by transcript duration and taxonomy size

![Estimated transcript-classification cost at 10, 25, and 50 questions](charts/generic-cost-vs-transcript-duration.png)

The panels use **10, 25, and 50 questions**. The bottom axes show transcript duration and the top axes show the corresponding state-token assumption. The architecture labels generalize the tested implementations:

- **Fixed trained encoder:** ModernBERT trained heads; one state encoding, but every label must be trained in advance.
- **Shared-state zero-shot encoder:** GLiClass Modern; runtime labels are serialized with the state.
- **Pairwise zero-shot encoder:** ModernBERT NLI; the state is repeated once for every question.
- **Hosted shared-state billing:** JEV Noul; the curve represents observed billing behavior, not verified internal compute.
- **Single-call LLM:** Luna or Sol; one prompt contains the state and all requested labels.

### Cost as the taxonomy grows

![Estimated transcript-classification cost from 1 to 100 questions](charts/generic-cost-vs-question-count.png)

This view fixes three representative transcript scenarios and varies the taxonomy from 1 to 100 questions. It makes the pairwise-NLI penalty explicit: longer transcripts make every additional state copy more expensive, while shared-state and one-call systems primarily add label, billing, or output overhead.

These are scenario projections rather than additional measured benchmarks. They assume 200 state tokens per transcript minute. Question-dependent overhead is allocated proportionally from the measured 27-question workload because separate fixed-intercept and per-question measurements are unavailable. Hidden Sol/Luna reasoning tokens remain excluded, and the trained-head curve treats logistic-head compute as negligible relative to the encoder pass.

## What ModernBERT actually processes

Your distinction is correct for zero-shot ModernBERT, but the trained model is even more specialized than `state + N * question_size`:

| System | Approximate inference input | Arbitrary new questions? |
|---|---|---|
| ModernBERT zero-shot NLI | `sum(tokenize(state, question_i))`, approximately `N * state + sum(question_i)` | Yes |
| Trained ModernBERT heads | `tokenize(state)` once, followed by `sigmoid(W h + b)` for 27 learned heads | No |
| JEV | One state plus arbitrary typed questions in one request | Yes |

Across the locked test set, the ModernBERT tokenizer measured a 224.46-token average state. Zero-shot NLI processed 6,435.42 tokens per transcript across the 27 pairs, a **28.67x amplification**. The trained model processed about 226.46 tokens once. It does not tokenize the 27 question descriptions at inference: their meaning has been absorbed into the trained head weights. A new attribute therefore requires a new or retrained head.

## Does this prove JEV has a new architecture?

No—not from public evidence currently available. TypeSafe publicly claims a “new model architecture,” a “parallel sampler,” and RLCD training. Its API unquestionably supports arbitrary state and arbitrary typed questions, something the fixed-head ModernBERT configuration cannot do. But the public API documentation does not disclose the internal compute graph, parameter count, attention arrangement, or state-reuse mechanism.

The observed billing equation and near-flat 1-to-16-question latency establish useful external behavior, not architectural novelty. A conventional batched cross-encoder can also show nearly flat latency until the GPU batch saturates while still repeating the state internally. A dual encoder, cached state encoder, late-interaction model, or shared-state cross-attention design could also provide arbitrary questions without being a fundamentally new model family.

The defensible conclusion is: **JEV exposes a valuable arbitrary-question/shared-state product abstraction and prices it as shared state; whether its internal architecture is genuinely novel remains unverified.**

## Complete quality table

![Accuracy, precision, recall, F1, and exact-match results](charts/all-metrics-table.png)

| Approach | Accuracy | Precision | Recall | F1 | Exact match |
|---|---:|---:|---:|---:|---:|
| DeBERTa v3 large zero-shot (`-c`) | 92.89% | 84.70% | 89.14% | 86.86% | 13.33% |
| ModernBERT zero-shot | 92.96% | 84.74% | 89.42% | 87.02% | 14.00% |
| GLiClass Large v3, 0.50 threshold | 93.04% | 81.90% | 94.48% | 87.74% | 12.00% |
| GLiClass Modern Large v3, 0.50 threshold | 94.10% | 83.89% | 96.07% | 89.57% | 16.67% |
| JEV Choice | 94.40% | 82.47% | 100.00% | 90.39% | 18.00% |
| ModernBERT zero-shot, optimized thresholds | 96.42% | 92.53% | 94.01% | 93.27% | 34.67% |
| GLiClass Large v3, optimized thresholds | 96.59% | 91.82% | 95.60% | 93.67% | 34.67% |
| GLiClass Modern Large v3, optimized thresholds | 97.90% | 96.41% | 95.60% | 96.00% | 53.33% |
| ModernBERT trained heads | 99.16% | 98.05% | 98.78% | 98.41% | 83.33% |
| JEV Noul, 0.50 threshold | 99.26% | 97.27% | 100.00% | 98.61% | 80.67% |
| GPT-5.6 Luna | 99.75% | 99.07% | 100.00% | 99.53% | 93.33% |
| JEV Noul, validation-calibrated | 99.90% | 99.63% | 100.00% | 99.81% | 97.33% |
| GPT-5.6 Sol | 99.93% | 99.72% | 100.00% | 99.86% | 98.00% |

Accuracy, precision, recall, and F1 are micro-aggregated across all 4,050 held-out decisions. Exact match requires all 27 attributes for a transcript to be correct.

## API-only cost per 1,000 transcripts

![Estimated API cost for processing 1,000 transcripts](charts/estimated-cost-1000-transcripts.png)

This older chart reports marginal API charges for 1,000 transcripts. It should not be used to compare JEV against local ModernBERT because the local entries omit GPU cost; the normalized scenario above is the appropriate serving-cost comparison.

- JEV Choice uses the actual mean input usage from all 150 test calls: approximately $0.134 per 1,000 transcripts at $0.042 per million input tokens.
- JEV Noul uses actual input usage from a 10-transcript length-spanning sample: approximately $0.117 per 1,000 transcripts. Output is free under the published pricing.
- Sol and Luna use a standardized compact one-call-per-transcript prompt measured with `o200k_base`: 728 mean input tokens and 212 output tokens. Estimated costs are $7.15 for Sol and $0.40 for Luna at prices verified on September 20, 2026.
- Local encoders show zero API charges. Hardware, electricity, hosting, engineering, and operations are not zero and are deliberately excluded.
- Hidden reasoning tokens for the original Codex Sol/Luna evaluation were unavailable, so the LLM amounts are standardized scenario estimates rather than invoices from the benchmark run.

The JEV accounting used here follows the observed scaling behavior:

```text
billable input ~= state tokens + N * question tokens + fixed request overhead
```

It does **not** assume `N * (state tokens + question tokens)`. In the separate scaling check, latency remained approximately flat as question count increased from 1 to 16. That is consistent with TypeSafe's claim that questions are evaluated in parallel, but it does not prove that the hidden neural architecture encodes the state exactly once.

Pricing sources: [TypeSafe.ai JEV launch and pricing](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [GPT-5.6 Sol pricing](https://developers.openai.com/api/docs/models/gpt-5.6-sol), and the [official OpenAI model catalog](https://developers.openai.com/api/docs/models) for Luna.

## What each variant means

- **ModernBERT zero-shot:** fixed NLI entailment scores with a universal 0.50 threshold.
- **ModernBERT zero-shot, optimized thresholds:** the same fixed NLI probabilities with 27 thresholds selected on validation only. No weights are trained.
- **ModernBERT trained heads:** frozen 1,024-dimensional ModernBERT embeddings plus 27 supervised logistic heads trained on 700 transcripts; thresholds are selected on validation.
- **DeBERTa zero-shot:** the commercially friendly DeBERTa-v3-large zero-shot checkpoint using the same hypotheses and 0.50 rule as ModernBERT.
- **GLiClass Modern Large v3:** a ModernBERT-based uni-encoder that scores all 27 natural-language attribute labels in one forward pass. Both the universal 0.50 and validation-calibrated operating points are reported.
- **GLiClass Large v3:** a DeBERTa-based uni-encoder. The 27 labels are divided deterministically into two 13–14-label passes so the complete transcript and label text fit its 512-token context; the two score groups are then reassembled before thresholding.
- **JEV Choice:** 27 binary present/absent Choice questions using a raw transcript state and generic criteria.
- **JEV Noul 0.50:** structured speaker turns and strict attribute boundaries with a universal 0.50 cutoff.
- **JEV Noul calibrated:** the same Noul probabilities with validation-selected thresholds; three ambiguous criteria were also refined using validation before the test was run.
- **Sol and Luna:** blind semantic classification of the same 27 attributes from transcript text, with no access to truth labels.

JEV Choice versus Noul is not a primitive-only A/B test. The Noul experiment also changed state structure and criteria specificity; its improvement cannot be attributed solely to the Noul API type.

## Repository layout

```text
charts/                         Generated comparison charts
data/
  synth_transcript.xlsx        1,000 synthetic labeled conversations
  summary_metrics.json         Chart-ready test metrics
  cost_assumptions.json        Pricing, token measurements, and caveats
  normalized_cost_scenario.json H100 throughput proxy, token amplification, and serving-cost assumptions
  benchmark_1000/              Fixed split and raw benchmark outputs
report/
  transcript-classifier-comparison.md
  transcript-classifier-comparison.pdf
  transcript-classifier-executive-summary.pdf
results/                        Consolidated JSON, model outputs, and workbook
scripts/                        Benchmark, calibration, chart, and report code
models/                         Local model downloads; ignored by Git
```

## Installation

Python 3.11 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the encoder checkpoints into the ignored `models/` directory:

```bash
hf download MoritzLaurer/ModernBERT-large-zeroshot-v2.0 \
  --local-dir models/modernbert-zeroshot

hf download MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c \
  --local-dir models/deberta-v3-large-zeroshot-v2.0-c

hf download knowledgator/gliclass-modern-large-v3.0 \
  --local-dir models/gliclass-modern-large-v3.0

hf download knowledgator/gliclass-large-v3.0 \
  --local-dir models/gliclass-large-v3.0
```

For JEV, copy the safe environment template and add the API key locally:

```bash
cp .env.example .env
```

`models/`, `.env`, checkpoints, and common model-weight extensions are excluded in `.gitignore`.

## Rebuild charts and report

```bash
MPLBACKEND=Agg python scripts/build_charts.py
python scripts/build_report.py
python scripts/build_executive_report.py
```

## Reproduce encoder scoring and calibration

The fixed split and existing raw outputs are committed under `data/benchmark_1000/`.

```bash
# Zero-shot ModernBERT on validation
HF_HUB_OFFLINE=1 python scripts/synthetic_1000_benchmark.py zero-shot \
  --part validation \
  --model models/modernbert-zeroshot \
  --output data/benchmark_1000/modernbert_zero_shot_validation_results.json \
  --batch-size 64

# Fit validation-only thresholds and apply them to existing locked-test scores
python scripts/optimize_modernbert_zeroshot_thresholds.py

# GLiClass Modern Large: all 27 labels in one forward pass
HF_HUB_OFFLINE=1 python scripts/run_gliclass_benchmark.py \
  --model-dir models/gliclass-modern-large-v3.0 \
  --name gliclass_modern_large_v3 \
  --batch-size 32
python scripts/optimize_gliclass_thresholds.py \
  --name gliclass_modern_large_v3

# GLiClass Large: two label batches preserve full transcript evidence within
# the checkpoint's 512-token context window
HF_HUB_OFFLINE=1 python scripts/run_gliclass_benchmark.py \
  --model-dir models/gliclass-large-v3.0 \
  --name gliclass_large_v3 \
  --batch-size 2 \
  --label-batches 2
python scripts/optimize_gliclass_thresholds.py \
  --name gliclass_large_v3

# Recalculate all consolidated metrics
python scripts/evaluate_synthetic_1000.py
```

The local CPU timings in the raw outputs are observed measurements, not hardware-normalized benchmarks. The H100 serving-cost scenario instead uses the documented throughput proxy and $5/GPU-hour assumption in `data/normalized_cost_scenario.json`.

## Full report

The detailed methodology, interpretation, cost assumptions, and limitations are available in [the Markdown report](report/transcript-classifier-comparison.md) and [the rendered PDF](report/transcript-classifier-comparison.pdf).

## License

Code and original report content are released under the MIT License. Model checkpoints and third-party datasets retain their own licenses and are not redistributed here.

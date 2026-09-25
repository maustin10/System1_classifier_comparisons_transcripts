# Testing AskATT System1 API

This note shows how `Askatt_system1_api` behaves on the existing transcript benchmark and on a new multiclass Choice suite. The tests use the local model through the same service layer that backs `POST /v1/systemone`; contract and real-model HTTP tests separately verify the endpoint.

## What the API does

One request sends a shared `state` and one or more typed questions:

```json
{
  "model": "jev-latest",
  "state": "[Caller]: An unfamiliar fee appeared on my bill.",
  "questions": {
    "unknown_charge": {
      "type": "noul",
      "instructions": "Is an unknown charge present?",
      "criteria": {"true": "Unknown charges on the bill"}
    },
    "department": {
      "type": "choice",
      "instructions": "Which department should own the request?",
      "criteria": {
        "billing": "Billing, charges, payments, refunds, credits, fees, or invoices",
        "technical": "Internet, mobile service, connectivity, device, activation, error, or performance troubleshooting",
        "sales": "New purchase, upgrade, plan recommendation, price quote, or adding service"
      }
    }
  }
}
```

An illustrative response contains a Noul probability and a normalized Choice distribution:

```json
{
  "model": "askatt-gliclass-modern-large-v3.0",
  "answers": {
    "unknown_charge": {"type": "noul", "noul": 0.9989},
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.97, "technical": 0.02, "sales": 0.01},
      "confidence": 0.86
    }
  },
  "usage": {"input_tokens": 92, "output_tokens": 0}
}
```

For Noul, the positive criterion becomes one GLiClass runtime label. Conventional binary Choice (`present/absent`, `yes/no`, or `true/false`) uses the same positive score and its complement. General Choice appends the question instructions to the state, compiles one label per option, converts the independent scores to log-odds, and applies a within-question softmax. `confidence` is normalized distribution concentration; it is not JEV confidence.

## Transcript benchmark

The locked test split contains 150 transcripts and 27 binary attributes per transcript, or 4,050 decisions. All 27 attributes fit in one shared-state pass and no transcript was truncated.

| API form | Accuracy | Precision | Recall | F1 | Exact transcript match |
|---|---:|---:|---:|---:|---:|
| Noul at 0.50 | 94.10% | 83.89% | 96.07% | 89.57% | 16.67% |
| Binary Choice | 94.10% | 83.89% | 96.07% | 89.57% | 16.67% |

Binary Choice is intentionally identical to Noul here; it is not a separate multiclass quality result. Each request averaged 521.46 serialized input tokens, one forward pass, and zero output tokens.

The latest local CPU rerun took 193.77 seconds for Noul and 97.58 seconds for binary Choice. A prior run took 78.62 and 83.10 seconds. Because the compiled labels and token counts are identical, this variation is environmental and should not be interpreted as an inherent speed advantage for either response shape. Use GPU measurements with controlled concurrency for capacity or latency decisions.

## Multiclass Choice benchmark

The new semantic suite contains 24 labeled customer-care conversations. Every state asks four Choice questions in one request:

1. Owning department: 5 options
2. Operational urgency: 4 options
3. Caller sentiment: 3 options
4. Immediate next action: 6 options

That produces 96 decisions from 18 compiled labels per state. To preserve the instructions, the fixed adapter appends each question to the state and evaluates that question's options together. It therefore uses four passes per state—one per question—while still sharing the state across the 3–6 options within each question. No state was truncated.

| Result | Declared option order | Reversed option order |
|---|---:|---:|
| Overall accuracy | 88.54% | 91.67% |
| Exact four-answer state match | 54.17% | 66.67% |
| Mean confidence | 80.41% | 82.08% |

Declared-order accuracy by question:

| Question | Accuracy |
|---|---:|
| Department | 83.33% |
| Urgency | 87.50% |
| Sentiment | 95.83% |
| Next action | 87.50% |

Declared-order accuracy by case difficulty:

| Difficulty | Accuracy |
|---|---:|
| Direct | 95.83% |
| Distractor | 81.25% |
| Multiple issues | 87.50% |
| Paraphrase | 75.00% |

Reversing the option order changed 9 of 96 answers, a **9.38% flip rate**; the reversed-order run happened to score three more correct answers, which reinforces why order stability must be measured separately from accuracy. The declared-order run made 11 errors, four with reported confidence at or above 0.50. This small synthetic benchmark supports the corrected API behavior, but it does not support unqualified production use without held-out domain validation.

### Question-context fix and ablation

The original adapter used each option definition as the runtime label but did not include the Choice question's `instructions`. That is a real semantic limitation: the same option can mean something different under a different question. Putting each question into the state context while retaining familiar option-definition labels fixed the semantic omission and raised accuracy by **11.46 percentage points**, from 77.08% to 88.54%.

The tradeoff is explicit: the legacy path processed all four questions in one pass, while the corrected path repeats the state in four question-scoped passes. Mean serialized input increased from 343.38 to 537.50 tokens per state. The API reports this in `X-AskATT-Forward-Passes`, and `ASKATT_CHOICE_INSTRUCTION_MODE=discard` remains available only to reproduce the legacy result.

Before finding the state-context solution, we also tested whether the question could remain in one shared pass by incorporating it into each label. The complete 24-state, 96-decision suite used a nine-format sweep plus the initial fully structured API encoding:

| Runtime-label encoding | Accuracy | Exact four-answer state match | Mean input tokens/state |
|---|---:|---:|---:|
| Option definition only (legacy baseline) | **77.08%** | **33.33%** | **343.38** |
| `Question: option definition` | 75.00% | 16.67% | 529.38 |
| `semantic question scope: option definition` | 73.96% | 25.00% | 407.38 |
| `option definition (question)` | 73.96% | 16.67% | 547.38 |
| `Question Answer: option definition` | 68.75% | 12.50% | 565.38 |
| Question, then newline and option definition | 67.71% | 8.33% | 546.38 |
| `option definition. Question: question` | 62.50% | 12.50% | 583.38 |
| Compact `question option: definition` | 57.29% | 4.17% | 583.38 |
| `semantic scope — option name: definition` | 56.25% | 0.00% | 453.38 |
| Fully structured `Question` / `Option` / `Definition` | 47.92% | 0.00% | 709.38 |

The direct structured fix was materially worse: accuracy fell from 77.08% to 47.92%, and option-order flips rose from 7.29% to 30.21%. The best concise question-aware encoding reached 75.00%. It improved department accuracy from 54.17% to 75.00%, but urgency fell from 70.83% to 54.17% and sentiment from 95.83% to 83.33%.

This result suggests that the pretrained checkpoint treats runtime labels primarily as flat semantic hypotheses; it has not learned the proposed `question + option + definition` grammar. Simply adding the missing text changes the model's task and introduces lexical biases—for example, the word “urgency” can pull the model toward the `urgent` option. Short semantic scopes such as “owning department” and “operational urgency level” also failed to beat the baseline. The fixed adapter therefore places the question in the state rather than changing the label grammar.

The durable efficiency improvement is to fine-tune the shared-state model on the exact composite-label grammar, with multiple questions per state, hard negative options, and option-order permutations. That training should use separate train, validation, and locked test splits; this exploratory ablation is evidence for the training design, not an unbiased estimate of a newly selected model.

Recommended release gates are:

- validate each production Choice question on held-out domain data;
- permute option order and require an acceptable flip rate;
- inspect class-level confusion rather than overall accuracy alone;
- calibrate abstention or escalation rules independently for each question;
- repeat the same held-out comparison when models, prompts, or label definitions change;
- improve the multiclass scoring adapter or fine-tune the underlying model before relying on weak categories.

## Comparison with TypeSafe.ai JEV and zero-shot ModernBERT

The same 24 states, four questions, option definitions, and 96 truth labels were sent to the hosted TypeSafe.ai JEV API and evaluated with `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`. JEV and AskATT each received one state plus four runtime questions per request. ModernBERT used the existing pairwise NLI framework: each state/question/option combination became a separate premise/hypothesis pair, and a softmax over each question's option entailment logits selected the answer. This is the runtime-label zero-shot ModernBERT model, not the separately trained fixed-taxonomy heads.

| Result | TypeSafe.ai JEV | AskATT question-scoped encoder | ModernBERT pairwise NLI |
|---|---:|---:|---:|
| Overall accuracy | **96.88%** | 88.54% | 59.38% |
| Correct decisions | **93 / 96** | 85 / 96 | 57 / 96 |
| Exact four-answer state match | **87.50%** | 54.17% | 4.17% |
| Option-order flip rate | 2.08% | 9.38% | **0.00%** |

Accuracy by question:

| Question | TypeSafe.ai JEV | AskATT | ModernBERT |
|---|---:|---:|---:|
| Department | **100.00%** | 83.33% | 62.50% |
| Urgency | **91.67%** | 87.50% | 16.67% |
| Sentiment | **95.83%** | **95.83%** | **95.83%** |
| Next action | **100.00%** | 87.50% | 62.50% |

Against AskATT on the paired decisions, both were correct 83 times, only JEV was correct 10 times, only AskATT was correct twice, and both were wrong once. Against ModernBERT, both AskATT and ModernBERT were correct 55 times, only AskATT was correct 30 times, only ModernBERT was correct twice, and both were wrong nine times.

JEV's three declared-order errors were two urgency judgments and one sentiment judgment. It was perfect on department and next-action selection. Reversing every option list changed two urgency answers, reducing accuracy from 96.88% to 94.79%. ModernBERT predicted `urgent` for every urgency case, including routine, elevated, and critical examples. That failure is specific to this prompt-and-label formulation and should be tested for remediation before treating 59.38% as a general ModernBERT ceiling.

The declared-order observations were 8.51 seconds for 24 sequential JEV hosted API calls, 19.52 seconds for AskATT on local CPU, and 256.03 seconds for ModernBERT on local CPU. JEV reported 21,489 input tokens and 5,113 output tokens. These timings mix a hosted service with two local CPU implementations and therefore must not be read as a hardware-normalized latency benchmark. The compute paths are also different: JEV's private architecture is unknown, fixed AskATT repeats the state four times—once per question—and jointly scores that question's options, while ModernBERT creates 18 separate NLI sequences per state and repeats the state in every sequence.

The comparison is directional: it contains only 24 synthetic states, none of the models was tuned for these four questions, and probability calibration was not applied. JEV's `confidence`, AskATT's entropy-based confidence, and ModernBERT's top softmax probability are not interchangeable. ModernBERT's zero option-order flips are expected from independent pairwise scoring. JEV and AskATT both showed measurable order sensitivity, so accuracy and robustness should remain separate release gates.

## Choice cardinality stress test

A separate synthetic exact-match test asks the model to select an explicitly named routing destination from 2, 4, 8, 16, 32, and 65 options. It is a mechanics and capacity test, not a semantic benchmark.

| Options | Accuracy | Mean CPU inference | Mean forward passes |
|---:|---:|---:|---:|
| 2 | 100% | 156 ms | 1 |
| 4 | 100% | 185 ms | 1 |
| 8 | 100% | 243 ms | 1 |
| 16 | 100% | 267 ms | 1 |
| 32 | 100% | 456 ms | 1 |
| 65 | 100% | 885 ms | 2 |

At 65 options the service crosses the configured 64-label boundary, issues two model passes, and repeats the state. The 100% result only shows that the API can preserve and select an explicit exact-match option; it does not imply 100% accuracy on real 65-way classification.

## Run everything

```bash
.venv/bin/python -m pytest -q tests/test_askatt_system1_api.py tests/test_askatt_choice_benchmark.py
ASKATT_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -q tests/test_askatt_system1_model.py

.venv/bin/python scripts/benchmark_askatt_system1_api.py \
  --output results/askatt_system1_api_benchmark_rerun.json

.venv/bin/python scripts/benchmark_askatt_choice.py \
  --output results/askatt_choice_benchmark.json

.venv/bin/python scripts/benchmark_askatt_label_formats.py

.venv/bin/python scripts/benchmark_askatt_question_in_state.py

.venv/bin/python scripts/benchmark_modernbert_choice.py \
  --output results/modernbert_choice_benchmark.json

.venv/bin/python scripts/benchmark_jev_choice.py \
  --output results/jev_semantic_choice_benchmark.json
```

Machine-readable outputs:

- `results/askatt_system1_api_benchmark_rerun.json`
- `results/askatt_choice_benchmark.json`
- `results/askatt_choice_benchmark_pre_question_context.json`
- `results/askatt_choice_benchmark_full_question_context.json`
- `results/askatt_choice_label_format_sweep.json`
- `results/askatt_choice_question_in_state.json`
- `results/modernbert_choice_benchmark.json`
- `results/jev_semantic_choice_benchmark.json`

The model checkpoint remains excluded from Git. See `askatt_system1_api/README.md` for installation and deployment instructions.

# AskATT and open JEV-like models: execution status

## Completed local baseline

AskATT now implements all three System One question types required by JevBench:
Noul, Choice, and ordered Score. Score levels are evaluated as runtime labels;
the API returns the level distribution, its expected ordinal value, and
distribution confidence.

The public JevBench run used revision
`1bcc55eb6c8cffde2306b3db03ede39b61c6152a` on an Apple M2 Pro with 16 GB unified
memory. It is a local public-cohort measurement, not an official sealed
JevBench submission.

| Cohort | AskATT base correct | Accuracy | Strict schema validity | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| Easy | 44/48 | 91.67% | 100% | 0.106 s | 0.127 s |
| Original | 41/72 | 56.94% | 100% | 0.107 s | 0.146 s |
| Hard | 40/111 | 36.04% | 100% | 0.508 s | 3.599 s |
| **Public total** | **125/231** | **54.11%** | **100%** | — | — |

The deployed hybrid configuration produced exactly the same 125/231 answers
and cohort metrics. Its hard-cohort p50/p95 were 0.863/5.698 seconds in this
separate sequential CPU run. The equality indicates that its conservative
question router did not materially alter this suite; the latency difference is
a local run/configuration observation, not a controlled throughput claim.

For context, TypeSafe.ai JEV 1.13 reports 200/231, or 86.58%, on the same public
cohorts. The result says AskATT remains strong on straightforward
classification but does not yet generalize to JevBench's arithmetic,
multi-hop, trade-off, policy, and adversarial decision cases. Its out-of-domain
probabilities are also overconfident: ECE is 0.322 on original and 0.408 on
hard.

## Kev-4B feasibility result

Kev-4B BF16 loaded successfully through MLX, after downloading its 9.32 GB base
model plus adapter. The 16 GB Mac then became heavily swap-bound: only two
original-cohort requests completed after several minutes. The run was stopped
because its latency would measure memory thrashing rather than Kev. This is a
hardware-capacity failure, not a model-quality result.

Run Kev-4B on the 32 GB M5 or an A100/H100 using the frozen commands in
`benchmarks/jevbench/README.md`.

## Razorback OpenJev feasibility result

Razorback OpenJev revision
`6900d153b9052417135729a85137e2d5f782add1` and its 4-bit MLX
DiffusionGemma checkpoint were installed and loaded successfully. The checkpoint
download reconstructed 16.5 GB of weights. A one-case JevBench smoke request
then timed out after 120.13 seconds without producing a decision.

During that request, cumulative swap-ins rose from 5,153,526 to 6,592,423 and
swap-outs rose from 6,871,107 to 8,173,545. That is severe memory thrashing:
the checkpoint requires about 16 GB of free unified memory, while this M2 Pro
has 16 GB total. The full 231-case run was therefore stopped. This is a measured
hardware-capacity failure, not a Razorback accuracy result. The preserved smoke
record is `results/jevbench/openjev-diffusion/smoke-m2-pro-16gb.jsonl`.

Run the MLX path on a 32 GB or larger Mac, or use Razorback's pinned vLLM
container on an NVIDIA GPU with at least 24 GB, before comparing quality and
throughput.

## Remaining accelerator matrix

| System | Preferred hardware | Purpose |
|---|---|---|
| Kev-4B | 32 GB M5, A100, or H100 | Trainable Qwen decision-model comparison |
| Razorback OpenJev / DiffusionGemma | 32 GB+ Mac or 24 GB+ NVIDIA GPU | Shared diffusion-canvas architecture |
| `openjev/openjev` 27B FP8 | 80 GB A100/H100 | Quality-oriented research ceiling |
| Zefan Cai Open-Jev 27B | 80 GB A100/H100 | Independent 27B trained control |

All systems should be evaluated through the same frozen public cohorts plus our
transcript, semantic Choice, and BFCL-routing suites. The public JevBench result
should remain separate from its official sealed leaderboard.

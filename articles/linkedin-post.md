# Proposed LinkedIn post

## Recommended opening

**A frontier AI model that refuses to write a sentence nearly tied GPT-5.6 Sol—and changed where the cost curve bends.**

## Alternative openings

1. **TypeSafe says the future of AI automation is not another LLM. My test suggests they may have a point.**
2. **JEV is either a new model category—or the sharpest productization yet of an old encoder idea. Either way, the benchmark got interesting.**
3. **What if the best model for an AI workflow is the one that cannot generate text?**
4. **We keep asking LLMs to make tiny decisions. TypeSafe.ai just made a model that does almost nothing else—and that may be the point.**

## Ready-to-publish post

**A frontier AI model that refuses to write a sentence nearly tied GPT-5.6 Sol—and changed where the cost curve bends.**

TypeSafe.ai recently released **JEV**, its first “System One” model. Instead of generating strings token by token, JEV accepts state plus typed questions and returns probabilistic decisions that software can use directly. TypeSafe describes it as a new architecture with parallel sampling and a training method called Reinforcement Learning for Calibrated Decisions.

That sounded both new and strangely familiar. Encoder classifiers have made fast, parallel decisions for years. So I built an independent smoke test against ModernBERT, GLiClass, and two frontier LLM configurations.

The test used 1,000 synthetic customer-care conversations with 27 binary attributes. Quality was measured on a locked 150-conversation test set: 4,050 decisions.

The headline results:

- **GPT-5.6 Sol:** 99.93% accuracy / 99.86% F1
- **TypeSafe.ai JEV Noul, validation-calibrated:** 99.90% / 99.81%
- **GPT-5.6 Luna:** 99.75% / 99.53%
- **ModernBERT with trained heads:** 99.16% / 98.41%
- **GLiClass Modern, validation-calibrated:** 97.90% / 96.00%
- **ModernBERT zero-shot, validation-calibrated:** 96.42% / 93.27%

The more interesting result was not a universal winner. It was the decision boundary.

For a fixed taxonomy and a well-utilized self-hosted H100, trained ModernBERT was the modeled cost leader. But those heads cannot accept a brand-new question at runtime. JEV can—and at low GPU utilization its usage-based economics can beat the self-hosted option. At our 6,000-token, 25-question reference workload, the modeled JEV/ModernBERT crossover was about **13.9% paid H100 utilization**.

My conclusion: **JEV looks compelling as a managed, runtime-flexible decision API. Trained encoders remain extremely hard to beat when labels are stable and infrastructure stays busy.** GLiClass deserves attention as an open, runtime-label middle ground.

One important caveat: this test validates observable quality, pricing behavior, and API ergonomics. It does **not** prove the internal architecture is fundamentally new; TypeSafe has not publicly disclosed enough of the compute graph to establish that.

Congratulations to **Diogo Almeida, Sasha Sheng, Erik Gafni, and the TypeSafe.ai team** for putting a genuinely interesting systems question back on the table: should automation models generate language at all, or should they make typed decisions?

Full methodology, charts, code, and data:
https://github.com/maustin10/System1_classifier_comparisons_transcripts

#AI #MachineLearning #SystemOne #BERT #LLM #MLOps #AgenticAI #Classification

## Shorter summary version

**What if the best model for an AI workflow is the one that cannot generate text?**

I independently compared TypeSafe.ai JEV with ModernBERT, GLiClass, and GPT-5.6 Sol/Luna on 4,050 held-out transcript decisions.

JEV reached **99.90% accuracy and 99.81% F1**, nearly matching Sol at 99.93% / 99.86%. Trained ModernBERT reached 99.16% / 98.41% and was the modeled cost winner when the taxonomy was fixed and an H100 stayed busy. JEV became attractive when questions had to change at runtime or GPU utilization fell.

The takeaway is not “JEV wins everything.” It is more useful: **fixed taxonomy favors trained encoders; runtime questions and managed-service economics make JEV compelling; GLiClass offers an intriguing open-source middle ground.**

This is a synthetic smoke test, not a production benchmark—and it verifies external behavior, not TypeSafe’s undisclosed architecture.

https://github.com/maustin10/System1_classifier_comparisons_transcripts


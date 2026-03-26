# RECoT / Rationalization-Augmented Training — Literature Notes & Framing

## Our Approach

Unlike standard chain-of-thought distillation where the teacher independently produces both reasoning and labels (risking noise propagation), we condition the teacher's reasoning on the human-annotated ground truth labels. This ensures label fidelity while enriching the training signal with structured rationales.

**Key distinction from standard CoT distillation:**
- Standard CoT distillation: teacher generates reasoning + answer → student learns both (inherits teacher errors)
- Our approach: human labels are fixed, teacher only generates reasoning that explains the known-correct labels → student gets clean labels + plausible reasoning

**Closest published concepts:**
- STaR (Zelikman et al., 2022) — self-generates rationales and filters by correctness. We use a stronger teacher constrained to the correct answer.
- Rationalization distillation — teacher writes post-hoc explanations for known answers.

**Our specific contributions:**
1. Reverse reasoning — teacher sees true labels and generates reasoning that arrives at them
2. Applied to multi-label hierarchical classification (77-class taxonomy), not math/QA
3. Structured SCAN/VERIFY reasoning format designed for this task
4. Domain: climate misinformation detection

## Key Citations

### CoT Distillation (Supportive)
- **Symbolic CoT Distillation (SCoTD)** — Li et al., ACL 2023. Small models (125M-1.3B) benefit from CoT distillation. Sampling many reasoning chains per instance is key. https://aclanthology.org/2023.acl-long.150/
- **CoT Collection** — Kim et al., 2023. CoT fine-tuning on Flan-T5 yields +4.34% (3B) and +2.60% (11B) on BIG-Bench-Hard zero-shot. arXiv:2305.14045
- **DeepSeek R1 Distillation** — 800K reasoning samples distilled into smaller models. Distillation outperformed direct RL. Models <=7B showed persistent performance gaps. https://github.com/deepseek-ai/DeepSeek-R1
- **STaR: Self-Taught Reasoner** — Zelikman et al., 2022. Self-generate rationales, filter by correctness, fine-tune iteratively. arXiv:2203.14465

### CoT Distillation (Critical / Limitations)
- **TextReasoningBench** — March 2026. Directly tested whether reasoning improves text classification. CoT provides only +1% to +3% on large models for classification. Complex reasoning strategies (ToT, GoT) often degrade performance. Small models (8B): unstable responses to reasoning. arXiv:2603.19558
- **"In Their Own Words"** — Teacher reasoning traces contain tokens that are low-probability under student's distribution. Direct distillation of teacher traces degraded Qwen3-0.6B by 20.5%. Student-aligned traces improved by 4.9%. arXiv:2509.22230
- **"Unveiling Key Factors"** — Three factors drive CoT distillation: (1) Granularity (most important), (2) Teacher selection, (3) Format (least important). Weaker students plateau or decline with excessive detail. arXiv:2502.18001
- **"Is CoT Reasoning a Mirage?"** — Argues CoT is pattern matching, not genuine reasoning. Models produce correct reasoning but wrong answers (or vice versa). arXiv:2508.01191
- **Data Repetition vs Scaling** — More exposure (even via repetition) can outperform more data. Relevant confound: RECoT has longer sequences = more gradient signal. arXiv:2602.11149

### Related Methods
- **RCOT** — Xue et al., 2023. Reverses CoT to reconstruct original problem for factual consistency checking. Different from our approach. arXiv:2305.11499
- **D-CoT** — Disciplined CoT with control tags as scaffolding. Reduces overthinking. arXiv:2602.21786
- **Adaptive CoT Distillation** — Adapt distillation data to student performance. MDPI 2025. https://www.mdpi.com/2227-7390/13/22/3646
- **TAID** — Sakana AI. Teacher-adaptive intermediate representations for distillation. https://sakana.ai/taid/

## Confounding Factors to Address

1. **Longer training signal** — RECoT responses have more tokens = more gradient per sample. Need length-matched control to isolate.
2. **Format consistency** — RECoT model has 1 parse failure vs 5 for norecot. Some F1 gap may come from output format, not classification quality.
3. **Answer anchoring** — Forcing reasoning before answer may act as "slow down" mechanism, not genuine reasoning.

## Suggested Experiments to Strengthen Claims

1. Run RECoT-trained model without CoT trigger at inference — if performance holds, training signal helped; if drops, it's inference-time reasoning doing the work.
2. Report parse failures explicitly as part of results.
3. Check reasoning faithfulness — does generated reasoning correspond to predictions?

## Our Empirical Results (Val Set, 615 samples)

RECoT consistently outperforms norecot across all levels and support filters:
- Level 3, Support >= 3: Micro F1 0.798 vs 0.749, Precision 0.854 vs 0.770
- Parse failures: RECoT=1, norecot=5
- See data/results/reports/metrics_summary.json for full results

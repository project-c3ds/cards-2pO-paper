# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "transformers>=5.2.0",
#     "flash-linear-attention",
#     "huggingface_hub[hf_transfer]",
#     "scikit-learn",
#     "peft",
# ]
# ///

import os
import sys
import json
import re
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

from unsloth import FastLanguageModel
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

HF_USERNAME = "iRanadheer"
ADAPTER_REPO = f"{HF_USERNAME}/cards_qwen35_4b_sft"
MAX_SEQ_LENGTH = 2560

# 1. Load model + adapter
print("[1/4] Loading model + LoRA adapter...")
start = time.time()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_REPO,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=True,
)
FastLanguageModel.for_inference(model)
print(f"Model loaded in {time.time() - start:.1f}s")

# 2. Load eval dataset
print("\n[2/4] Loading eval dataset...")
eval_dataset = load_dataset(
    f"{HF_USERNAME}/cards_sft_dataset",
    data_files="qwen_sft_eval.jsonl",
    split="train"
)
print(f"Eval examples: {len(eval_dataset)}")

# 3. Run inference
print("\n[3/4] Running inference...")
start = time.time()

def extract_categories_from_yaml(text):
    """Extract category codes from model output. Tries YAML, JSON, then regex fallback."""
    # Get text after </think> if present
    output = text.split('</think>')[-1] if '</think>' in text else text

    # Try YAML format: "  - 4_1_1"
    cats = re.findall(r'^\s*-\s*(\S+)', output, re.MULTILINE)
    cats = [c.strip('<>') for c in cats if re.match(r'<?(\d+_\d+_\d+)', c)]

    # Fallback: try JSON format: "category": "<4_1_1>"
    if not cats:
        cats = re.findall(r'"category":\s*"<?([^">]+)>?"', text)

    # Fallback: find any X_Y_Z pattern in the output
    if not cats:
        cats = re.findall(r'\b(\d+_\d+_\d+(?:_\d+)?)\b', output)

    # Clean angle brackets
    cats = [c.strip('<>') for c in cats]
    return cats

def extract_ground_truth(assistant_text):
    """Extract ground truth categories from assistant response."""
    yaml_section = assistant_text.split('</think>')[-1] if '</think>' in assistant_text else assistant_text
    cats = re.findall(r'^\s*-\s*(\S+)', yaml_section, re.MULTILINE)
    return [c.strip() for c in cats if c.strip()]

results = []
for i, example in enumerate(eval_dataset):
    messages = example["messages"]
    system_prompt = messages[0]["content"]
    user_text = messages[1]["content"]
    ground_truth = extract_ground_truth(messages[2]["content"])

    # Build prompt string manually to bypass vision processor issues
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    # Use the inner tokenizer directly, not the vision processor
    inner_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    input_ids = inner_tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Generate
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
    )

    # Decode only new tokens
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    predicted = extract_categories_from_yaml(generated)

    results.append({
        "index": i,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "generated_text": generated,
    })

    if i < 5 or i % 50 == 0:
        print(f"  [{i+1}/{len(eval_dataset)}] GT: {ground_truth} | Pred: {predicted}")

    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(eval_dataset)} ({time.time() - start:.1f}s)")

print(f"\nInference completed in {time.time() - start:.1f}s")

# 4. Compute metrics
print("\n[4/4] Computing metrics...")

y_true = [r["ground_truth"] for r in results]
y_pred = [r["predicted"] for r in results]

# No defaults — empty predictions are penalized naturally
parse_failures = sum(1 for p in y_pred if not p)

# Get all unique labels
all_labels = sorted(set(l for labels in y_true + y_pred for l in labels))
mlb = MultiLabelBinarizer(classes=all_labels)
y_true_bin = mlb.fit_transform(y_true)
y_pred_bin = mlb.transform(y_pred)

# Metrics
micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
samples_f1 = f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
micro_precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
micro_recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)

# Detection metrics (claim vs no-claim)
y_true_detect = [0 if cats == ["0_0_0"] else 1 for cats in y_true]
y_pred_detect = [0 if cats == ["0_0_0"] else 1 for cats in y_pred]
detect_f1 = f1_score(y_true_detect, y_pred_detect, average='binary', zero_division=0)
detect_precision = precision_score(y_true_detect, y_pred_detect, average='binary', zero_division=0)
detect_recall = recall_score(y_true_detect, y_pred_detect, average='binary', zero_division=0)


print("\n" + "=" * 60)
print("RESULTS: cards_qwen35_4b_sft (1 epoch)")
print("=" * 60)
print(f"\nClassification Metrics:")
print(f"  Micro F1:      {micro_f1:.4f}")
print(f"  Macro F1:      {macro_f1:.4f}")
print(f"  Weighted F1:   {weighted_f1:.4f}")
print(f"  Samples F1:    {samples_f1:.4f}")
print(f"  Micro Prec:    {micro_precision:.4f}")
print(f"  Micro Recall:  {micro_recall:.4f}")
print(f"\nDetection Metrics (claim vs no-claim):")
print(f"  F1:            {detect_f1:.4f}")
print(f"  Precision:     {detect_precision:.4f}")
print(f"  Recall:        {detect_recall:.4f}")
print(f"\nParse failures:  {parse_failures}/{len(results)}")
print(f"Unique predicted categories: {len(set(c for r in results for c in r['predicted']))}")
print(f"Unique true categories:      {len(set(c for r in results for c in r['ground_truth']))}")

# Save detailed results
output = {
    "model": "cards_qwen35_4b_sft",
    "epochs": 1,
    "eval_examples": len(results),
    "metrics": {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "samples_f1": samples_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "detect_f1": detect_f1,
        "detect_precision": detect_precision,
        "detect_recall": detect_recall,
        "parse_failures": parse_failures,
    },
    "predictions": results,
}

# Save results — print as JSON so we can capture from logs
print("\n=== RESULTS JSON ===")
print(json.dumps(output))
print("=== END RESULTS JSON ===")
print("\nDone!")

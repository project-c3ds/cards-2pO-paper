# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl>=0.12.0",
#     "huggingface_hub[hf_transfer]",
#     "tensorboard",
#     "transformers>=5.2.0",
#     "flash-linear-attention",
# ]
# ///

import os
import sys
import subprocess
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Check GPU and CUDA driver version
subprocess.run(["nvidia-smi"], check=False)

# Force reinstall torch with cu128 to match CUDA driver
subprocess.run([
    "uv", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu128",
    "--reinstall",
    "--python", sys.executable,
], check=True)

# Verify CUDA is now available
import torch
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HF_USERNAME = "iRanadheer"
DATASET_REPO = f"{HF_USERNAME}/cards_sft_dataset"
MAX_SEQ_LENGTH = 4096

# 1. Load model
print("[1/5] Loading Qwen3.5-4B...")
start = time.time()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(f"Model loaded in {time.time() - start:.1f}s")

# 2. Load dataset
print("\n[2/5] Loading dataset...")
start = time.time()

full_dataset = load_dataset(DATASET_REPO, data_files="cards_train.jsonl", split="train")
split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

def apply_template(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(apply_template, batched=True, remove_columns=["messages"])
eval_dataset = eval_dataset.map(apply_template, batched=True, remove_columns=["messages"])

print(f"Sample (first 200 chars): {train_dataset[0]['text'][:200]}")
print(f"Dataset ready in {time.time() - start:.1f}s")

# 3. Configure trainer
print("\n[3/5] Configuring trainer...")

config = SFTConfig(
    output_dir="cards_qwen35_v3",
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=f"{HF_USERNAME}/cards_qwen35_v3",
    hub_private_repo=True,

    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=MAX_SEQ_LENGTH,

    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=6,

    eval_strategy="steps",
    eval_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    warmup_steps=10,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    seed=42,
    bf16=True,

    report_to=["tensorboard"],
    run_name="qwen35-4b-sft-v3-full-loss",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
)

# No train_on_responses_only — full sequence loss
# Research shows this helps for classification with high prompt-to-completion ratio

# 4. Train
print("\n[4/5] Training...")
start = time.time()
train_result = trainer.train()
train_time = time.time() - start

print(f"\nTraining completed in {train_time / 60:.1f} minutes")
train_loss = train_result.metrics.get("train_loss")
if train_loss:
    print(f"  Final train loss: {train_loss:.4f}")

print("\nRunning final evaluation...")
try:
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")
    if eval_loss:
        print(f"  Final eval loss: {eval_loss:.4f}")
except Exception as e:
    print(f"  Eval failed: {e}")

# 5. Save and push (best checkpoint auto-loaded)
print("\n[5/5] Pushing best checkpoint to Hub...")
trainer.push_to_hub()

print(f"\nModel saved: https://huggingface.co/{HF_USERNAME}/cards_qwen35_v3")

# =========================================================================
# 6. Benchmark evaluation on congress_test.csv
# =========================================================================
print("\n[6/7] Running benchmark evaluation...")

import re
import json
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

# --- Load test data and taxonomy from Hub ---
test_csv_path = hf_hub_download(repo_id=DATASET_REPO, filename="cards_val.jsonl", repo_type="dataset")
taxonomy_csv_path = hf_hub_download(repo_id=DATASET_REPO, filename="taxonomy.csv", repo_type="dataset")

# Build slim codebook from taxonomy
df_tax = pd.read_csv(taxonomy_csv_path)
slim_lines = []
for _, row in df_tax.drop_duplicates('category_number').iterrows():
    code = row['category_number']
    label = row['short_label'] if pd.notna(row.get('short_label')) and str(row.get('short_label', '')).strip() else row['prompt_label']
    slim_lines.append(f'<{code}> {label}')
slim_codebook = "\n".join(slim_lines)

INSTRUCTION_TEMPLATE = """You are an expert in climate communication. Your task is to classify the given text into categories based on the provided codebook. This is a multi-label classification task.

### CODEBOOK:
{codebook}

### INSTRUCTIONS:

1. **Hierarchical Classification**:
   - The codebook is hierarchical. Superclaims end with `_0_0`, subclaims end with `_0`.
   - First check if the text fits `0_0_0` (no relevant claim). If so, assign only that category.
   - Otherwise, scan every superclaim group (1_ through 7_) and list all plausible codes.
   - Then verify each candidate — keep or remove — to arrive at the final set.

2. **Precision and Recall**:
   - Do not leave any relevant claim unassigned.
   - Do not assign any irrelevant claim.

3. **Irrelevant Text**:
   - If the text does not express climate skepticism, promote fossil fuels, or attack renewables, use `0_0_0`.
   - `0_0_0` is mutually exclusive with all other categories.

4. **Description vs Endorsement**:
   - Only classify claims the text actively endorses or promotes.
   - Meta-commentary or criticism of skeptical arguments should be `0_0_0`.

5. **Granularity Rule**:
   - When a text matches both a parent and its subcategories, only include the most specific subcategories.
   - When unsure between a parent and subclaim, ask: does the text explicitly make the subclaim's specific argument? If yes, use the subclaim. If the text is broader or vaguer, use the parent. Do not deliberate — decide and move on.

6. **Cross-Reference Hints**:
   - Economic impacts (4_X_X) often overlap with fossil fuel benefits (7_X_X).
   - Science uncertain (5_X_X) often overlaps with proponents corrupt (6_X_X).
   - Global cooling / natural variation: also check natural drivers (2_1_0, 2_1_1, 2_1_3).
   - Renewable energy feasibility: check both 4_2_7_2 and 7_3_0.
   - Fossil fuel benefits: check all 7_X_X claims.

### OUTPUT FORMAT:
Reason inside <think> tags following this structure, then output YAML:

<think>
1. CLAIMS: Direct quotes only. No paraphrasing, no commentary, no analysis.
2. CONTEXT: One line. Text type, tone, intent. Sincere or satire/irony?
3. SCAN: Go through each superclaim group (1_ through 7_). For each group, state "not relevant" or list all plausible codes.
4. VERIFY: One line per code from SCAN. Format: "[code]: KEEP/REMOVE — [max 10 words why]." Then state final codes.
</think>
```yaml
categories:
  - <category_code>
```

STRICT RULES:
- All reasoning must be inside <think> tags. Nothing after </think> except the YAML block.
- Be concise. VERIFY entries must be one line each.
"""

eval_system_prompt = INSTRUCTION_TEMPLATE.format(codebook=slim_codebook)
COT_TRIGGER = "Let's work this out in a step by step way to be sure we have the right answer."

# --- Switch to inference mode ---
FastLanguageModel.for_inference(model)

# --- Helpers ---
def parse_response(response):
    if '</think>' in response:
        after_think = response.split('</think>')[-1].strip()
    else:
        after_think = response
    cat_match = re.search(r'categories:\s*\n((?:\s*-\s*.+\n?)+)', after_think)
    if cat_match:
        return re.findall(r'-\s*([\d_]+)', cat_match.group(1))
    return []

def parse_true_claims(val):
    if isinstance(val, list):
        return sorted(val)
    # Labels are stored as numpy-style lists: ['4_1_1' '4_2_4']
    return sorted(re.findall(r'[\d_]+', str(val)))

# --- Run inference ---
df_test = pd.read_json(test_csv_path, lines=True)
# true_claims already parsed as lists from JSONL
print(f"  Test samples: {len(df_test)}")

BATCH_SIZE = 32

# Prepare all prompts
all_prompts = []
for _, row in df_test.iterrows():
    messages = [
        {"role": "system", "content": eval_system_prompt},
        {"role": "user", "content": f"### Text:\n{row['text']}\n\n{COT_TRIGGER}"},
    ]
    all_prompts.append(tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    ))

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

responses = []
start = time.time()

for i in range(0, len(all_prompts), BATCH_SIZE):
    batch_prompts = all_prompts[i:i + BATCH_SIZE]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_lengths = (inputs['attention_mask']).sum(dim=1)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False,
        )

    for j, (out, in_len) in enumerate(zip(output_ids, input_lengths)):
        generated = tokenizer.decode(out[in_len:], skip_special_tokens=True)
        responses.append(generated)

    if (i + len(batch_prompts)) % 25 < BATCH_SIZE or i + BATCH_SIZE >= len(all_prompts):
        print(f"  Processed {min(i + len(batch_prompts), len(all_prompts))}/{len(all_prompts)}")

eval_time = time.time() - start
print(f"  Inference completed in {eval_time / 60:.1f} minutes")

df_results = df_test.copy()
df_results['response'] = responses
df_results['pred_claims'] = df_results['response'].map(parse_response)
# true_claims already proper lists from JSONL, no parsing needed

parse_failures = (df_results['pred_claims'].map(len) == 0).sum()
print(f"  Parse failures: {parse_failures}/{len(df_results)} ({parse_failures/len(df_results)*100:.1f}%) — penalized as empty predictions")

# =========================================================================
# 7. Compute metrics at all hierarchy levels and save reports
# =========================================================================
print("\n[7/7] Computing metrics and saving reports...")

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

EVAL_DIR = "eval_reports"
os.makedirs(EVAL_DIR, exist_ok=True)

MIN_SUPPORT_THRESHOLDS = [3, 10]

def get_level_code(code, level=3):
    parts = code.split('_')
    return '_'.join(parts[:level])

def compute_metrics(df, level=3, min_support=3):
    y_true = df['true_claims'].apply(lambda x: list(set(get_level_code(c, level) for c in x))).tolist()
    y_pred = df['pred_claims'].apply(lambda x: list(set(get_level_code(c, level) for c in x))).tolist()

    mlb = MultiLabelBinarizer()
    all_labels = set()
    for labels in y_true + y_pred:
        all_labels.update(labels)
    mlb.fit([sorted(all_labels)])

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Filter out low-support labels
    label_support = y_true_bin.sum(axis=0)
    high_support_mask = label_support >= min_support
    high_support_labels = [l for l, keep in zip(mlb.classes_, high_support_mask) if keep]
    dropped_labels = [l for l, keep in zip(mlb.classes_, high_support_mask) if not keep]

    y_true_filtered = y_true_bin[:, high_support_mask]
    y_pred_filtered = y_pred_bin[:, high_support_mask]

    report_text = classification_report(
        y_true_filtered, y_pred_filtered,
        target_names=high_support_labels,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true_filtered, y_pred_filtered,
        target_names=high_support_labels,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        'samples_f1': round(f1_score(y_true_filtered, y_pred_filtered, average='samples', zero_division=0), 3),
        'macro_f1': round(f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0), 3),
        'micro_f1': round(f1_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0), 3),
        'micro_precision': round(precision_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0), 3),
        'micro_recall': round(recall_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0), 3),
        'accuracy': round(accuracy_score(y_true_filtered, y_pred_filtered), 3),
        'hamming_loss': round(hamming_loss(y_true_filtered, y_pred_filtered), 3),
        'n_labels': len(high_support_labels),
        'dropped_labels': dropped_labels,
    }

    return metrics, report_text, report_dict

all_metrics = {}
for min_sup in MIN_SUPPORT_THRESHOLDS:
    print(f"\n  ===== min_support={min_sup} =====")
    for level in [1, 2, 3]:
        metrics, report_text, report_dict = compute_metrics(df_results, level=level, min_support=min_sup)
        key = f'level_{level}_minsup_{min_sup}'
        all_metrics[key] = metrics

        print(f"\n  --- Level {level} (min_support={min_sup}) ---")
        print(f"  Micro F1: {metrics['micro_f1']}, Macro F1: {metrics['macro_f1']}, Samples F1: {metrics['samples_f1']}")
        print(f"  Precision: {metrics['micro_precision']}, Recall: {metrics['micro_recall']}")
        print(f"  Exact Match: {metrics['accuracy']}, Hamming Loss: {metrics['hamming_loss']}")

        with open(os.path.join(EVAL_DIR, f"classification_report_level_{level}_minsup_{min_sup}.txt"), "w") as f:
            f.write(f"Classification Report — Level {level} (min_support={min_sup})\n")
            f.write("=" * 80 + "\n\n")
            f.write(report_text)

        with open(os.path.join(EVAL_DIR, f"classification_report_level_{level}_minsup_{min_sup}.json"), "w") as f:
            json.dump(report_dict, f, indent=2)

# Save aggregated metrics
with open(os.path.join(EVAL_DIR, "metrics_summary.json"), "w") as f:
    json.dump({
        "dataset": "cards_val",
        "n_samples": len(df_results),
        "min_support_thresholds": MIN_SUPPORT_THRESHOLDS,
        "eval_time_minutes": round(eval_time / 60, 1),
        "train_loss": train_loss,
        "levels": all_metrics,
    }, f, indent=2)

# Save raw results
df_results.to_csv(os.path.join(EVAL_DIR, "eval_results.csv"), index=False)

# --- Upload eval_reports folder to Hub ---
api = HfApi()
api.upload_folder(
    folder_path=EVAL_DIR,
    path_in_repo="eval_reports",
    repo_id=f"{HF_USERNAME}/cards_qwen35_v3",
    repo_type="model",
    commit_message=f"Add eval reports: L3 micro_f1={all_metrics['level_3']['micro_f1']}",
)

print(f"\n  Reports uploaded to: https://huggingface.co/{HF_USERNAME}/cards_qwen35_v3/tree/main/eval_reports")
print(f"\nDone! Training + evaluation complete.")

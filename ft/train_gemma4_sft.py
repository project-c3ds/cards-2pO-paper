# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl>=0.12.0",
#     "huggingface_hub[hf_transfer]",
#     "tensorboard",
#     "transformers>=5.2.0",
# ]
# ///

"""
CARDS Gemma4 SFT Training

Usage:
    python train_gemma4_sft.py --model e4b
    python train_gemma4_sft.py --model 31b
    python train_gemma4_sft.py --model e4b --org c3ds
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
MODELS = {
    "e2b": "unsloth/gemma-4-E2B-it",
    "e4b": "unsloth/gemma-4-E4B-it",
    "31b": "unsloth/gemma-4-31B-it",
}

parser = argparse.ArgumentParser(description="CARDS Gemma4 SFT Training")
parser.add_argument("--model", type=str, default="e4b", choices=MODELS.keys(),
                    help="Model size (default: e4b)")
parser.add_argument("--org", type=str, default="iRanadheer",
                    help="HuggingFace org/user for model repo (default: iRanadheer)")
parser.add_argument("--merge-and-push", action="store_true", default=False,
                    help="Merge LoRA with base model and push full model to Hub")
args = parser.parse_args()

MODEL_SIZE = args.model
BASE_MODEL = MODELS[MODEL_SIZE]
VARIANT = f"gemma4_{MODEL_SIZE}_recot_full"

print(f"{'='*60}")
print(f"  Variant: {VARIANT}")
print(f"  Base model: {BASE_MODEL}")
print(f"  Org: {args.org}")
print(f"  Merge and push: {args.merge_and_push}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import subprocess
subprocess.run(["nvidia-smi"], check=False)

import torch
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DATASET_REPO = "iRanadheer/cards_sft_dataset"
MODEL_REPO = f"{args.org}/cards_{VARIANT}"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = f"cards_{VARIANT}"

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print(f"[1/5] Loading {BASE_MODEL}...")
start = time.time()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(f"Model loaded in {time.time() - start:.1f}s")

# ---------------------------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------------------------
print("\n[2/5] Loading dataset...")
start = time.time()

HF_TOKEN = os.environ.get("HF_TOKEN")
train_dataset = load_dataset(DATASET_REPO, data_files="cards_train.jsonl", split="train", token=HF_TOKEN)
eval_dataset = load_dataset(DATASET_REPO, data_files="cards_train_eval.jsonl", split="train", token=HF_TOKEN)
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Gemma 4 uses standard system/user/assistant roles
def apply_template(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(apply_template, batched=True, remove_columns=["messages"])
eval_dataset = eval_dataset.map(apply_template, batched=True, remove_columns=["messages"])

print(f"Sample (first 200 chars): {train_dataset[0]['text'][:200]}")
print(f"Dataset ready in {time.time() - start:.1f}s")

# ---------------------------------------------------------------------------
# 3. Configure trainer (same hyperparams across all sizes for fair comparison)
# ---------------------------------------------------------------------------
print(f"\n[3/5] Configuring trainer...")

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=MODEL_REPO,
    hub_private_repo=True,

    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
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
    run_name=f"cards-{VARIANT}",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
)

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
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
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 5. Save and push
# ---------------------------------------------------------------------------
print("\n[5/5] Pushing best checkpoint to Hub...")
try:
    trainer.push_to_hub()
    print(f"\nModel saved: https://huggingface.co/{MODEL_REPO}")
except Exception as e:
    print(f"  push_to_hub failed: {e}")
    print("  Saving locally and uploading manually...")
    trainer.save_model(OUTPUT_DIR)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(MODEL_REPO, private=True, exist_ok=True)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=MODEL_REPO, repo_type="model",
                      commit_message=f"Upload {VARIANT} model")
    print(f"\nModel saved: https://huggingface.co/{MODEL_REPO}")

if args.merge_and_push:
    MERGED_REPO = f"{args.org}/cards_{VARIANT}_merged"
    print(f"\nMerging LoRA and pushing full model to {MERGED_REPO}...")
    try:
        merged_dir = f"{OUTPUT_DIR}_merged"
        model.save_pretrained_merged(merged_dir, tokenizer)
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(MERGED_REPO, private=True, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=MERGED_REPO, repo_type="model",
                          commit_message=f"Upload merged {VARIANT} model")
        print(f"  Merged model saved: https://huggingface.co/{MERGED_REPO}")
    except Exception as e:
        print(f"  Merge+push failed: {e}")

print(f"\nDone! Training complete ({VARIANT}).")

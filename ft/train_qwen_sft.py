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
#     "scikit-learn",
#     "pandas",
# ]
# ///

"""
CARDS Qwen3.5-4B SFT Training + Benchmark Evaluation

Ablation variants (controlled via CLI args):
  --recot / --no-recot       Use RECoT (chain-of-thought) training data or YAML-only
  --full-loss / --resp-only   Train on full sequence or responses only

Examples:
  python train_qwen_sft.py --recot --full-loss
  python train_qwen_sft.py --no-recot --resp-only
"""

import argparse
import os
import sys
import subprocess
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
MODELS = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
    "27b": "Qwen/Qwen3.5-27B",
}

parser = argparse.ArgumentParser(description="CARDS SFT Training")
parser.add_argument("--model", type=str, default="4b", choices=MODELS.keys(),
                    help="Model size (default: 4b)")
parser.add_argument("--recot", dest="recot", action="store_true", default=True,
                    help="Use RECoT training data (default)")
parser.add_argument("--no-recot", dest="recot", action="store_false",
                    help="Use non-RECoT (YAML-only) training data")
parser.add_argument("--full-loss", dest="full_loss", action="store_true", default=True,
                    help="Train on full sequence loss (default)")
parser.add_argument("--resp-only", dest="full_loss", action="store_false",
                    help="Train on responses only")
parser.add_argument("--merge-and-push", action="store_true", default=False,
                    help="Merge LoRA with base model and push full model to Hub")
args = parser.parse_args()

# Derive variant name for model repo and output dir
recot_tag = "recot" if args.recot else "norecot"
loss_tag = "full" if args.full_loss else "resp"
MODEL_SIZE = args.model
BASE_MODEL = MODELS[MODEL_SIZE]
VARIANT = f"{MODEL_SIZE}_{recot_tag}_{loss_tag}"

print(f"{'='*60}")
print(f"  Variant: {VARIANT}")
print(f"  Base model: {BASE_MODEL}")
print(f"  RECoT: {args.recot}, Full loss: {args.full_loss}")
print(f"  Merge and push: {args.merge_and_push}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
subprocess.run(["nvidia-smi"], check=False)

subprocess.run([
    "uv", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu128",
    "--reinstall",
    "--python", sys.executable,
], check=True)

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
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HF_USERNAME = "iRanadheer"
DATASET_REPO = f"{HF_USERNAME}/cards_sft_dataset"
MODEL_REPO = f"{HF_USERNAME}/cards_qwen35_{VARIANT}"
MAX_SEQ_LENGTH = 4096

# Training data file based on variant
TRAIN_FILE = "cards_train.jsonl" if args.recot else "cards_train_norecot.jsonl"
EVAL_FILE = "cards_train_eval.jsonl" if args.recot else "cards_train_eval_norecot.jsonl"

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print(f"[1/7] Loading {BASE_MODEL}...")
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
print(f"\n[2/7] Loading dataset ({TRAIN_FILE})...")
start = time.time()

train_dataset = load_dataset(DATASET_REPO, data_files=TRAIN_FILE, split="train")
eval_dataset = load_dataset(DATASET_REPO, data_files=EVAL_FILE, split="train")
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

# ---------------------------------------------------------------------------
# 3. Configure trainer
# ---------------------------------------------------------------------------
print(f"\n[3/7] Configuring trainer (variant={VARIANT})...")

OUTPUT_DIR = f"cards_qwen35_{VARIANT}"

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=MODEL_REPO,
    hub_private_repo=True,

    num_train_epochs=3,
    per_device_train_batch_size=2,
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
    run_name=f"qwen35-{VARIANT}",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
)

# Apply train_on_responses_only if resp-only variant
if not args.full_loss:
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    print("  train_on_responses_only enabled")

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
print(f"\n[4/7] Training ({VARIANT})...")
start = time.time()
train_result = trainer.train()
train_time = time.time() - start

print(f"\nTraining completed in {train_time / 60:.1f} minutes")
train_loss = train_result.metrics.get("train_loss")
if train_loss:
    print(f"  Final train loss: {train_loss:.4f}")

print("\nRunning final evaluation...")
eval_loss = None
try:
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")
    if eval_loss:
        print(f"  Final eval loss: {eval_loss:.4f}")
except Exception as e:
    print(f"  Eval failed: {e}")
    # Reset CUDA state so push_to_hub doesn't fail
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 5. Save and push
# ---------------------------------------------------------------------------
print(f"\n[5/7] Pushing best checkpoint to Hub...")
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

# ---------------------------------------------------------------------------
# 5b. Merge and push full model (optional)
# ---------------------------------------------------------------------------
if args.merge_and_push:
    MERGED_REPO = f"{HF_USERNAME}/cards_qwen35_{VARIANT}_merged"
    print(f"\n[5b/7] Merging LoRA and pushing full model to {MERGED_REPO}...")
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

print(f"\nDone! Training complete ({VARIANT}). Run eval locally with eval_benchmark.py.")

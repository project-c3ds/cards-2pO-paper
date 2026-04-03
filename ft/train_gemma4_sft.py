"""
Train Gemma-4-31B with Unsloth LoRA on single H200 GPU.

Usage:
    pip install unsloth datasets "trl>=0.12.0" "huggingface_hub[hf_transfer]" tensorboard
    huggingface-cli login
    tmux new -d -s train "python train_gemma4_sft.py 2>&1 | tee train.log"
    tmux attach -t train
"""

import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import subprocess
subprocess.run(["nvidia-smi"], check=False)

import torch
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HF_USERNAME = "iRanadheer"
DATASET_REPO = f"{HF_USERNAME}/cards_sft_dataset"
MODEL_REPO = f"{HF_USERNAME}/cards_gemma4_31b_recot_full"
BASE_MODEL = "unsloth/gemma-4-31B-it"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "cards_gemma4_31b_recot_full"

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print("[1/5] Loading Gemma-4-31B-IT...")
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

# Gemma 4 uses standard system/user/assistant roles (unlike Gemma 3 which used "model")
# No conversion needed — our dataset already uses these roles.
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
# 3. Configure trainer
# ---------------------------------------------------------------------------
print("\n[3/5] Configuring trainer...")

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
    run_name="gemma4-31b-sft-recot-full",
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
                      commit_message="Upload Gemma 4 31B model")
    print(f"\nModel saved: https://huggingface.co/{MODEL_REPO}")

print(f"\nDone! Training complete.")

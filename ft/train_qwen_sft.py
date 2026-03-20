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
import time

# Force unbuffered output for HF Jobs logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HF_USERNAME = "iRanadheer"
MAX_SEQ_LENGTH = 2560

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
    lora_alpha=64,
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

train_dataset = load_dataset("iRanadheer/cards_sft_dataset", data_files="qwen_sft_train.jsonl", split="train")
eval_dataset = load_dataset("iRanadheer/cards_sft_dataset", data_files="qwen_sft_eval.jsonl", split="train")
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Apply chat template to convert messages -> text
def apply_template(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        # Remove BOS token to avoid duplicates
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(apply_template, batched=True, remove_columns=["messages"])
eval_dataset = eval_dataset.map(apply_template, batched=True, remove_columns=["messages"])

print(f"Sample (first 200 chars): {train_dataset[0]['text'][:200]}")
print(f"Sample (last 200 chars): {train_dataset[0]['text'][-200:]}")
print(f"Dataset ready in {time.time() - start:.1f}s")

# 3. Configure trainer
print("\n[3/5] Configuring trainer...")

config = SFTConfig(
    output_dir="cards_qwen35_4b_sft",
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=f"{HF_USERNAME}/cards_qwen35_4b_sft",
    hub_private_repo=True,

    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=MAX_SEQ_LENGTH,

    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,

    eval_strategy="steps",
    eval_steps=25,

    warmup_steps=3,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    seed=42,
    bf16=True,

    report_to=["tensorboard"],
    run_name="qwen35-4b-sft-v1",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
)

# Train only on assistant responses (mask system prompt + user text from loss)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# 4. Train
print("\n[4/5] Training...")
start = time.time()
train_result = trainer.train()
train_time = time.time() - start

print(f"\nTraining completed in {train_time / 60:.1f} minutes")
train_loss = train_result.metrics.get("train_loss")
if train_loss:
    print(f"  Final train loss: {train_loss:.4f}")

# Final eval
print("\nRunning final evaluation...")
try:
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")
    if eval_loss:
        print(f"  Final eval loss: {eval_loss:.4f}")
except Exception as e:
    print(f"  Eval failed: {e}")

# 5. Save and push
print("\n[5/5] Pushing to Hub...")
trainer.push_to_hub()

print(f"\nDone! Model at: https://huggingface.co/{HF_USERNAME}/cards_qwen35_4b_sft")

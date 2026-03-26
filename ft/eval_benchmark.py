# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm>=0.17.1",
#     "peft",
#     "huggingface_hub[hf_transfer]",
#     "scikit-learn",
#     "pandas",
#     "openai",
# ]
# ///

"""
CARDS Benchmark Evaluation — merges LoRA, serves with vLLM, runs val benchmark.

Evaluates all 4 ablation variants sequentially:
  1. Merge LoRA adapter with base model
  2. Serve merged model with vLLM
  3. Run concurrent inference on cards_val.jsonl
  4. Compute metrics, save reports, upload to Hub
  5. Shut down vLLM, free memory, repeat for next variant

Usage:
    python eval_benchmark.py
    python eval_benchmark.py --variants recot_full norecot_full
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

subprocess.run(["nvidia-smi"], check=False)

import torch
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import login, hf_hub_download, HfApi
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss, classification_report,
)
from sklearn.preprocessing import MultiLabelBinarizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variants", nargs="+",
                    default=["recot_full", "recot_resp", "norecot_full", "norecot_resp"])
parser.add_argument("--max-workers", type=int, default=50)
parser.add_argument("--vllm-port", type=int, default=8000)
args = parser.parse_args()

HF_USERNAME = "iRanadheer"
DATASET_REPO = f"{HF_USERNAME}/cards_sft_dataset"
BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH = 4096
MIN_SUPPORT_THRESHOLDS = [3, 10]

RECOT_VARIANTS = {"recot_full", "recot_resp"}

# ---------------------------------------------------------------------------
# Load val data + taxonomy (once)
# ---------------------------------------------------------------------------
print("Loading val data and taxonomy...")
val_path = hf_hub_download(repo_id=DATASET_REPO, filename="cards_val.jsonl", repo_type="dataset")
taxonomy_path = hf_hub_download(repo_id=DATASET_REPO, filename="taxonomy.csv", repo_type="dataset")

df_val = pd.read_json(val_path, lines=True)
print(f"  Val samples: {len(df_val)}")

df_tax = pd.read_csv(taxonomy_path)
slim_lines = []
for _, row in df_tax.drop_duplicates('category_number').iterrows():
    code = row['category_number']
    label = (row['short_label']
             if pd.notna(row.get('short_label')) and str(row.get('short_label', '')).strip()
             else row['prompt_label'])
    slim_lines.append(f'<{code}> {label}')
slim_codebook = "\n".join(slim_lines)

SYSTEM_PROMPT = """You are an expert in climate communication. Your task is to classify the given text into categories based on the provided codebook. This is a multi-label classification task.

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
""".format(codebook=slim_codebook)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_response(response):
    if '</think>' in response:
        after_think = response.split('</think>')[-1].strip()
    else:
        after_think = response
    cat_match = re.search(r'categories:\s*\n((?:\s*-\s*.+\n?)+)', after_think)
    if cat_match:
        return re.findall(r'-\s*([\d_]+)', cat_match.group(1))
    return []


def get_level_code(code, level=3):
    return '_'.join(code.split('_')[:level])


def compute_metrics(df, level=3, min_support=3):
    y_true = df['true_claims'].apply(
        lambda x: list(set(get_level_code(c, level) for c in x))).tolist()
    y_pred = df['pred_claims'].apply(
        lambda x: list(set(get_level_code(c, level) for c in x))).tolist()

    mlb = MultiLabelBinarizer()
    all_labels = set()
    for labels in y_true + y_pred:
        all_labels.update(labels)
    mlb.fit([sorted(all_labels)])

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    label_support = y_true_bin.sum(axis=0)
    mask = label_support >= min_support
    labels = [l for l, k in zip(mlb.classes_, mask) if k]
    dropped = [l for l, k in zip(mlb.classes_, mask) if not k]

    y_true_f = y_true_bin[:, mask]
    y_pred_f = y_pred_bin[:, mask]

    report_text = classification_report(
        y_true_f, y_pred_f, target_names=labels, zero_division=0)
    report_dict = classification_report(
        y_true_f, y_pred_f, target_names=labels, zero_division=0, output_dict=True)

    metrics = {
        'samples_f1': round(f1_score(y_true_f, y_pred_f, average='samples', zero_division=0), 3),
        'macro_f1': round(f1_score(y_true_f, y_pred_f, average='macro', zero_division=0), 3),
        'micro_f1': round(f1_score(y_true_f, y_pred_f, average='micro', zero_division=0), 3),
        'micro_precision': round(precision_score(y_true_f, y_pred_f, average='micro', zero_division=0), 3),
        'micro_recall': round(recall_score(y_true_f, y_pred_f, average='micro', zero_division=0), 3),
        'accuracy': round(accuracy_score(y_true_f, y_pred_f), 3),
        'hamming_loss': round(hamming_loss(y_true_f, y_pred_f), 3),
        'n_labels': len(labels),
        'dropped_labels': dropped,
    }
    return metrics, report_text, report_dict


def merge_lora_model(adapter_repo, output_dir):
    """Merge LoRA adapter with base model using peft and save locally."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gc

    print(f"    Loading base model {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",  # merge on CPU to save GPU memory for vLLM
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"    Loading adapter from {adapter_repo}...")
    model = PeftModel.from_pretrained(model, adapter_repo)

    print(f"    Merging and saving to {output_dir}...")
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Free memory
    del model, tokenizer
    gc.collect()
    print(f"    Merge complete.")


def start_vllm(model_dir, port):
    """Start vLLM server in background, wait until ready."""
    print(f"    Starting vLLM on port {port}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir,
            "--port", str(port),
            "--max-model-len", str(MAX_SEQ_LENGTH),
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.90",
            "--enable-prefix-caching",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready (max 5 min)
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://localhost:{port}/health")
            if resp.status == 200:
                print(f"    vLLM ready (pid={proc.pid})")
                return proc
        except Exception:
            pass
        # Check if process died
        if proc.poll() is not None:
            output = proc.stdout.read().decode() if proc.stdout else ""
            print(f"    vLLM failed to start:\n{output[-1000:]}")
            raise RuntimeError("vLLM process exited")
        time.sleep(3)

    proc.kill()
    raise TimeoutError("vLLM did not start within 5 minutes")


def stop_vllm(proc):
    """Stop vLLM server and wait for cleanup."""
    if proc is None:
        return
    print(f"    Stopping vLLM (pid={proc.pid})...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    # Give GPU time to release
    time.sleep(3)
    torch.cuda.empty_cache()
    print(f"    vLLM stopped, GPU memory freed.")


def run_inference(variant, port, max_workers):
    """Run concurrent inference via vLLM OpenAI API."""
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    use_recot = variant in RECOT_VARIANTS

    if use_recot:
        user_template = "### Text:\n{text}\n\nLet's work this out in a step by step way to be sure we have the right answer."
    else:
        user_template = "### Text:\n{text}"

    def process_row(row):
        row = row.copy()
        try:
            response = client.chat.completions.create(
                model=row['_model_name'],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_template.format(text=row['text'])},
                ],
                temperature=0,
                max_tokens=2048,
            )
            row['response'] = response.choices[0].message.content
        except Exception as e:
            row['response'] = f"ERROR: {e}"
        return row

    # Get model name from vLLM
    models = client.models.list()
    model_name = models.data[0].id
    print(f"    vLLM serving model: {model_name}")

    rows = []
    for _, row in df_val.iterrows():
        r = row.to_dict()
        r['_model_name'] = model_name
        rows.append(r)

    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, r): i for i, r in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 100 == 0 or done == len(rows):
                print(f"    {done}/{len(rows)}")

    elapsed = time.time() - start
    print(f"    Inference completed in {elapsed / 60:.1f} minutes ({len(rows)/elapsed:.1f} samples/sec)")

    df_results = pd.DataFrame(results).drop(columns=['_model_name'])
    df_results['pred_claims'] = df_results['response'].map(parse_response)

    parse_failures = (df_results['pred_claims'].map(len) == 0).sum()
    error_count = df_results['response'].str.startswith('ERROR:').sum()
    print(f"    Parse failures: {parse_failures}/{len(df_results)} ({parse_failures/len(df_results)*100:.1f}%)")
    if error_count:
        print(f"    API errors: {error_count}/{len(df_results)}")

    return df_results, elapsed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
api = HfApi()
summary_all = {}

for variant in args.variants:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {variant}")
    print(f"{'='*60}")

    model_repo = f"{HF_USERNAME}/cards_qwen35_{variant}"
    merged_dir = f"/tmp/merged_{variant}"
    vllm_proc = None

    try:
        # 1. Merge LoRA
        print(f"\n  [1/4] Merging LoRA weights...")
        merge_lora_model(model_repo, merged_dir)

        # 2. Start vLLM
        print(f"\n  [2/4] Starting vLLM server...")
        vllm_proc = start_vllm(merged_dir, args.vllm_port)

        # 3. Run inference
        print(f"\n  [3/4] Running inference...")
        df_results, eval_time = run_inference(variant, args.vllm_port, args.max_workers)

        # 4. Compute metrics and save
        print(f"\n  [4/4] Computing metrics...")
        eval_dir = f"eval_reports_{variant}"
        os.makedirs(eval_dir, exist_ok=True)

        all_metrics = {}
        for min_sup in MIN_SUPPORT_THRESHOLDS:
            print(f"\n    --- min_support={min_sup} ---")
            for level in [1, 2, 3]:
                metrics, report_text, report_dict = compute_metrics(
                    df_results, level=level, min_support=min_sup)
                key = f'level_{level}_minsup_{min_sup}'
                all_metrics[key] = metrics

                print(f"      Level {level}: micro_f1={metrics['micro_f1']}, "
                      f"macro_f1={metrics['macro_f1']}, precision={metrics['micro_precision']}, "
                      f"recall={metrics['micro_recall']}")

                with open(os.path.join(eval_dir, f"classification_report_level_{level}_minsup_{min_sup}.txt"), "w") as f:
                    f.write(f"Classification Report — Level {level} (min_support={min_sup})\n")
                    f.write(f"Variant: {variant}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(report_text)

                with open(os.path.join(eval_dir, f"classification_report_level_{level}_minsup_{min_sup}.json"), "w") as f:
                    json.dump(report_dict, f, indent=2)

        with open(os.path.join(eval_dir, "metrics_summary.json"), "w") as f:
            json.dump({
                "variant": variant,
                "recot": variant in RECOT_VARIANTS,
                "dataset": "cards_val",
                "n_samples": len(df_results),
                "parse_failures": int((df_results['pred_claims'].map(len) == 0).sum()),
                "min_support_thresholds": MIN_SUPPORT_THRESHOLDS,
                "eval_time_minutes": round(eval_time / 60, 1),
                "levels": all_metrics,
            }, f, indent=2)

        df_results.to_csv(os.path.join(eval_dir, "eval_results.csv"), index=False)

        try:
            api.upload_folder(
                folder_path=eval_dir,
                path_in_repo="eval_reports",
                repo_id=model_repo,
                repo_type="model",
                commit_message=f"Add eval reports: L3 micro_f1={all_metrics.get('level_3_minsup_3', {}).get('micro_f1', 'N/A')}",
            )
            print(f"  Uploaded to {model_repo}/eval_reports")
        except Exception as e:
            print(f"  Upload failed: {e}")

        summary_all[variant] = all_metrics

    except Exception as e:
        print(f"\n  ERROR evaluating {variant}: {e}")
        import traceback
        traceback.print_exc()
        summary_all[variant] = {"error": str(e)}

    finally:
        # Always clean up
        stop_vllm(vllm_proc)
        # Remove merged model to free disk
        subprocess.run(["rm", "-rf", merged_dir], check=False)
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("  FINAL RESULTS (Level 3, min_support=3)")
print(f"{'='*60}")
print(f"{'Variant':<20} {'Micro F1':>10} {'Macro F1':>10} {'Precision':>10} {'Recall':>10} {'EM':>10}")
print("-" * 70)
for variant in args.variants:
    m = summary_all.get(variant, {}).get('level_3_minsup_3', {})
    if isinstance(m, dict) and 'micro_f1' in m:
        print(f"{variant:<20} {m['micro_f1']:>10} {m['macro_f1']:>10} "
              f"{m['micro_precision']:>10} {m['micro_recall']:>10} {m['accuracy']:>10}")
    else:
        print(f"{variant:<20} {'FAILED':>10}")

with open("eval_summary_all.json", "w") as f:
    json.dump(summary_all, f, indent=2)

print(f"\nDone! Combined summary saved to eval_summary_all.json")

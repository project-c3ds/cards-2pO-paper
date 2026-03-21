"""Submit a training script to HF Jobs."""

import os
import argparse
from huggingface_hub import HfApi

HF_USERNAME = "iRanadheer"
DATASET_REPO = f"{HF_USERNAME}/cards_sft_dataset"

def main():
    parser = argparse.ArgumentParser(description="Submit training script to HF Jobs")
    parser.add_argument("script", help="Path to local training script")
    parser.add_argument("--flavor", default="a100-large", help="GPU flavor (default: a100-large)")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout in seconds (default: 7200)")
    args = parser.parse_args()

    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    os.environ["HF_TOKEN"] = token

    api = HfApi()

    # Upload script to dataset repo
    script_name = os.path.basename(args.script)
    api.upload_file(
        path_or_fileobj=args.script,
        path_in_repo=script_name,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    script_url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{script_name}"
    print(f"Uploaded: {script_url}")

    # Submit job
    job = api.run_uv_job(
        script=script_url,
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
    )
    print(f"Job ID: {job.id}")
    print(f"Monitor: {job.url}")
    print(f"Status: {job.status.stage}")
    print(f"\nhf jobs logs {job.id}")

if __name__ == "__main__":
    main()

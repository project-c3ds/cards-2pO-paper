"""Reformat training data for Qwen3.5 fine-tuning.

Converts existing Claude 3.5 ReCOT training data to:
- Slim codebook in system prompt
- Text in user turn with CoT trigger
- <think> tags around condensed reasoning + YAML output
- Messages format (role/content) for TRL
"""

import json
import re
import sys
import os
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import slim_codebook

# --- Config ---
INPUT_FILE = os.path.join(os.path.dirname(__file__), "claude_messages_plus_updated_prompt.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "qwen_sft_train.jsonl")
OUTPUT_EVAL_FILE = os.path.join(os.path.dirname(__file__), "qwen_sft_eval.jsonl")
EVAL_RATIO = 0.1
SEED = 42

SYSTEM_PROMPT = f"""You are an expert in climate communication. Classify the given text into categories from the codebook. This is a multi-label classification task.

### CODEBOOK:
{slim_codebook}

### OUTPUT FORMAT:
Reason step by step inside <think> tags, then list ALL matching category codes in YAML:
```yaml
categories:
  - <code>
  - ...
```

### RULES:
- Use 0_0_0 if no relevant climate skepticism claim is detected. It is mutually exclusive with all other categories.
- Return the most granular matching categories.
- Only classify claims the text endorses, not describes."""


def extract_categories(assistant_text: str) -> list[str]:
    """Extract category codes from the assistant response JSON."""
    cats = re.findall(r'"category":\s*"<([^>]+)>"', assistant_text)
    return cats


def extract_review(assistant_text: str) -> str:
    """Extract review field from the assistant response JSON."""
    match = re.search(r'"review":\s*"(yes|no)"', assistant_text)
    return match.group(1) if match else "no"


def condense_cot(assistant_text: str) -> str:
    """Extract core reasoning from verbose CoT, removing boilerplate."""
    # Remove the JSON block at the end
    json_match = re.search(r'\{\s*\n\s*"review"', assistant_text)
    if json_match:
        cot = assistant_text[:json_match.start()].strip()
    else:
        cot = assistant_text.strip()

    # Remove common boilerplate lines
    lines = cot.split('\n')
    filtered = []
    skip_patterns = [
        r'^let me analyze',
        r'^based on this analysis',
        r'^based on my analysis',
        r'^the classification is clear',
        r'^the reasoning is clear',
        r'^therefore,?\s*(it|this|i would)',
        r'^\d+\)\s*(first,?\s*)?let\'?s?\s*(identify|consider|check|break)',
        r'^\d+\)\s*(final|conclusion)',
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(p, stripped.lower()) for p in skip_patterns):
            continue
        filtered.append(stripped)

    # Join and clean up
    result = '\n'.join(filtered)

    # Remove numbered list prefixes for cleaner output
    result = re.sub(r'^\d+\)\s*', '', result, flags=re.MULTILINE)

    # Trim to reasonable length (aim for ~100-200 tokens)
    sentences = re.split(r'(?<=[.!?])\s+', result)
    if len(sentences) > 8:
        result = ' '.join(sentences[:8])

    return result.strip()


def extract_text(messages: list[dict]) -> str:
    """Extract the text to classify from the second system message."""
    text_msg = messages[1]['content'].strip()
    # Remove "### Text:" prefix if present
    text_msg = re.sub(r'^###\s*Text:\s*', '', text_msg).strip()
    # Remove leading artifacts (dots, colons, etc.)
    text_msg = re.sub(r'^[.:]+\s*', '', text_msg).strip()
    return text_msg


def format_yaml_output(categories: list[str]) -> str:
    """Format output as YAML."""
    lines = ["categories:"]
    for cat in categories:
        lines.append(f"  - {cat}")
    return '\n'.join(lines)


def reformat_example(example: dict) -> dict | None:
    """Reformat a single example to the new format."""
    messages = example['messages']

    # Extract components
    text = extract_text(messages)
    assistant_text = messages[3]['content']
    categories = extract_categories(assistant_text)

    if not categories:
        return None

    # Condense CoT and format output
    cot = condense_cot(assistant_text)
    yaml_output = format_yaml_output(categories)

    # Build new assistant response
    assistant_response = f"<think>\n{cot}\n</think>\n{yaml_output}"

    # Build new message format
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{text}\n\nLet's work this out step by step."},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def stratified_split(examples: list[dict], eval_ratio: float, seed: int):
    """Split data ensuring all categories appear in both sets."""
    random.seed(seed)

    # Index examples by their categories (from YAML section only)
    cat_to_indices = {}
    for i, ex in enumerate(examples):
        yaml_section = ex['messages'][2]['content'].split('</think>')[-1]
        cats = re.findall(r'^\s+- (.+)$', yaml_section, re.MULTILINE)
        for cat in cats:
            cat_to_indices.setdefault(cat, []).append(i)

    eval_indices = set()

    # Ensure at least one example per category in eval
    for cat, indices in cat_to_indices.items():
        if len(indices) >= 2:
            # Pick one for eval that isn't already selected
            available = [i for i in indices if i not in eval_indices]
            if available:
                eval_indices.add(random.choice(available))

    # Fill up eval set to target ratio (only add more if we haven't exceeded target)
    target_eval_size = int(len(examples) * eval_ratio)
    if len(eval_indices) < target_eval_size:
        all_indices = [i for i in range(len(examples)) if i not in eval_indices]
        random.shuffle(all_indices)
        for i in all_indices:
            if len(eval_indices) >= target_eval_size:
                break
            eval_indices.add(i)

    train = [ex for i, ex in enumerate(examples) if i not in eval_indices]
    eval_set = [ex for i, ex in enumerate(examples) if i in eval_indices]

    return train, eval_set


def main():
    # Load data
    with open(INPUT_FILE) as f:
        raw_data = [json.loads(line) for line in f]

    print(f"Loaded {len(raw_data)} examples from {INPUT_FILE}")

    # Reformat
    reformatted = []
    failed = 0
    for ex in raw_data:
        result = reformat_example(ex)
        if result:
            reformatted.append(result)
        else:
            failed += 1

    print(f"Reformatted: {len(reformatted)}, Failed: {failed}")

    # Category distribution (from YAML section only)
    cat_counter = Counter()
    for ex in reformatted:
        yaml_section = ex['messages'][2]['content'].split('</think>')[-1]
        cats = re.findall(r'^\s+- (.+)$', yaml_section, re.MULTILINE)
        for cat in cats:
            cat_counter[cat] += 1

    no_claim = sum(1 for ex in reformatted
                   if re.findall(r'^\s+- (.+)$', ex['messages'][2]['content'].split('</think>')[-1], re.MULTILINE) == ['0_0_0'])
    has_claim = len(reformatted) - no_claim
    print(f"No claim: {no_claim}, Has claim: {has_claim}")
    print(f"Unique categories: {len(cat_counter)}")

    # Stratified split
    train, eval_set = stratified_split(reformatted, EVAL_RATIO, SEED)
    print(f"Train: {len(train)}, Eval: {len(eval_set)}")

    # Check eval category coverage
    eval_cats = set()
    for ex in eval_set:
        yaml_section = ex['messages'][2]['content'].split('</think>')[-1]
        cats = re.findall(r'^\s+- (.+)$', yaml_section, re.MULTILINE)
        eval_cats.update(cats)
    print(f"Categories in eval: {len(eval_cats)}/{len(cat_counter)}")

    # Write output
    with open(OUTPUT_FILE, 'w') as f:
        for ex in train:
            f.write(json.dumps(ex) + '\n')

    with open(OUTPUT_EVAL_FILE, 'w') as f:
        for ex in eval_set:
            f.write(json.dumps(ex) + '\n')

    print(f"\nWritten: {OUTPUT_FILE} ({len(train)} examples)")
    print(f"Written: {OUTPUT_EVAL_FILE} ({len(eval_set)} examples)")

    # Show a sample
    print("\n=== SAMPLE OUTPUT ===")
    sample = train[1]
    print(f"SYSTEM ({len(sample['messages'][0]['content'])} chars): ...{sample['messages'][0]['content'][-100:]}")
    print(f"\nUSER: {sample['messages'][1]['content']}")
    print(f"\nASSISTANT:\n{sample['messages'][2]['content']}")


if __name__ == "__main__":
    main()

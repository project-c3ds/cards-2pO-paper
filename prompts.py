import pandas as pd
import os

# ---------------------------------------------------------------------------
# Codebooks (built from taxonomy.csv)
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
df_codebook = pd.read_csv(f'{current_dir}/data/taxonomy.csv')

# Full codebook: verbose XML-tagged descriptions (for big models)
codebook = "\n".join(df_codebook.xml_prompt_label.unique())

# Slim codebook: short labels (for fine-tuned small models)
_slim_codebook_list = []
for _, row in df_codebook.drop_duplicates('category_number').iterrows():
    code = row['category_number']
    label = row['short_label'] if pd.notna(row['short_label']) and str(row['short_label']).strip() else row['prompt_label']
    _slim_codebook_list.append(f'<{code}> {label}')
slim_codebook = "\n".join(_slim_codebook_list)

# ---------------------------------------------------------------------------
# System instructions (pre-formatted with codebook, ready to use)
# ---------------------------------------------------------------------------

_instruction_template = """You are an expert in climate communication. Your task is to classify the given text into categories based on the provided codebook. This is a multi-label classification task.

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

system_instruction = _instruction_template.format(codebook=codebook)
slim_system_instruction = _instruction_template.format(codebook=slim_codebook)

# ---------------------------------------------------------------------------
# Few-shot instruction (appended to system instruction when enabled)
# ---------------------------------------------------------------------------

fewshot_instruction = """
You will be given a few examples annotated by human experts. Examples given to you are selected based on the similarity, Have a look at the examples and try to understand the pattern in the annotations. Please note, the human annotations are not perfect, use your best judgment to understand the pattern and apply it to the new examples. You need to explain if any of these examples have helped you in understanding the pattern.

### Examples:
{fewshot}
"""

# ---------------------------------------------------------------------------
# Triggers (final user messages)
# ---------------------------------------------------------------------------

# CoT trigger: used as the final user message at inference
cot_trigger = """Let's work this out in a step by step way to be sure we have the right answer."""

# RECoT trigger: used when generating training data with a teacher model.
# Tells the teacher to produce reasoning that arrives at the known true labels.
recot_trigger = """The true labels above are the correct classification. Generate expert-level reasoning that arrives at these exact labels.

Rules:
- Write as a confident expert who has never seen the true labels. Do not mention or hint at them.
- In SCAN, include the true labels among the plausible candidates. In VERIFY, eliminate the wrong ones confidently.
- Be decisive. Single pass, no second-guessing, no repetition.
- Final classification must exactly match the true labels."""

# ---------------------------------------------------------------------------
# Synthetic data generation prompt
# ---------------------------------------------------------------------------

synthetic_prompt = """You are an expert in climate communication and disinformation research.

### FULL TAXONOMY:
{codebook}

### TARGET CATEGORY:
Code: {target_code}
Description: {target_description}

### EXISTING EXAMPLES FOR THIS CATEGORY:
{examples}

### TASK:
Generate {n} new realistic texts that express the target category claim. Each text must:

1. Include the target category ({target_code}) as one of its claims
2. Naturally combine with other claims from the taxonomy (2-5 total labels per text)
3. Sound like real content — social media posts, opinion pieces, congressional testimony, news comments, blog excerpts, forum posts
4. Vary in length (1-6 sentences), tone (formal, casual, sarcastic, angry, concerned), and source type
5. Endorse/promote the claims, not just describe or report them
6. Be distinct from the existing examples — different framing, context, arguments

Assign labels at the most granular level only. Do not include parent codes if a subclaim applies.

Return as a JSON array of objects: {{"text": "...", "true_claims": ["code1", "code2", ...]}}"""

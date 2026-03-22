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
   - Otherwise, scan every superclaim group (1_ through 7_) for relevance before evaluating any in detail.
   - For each relevant group, evaluate subclaims at the most granular level.
   - Always return the most granular matching level.

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
3. SCAN: One line per superclaim group (1_ through 7_). Relevant or not relevant.
4. EVALUATE: One entry per relevant claim using this format:
   - Claim N → [code]: [what it means + endorsed or not]. [why not the closest alternative].
5. VERIFY: Codes consistent together? Granularity correct? Overlaps missed?
</think>
```yaml
categories:
  - <category_code>
```

STRICT RULES:
- All reasoning must be inside <think> tags. Nothing after </think> except the YAML block.
- Match depth to complexity. Simple texts get short steps.
- Never deliberate between codes in EVALUATE. Pick one, justify it, move on.
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
- Be decisive. Single pass, no second-guessing, no repetition.
- Final classification must exactly match the true labels."""

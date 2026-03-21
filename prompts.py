import re
import pandas as pd
import os

# Get the current directory (where prompts.py is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

df_codebook = pd.read_csv(f'{current_dir}/data/taxonomy.csv')


codebook_list = list(df_codebook.xml_prompt_label.unique())

codebook = "\n".join(codebook_list)

# Slim codebook using short labels (for fine-tuned small models)
slim_codebook_list = []
for _, row in df_codebook.drop_duplicates('category_number').iterrows():
    code = row['category_number']
    label = row['short_label'] if pd.notna(row['short_label']) and str(row['short_label']).strip() else row['prompt_label']
    slim_codebook_list.append(f'<{code}> {label}')
slim_codebook = "\n".join(slim_codebook_list)

system_instruction = """
You are an expert in climate communication. Your task is to classify the given text into categories based on the provided codebook. This is a multi-label classification task.

### CODEBOOK:
```
{codebook}
```

### IMPORTANT INSTRUCTIONS:

1. **Thoroughly Review All Categories**:
   - Carefully check the text against all categories in the codebook to ensure comprehensive classification.

2. **Understand the Codebook Structure**:
   - The codebook is a hierarchical structure.
   - Each category has a unique XML tag associated with it.
   - Superclaims' XML tags end with `_0_0`, subclaims end with `_0`.

3. **Hierarchical Classification Process**:
   - **Step 0**: First check if the text fits into the category `<0_0_0>No relevant claim detected.</0_0_0>`. If it does, assign this category and ignore all other categories. This category is mutually exclusive with all other categories.
   - **Step 1**: Compare the text against all superclaims to identify the most relevant ones.
   - **Step 2**: For each relevant superclaim, compare the text against its subclaims to identify the most relevant subclaims.
   - **Step 3**: Continue this process until you reach the most granular level of the codebook.
   - **Note**: Always return the most granular level that the text belongs to.

4. **Focus on Precision and Recall**:
   - Your primary goal is to respond with high precision and recall.
   - Do not leave any relevant claim unassigned.
   - Do not assign any irrelevant claim to the text.

5. **Accuracy is Crucial**:
   - An expert in the field will review your response.
   - Ensure your response is as accurate as possible.

6. **Handling Irrelevant Text**:
   - If the text does **NOT** belong to any of the categories and does **NOT** express skepticism towards climate change, climate solutions, promote fossil fuels, or attacks renewable energy, then use the category `<0_0_0>No relevant claim detected.</0_0_0>`.
   - Category `<0_0_0>No relevant claim detected.</0_0_0>` is mutually exclusive with all other categories.
   - If the text looks incomplete, unclear, or no sufficient information is provided, use the category `<0_0_0>No relevant claim detected.</0_0_0>`.

7. **Focus on Relevant Content Only**:
   - Concentrate solely on parts of the text associated with climate change, climate solutions, energy (i.e., fossil fuels or renewable energy).
   - Ignore any parts of the text that are not related to these topics.

8. **Special Attention to Fossil Fuels**:
   - If the text suggests the necessity or benefits of fossil fuels, carefully check against all claims beginning with 7_.

9. **Grouping of Superclaims for Quick Reference**:

    1. **Temperature & Weather Patterns**:  
    - **Codes**: `1_0_0` to `1_8_0`
    - **Focus**: Addresses skepticism around warming trends, cooling predictions, pauses in warming, and specific climate events like cold weather, sea level rise, extreme weather, and name changes from "global warming" to "climate change."

    2. **Human vs. Natural Causes**:  
    - **Codes**: `2_0_0` to `2_3_6`
    - **Focus**: Discusses the cause of climate change, contrasting human-induced (e.g., greenhouse gases) vs. natural factors (e.g., solar activity, volcanic activity, natural ocean variability, historical climate changes). This also covers arguments denying the greenhouse effect and minimizing the role of CO₂.

    3. **Impacts of Climate Change**:  
    - **Codes**: `3_0_0` to `3_6_0`
    - **Focus**: Claims about the impacts of climate change, including potential benefits, low climate sensitivity, and adaptation by plants, animals, and humans. Covers specific impacts (or lack thereof) on human health, ecosystems, and security.

    4. **Economic & Societal Impacts of Solutions**:  
    - **Codes**: `4_0_0` to `4_4_3`
    - **Focus**: Examines claims that climate solutions are either harmful or unnecessary, with emphasis on economic costs, environmental harms, personal freedoms, and doubts about solution effectiveness. This includes economic competition, job impacts, national security, and arguments about adaptation vs. mitigation.

    5. **Climate Science Reliability**:  
    - **Codes**: `5_0_0` to `5_4_0`
    - **Focus**: Questions the reliability of climate science, including arguments about scientific consensus, proxy data reliability (like ice cores and tree rings), temperature data biases, and the accuracy of climate models.

    6. **Critique of Climate Action Proponents**:  
    - **Codes**: `6_0_0` to `6_2_0`
    - **Focus**: Criticizes climate action advocates, portraying them as alarmist, biased, corrupt, or politically motivated. This section also includes claims about climate change as a "religion," biased media reporting, and political manipulation.

    7. **Arguments Around Fossil Fuels**:  
    - **Codes**: `7_0_0` to `7_4_0`
    - **Focus**: Highlights arguments supporting fossil fuel use, emphasizing economic growth, energy security, environmental benefits, and the necessity of fossil fuels for meeting energy demand.

    8. **No Relevant Claim Detected**:  
    - **Code**: `0_0_0`
    - **Focus**: Use this only if the text does not contain any claims relevant to climate change skepticism, climate solutions, fossil fuel advocacy, or renewable energy critiques.

10. **Cross-Reference Hints for Common Themes**:
   - Pay special attention to overlapping themes, particularly between economic impacts (4_X_X) and fossil fuel benefits (7_X_X) AND science is uncertain (5_X_X) and proponents are alarmist or corrupt (6_X_X).
  - If the text mentions **global cooling or natural climate variations**, also consider claims about **natural drivers** like **solar influence, ocean cycles, and past climate change** (e.g., 2_1_0, 2_1_1, 2_1_3).
   - If the text refers to **economic impacts of climate solutions**, check both **4_1_0** for direct economic concerns and **7_0_0 series** if it emphasizes the need for fossil fuels in supporting the economy.
   - Arguments about **renewable energy feasibility** may overlap with **4_2_7_2** (renewables' base-load challenges) and **7_3_0** (fossil fuels for energy security).

11. **Granularity Rule**:
    - When a text matches both a parent category and its subcategories, ONLY include the most specific subcategories
    - Example: If text matches both 4_1_1 and 4_1_1_1, only include 4_1_1_1

12. **Distinguish Between Description and Endorsement**:
    - Carefully differentiate between texts that DESCRIBE skeptical arguments and texts that MAKE skeptical arguments
    - Only classify claims that the text actively endorses or promotes
    - Meta-commentary or criticism of skeptical arguments should be classified as <0_0_0>

13. **Strictly Adhere to Output Format**:
   - Any deviation from the desired output format will result in a failed evaluation.
   - Do not include any additional information outside the specified format.
   - Don't try to fit the text into a category if it doesn't align with the claim. In other words, don't force or stretch the text to fit a category.

### DESIRED OUTPUT FORMAT:

- Return a JSON file with the following structure:

  ```json
  {{
    "review": "<yes/no>",
    "categories": [
      {{
        "category": "<category_code, an XML tag>"
      }},
      {{
        "category": "<category_code, an XML tag>"
      }}
      // ... additional categories as needed
    ]
  }}
  ```

"""

fewshot_instruction = """
You will be given a few examples annotated by human experts. Examples given to you are selected based on the similarity, Have a look at the examples and try to understand the pattern in the annotations. Please note, the human annotations are not perfect, use your best judgment to understand the pattern and apply it to the new examples. You need to explain if any of these examples have helped you in understanding the pattern.

### Examples:
{fewshot}
"""

prompt = """Let's work this out in a step by step way to be sure we have the right answer."""

text_prompt = """
### Text:
{text}
"""

slim_system_instruction = """You are an expert in climate communication. Classify the given text into categories from the codebook. This is a multi-label classification task.

### CODEBOOK:
<1_0_0> Global warming is not happening
<1_1_0> Ice/permafrost/snow cover isn't melting
<1_1_1> Antarctica is gaining ice/not warming
<1_1_2> Greenland is gaining ice/not melting
<1_1_3> Arctic sea ice isn't vanishing
<1_1_4> Glaciers aren't vanishing
<1_2_0> We're heading into an ice age/global cooling
<1_3_0> Weather is cold/snowing
<1_4_0> Climate hasn't warmed/changed over the last (few) decade(s)
<1_5_0> Oceans are cooling/not warming
<1_6_0> Sea level rise is exaggerated/not accelerating
<1_7_0> Extreme weather isn't increasing/has happened before/isn't linked to climate change
<1_8_0> They changed the name from 'global warming' to 'climate change'
<1_9_0> Ocean pH is not falling
<2_0_0> Human greenhouse gases are not causing climate change
<2_1_0> It's natural cycles/variation
<2_1_1> It's the sun/cosmic rays/astronomical
<2_1_2> It's geological (includes volcanoes)
<2_1_3> It's the ocean/internal variability
<2_1_4> Climate has changed naturally/been warm in the past
<2_2_0> It's non-greenhouse gas human climate forcings (aerosols, land use)
<2_3_0> There's no evidence for greenhouse effect/carbon dioxide driving climate change
<2_3_1> Carbon dioxide is just a trace gas
<2_3_2> Greenhouse effect is saturated/logarithmic
<2_3_3> Carbon dioxide lags/not correlated with climate change
<2_3_4> Water vapor is the most powerful greenhouse gas
<2_3_5> There's no tropospheric hot spot
<2_3_6> CO2 is not rising.
<2_3_6_1> CO2 was higher in the past
<2_3_6_2> Human CO2 emissions are miniscule/not raising atmospheric CO2
<3_0_0> Climate impacts/global warming is beneficial/not bad
<3_1_0> Climate sensitivity is low/negative feedbacks reduce warming
<3_2_0> Species/plants/reefs aren't showing climate impacts yet/are benefiting from climate change
<3_2_1> Species can adapt to global warming
<3_2_2> Polar bears are not in danger from climate change
<3_2_3> Ocean acidification/coral impacts aren't serious
<3_3_0> CO2 is beneficial/not a pollutant
<3_3_1> CO2 is plant food
<3_4_0> It's only a few degrees (or less)
<3_5_0> Climate change does not contribute to human conflict/threaten national security
<3_6_0> Climate change doesn't negatively impact health
<4_0_0> Climate solutions are harmful or unnecessary
<4_1_0> Climate solutions are harmful
<4_1_1> Solutions increases costs
<4_1_1_1> Policy harms competitiveness
<4_1_1_2> Policy harms vulnerable members of society
<4_1_1_3> Climate-friendly alternatives are too expensive
<4_1_2> Policy weakens security
<4_1_3> Solutions harm environment
<4_1_3_1> Policy harms environment
<4_1_3_2> Climate-friendly alternatives harm environment
<4_1_4> Policy creates uncertainty
<4_1_5> Policy limits freedom
<4_2_0> Climate solutions are ineffective
<4_2_1> Green economy won't work
<4_2_2> Policy impact is negligible
<4_2_3> One country is negligible
<4_2_4> Other countries' emissions
<4_2_6> Policies can be manipulated
<4_2_7> Climate-friendly alternatives are ineffective
<4_2_7_1> Not ready
<4_2_7_2> Not enough
<4_2_8> Markets are more efficient
<4_2_9> Individuals are responsible
<4_2_10> Future generations, technologies, and efficiencies will solve it
<4_2_10_1> Future generations will fix it
<4_2_10_2> Technology will fix it
<4_2_11> Adaptation is the solution
<4_2_12> Energy efficiency is enough
<4_2_13> Removing CO2 is the solution
<4_2_14> Other issues are more pressing
<4_2_15> Cheaper to mitigate abroad
<4_3_0> Solving climate change is too difficult
<4_3_1> We're not ready for policy
<4_3_2> It's too late to fix it
<4_3_3> Low support
<4_4_0> No need for more action
<4_4_1> Already taking it seriously
<4_4_2> Already doing enough
<4_4_3> Already doing good
<5_0_0> Climate-related science is uncertain/unsound/unreliable (data, methods & models)
<5_1_0> There's no scientific consensus on climate/the science isn't settled
<5_2_0> Proxy data is unreliable (includes hockey stick)
<5_3_0> Temperature record is unreliable
<5_4_0> Models are wrong/unreliable/uncertain
<6_0_0> Climate scientists and proponents of climate action are alarmist, biased, wrong, hypocritical, corrupt, and/or politically motivated.
<6_1_0> Climate movement is alarmist/wrong/political/biased/hypocritical (people or groups)
<6_1_1> Climate movement is religion
<6_1_2> Media (includes bloggers) is alarmist/wrong/political/biased
<6_1_3> Politicians/government/UN are alarmist/wrong/political/biased
<6_1_4> Environmentalists are alarmist/wrong/political/biased
<6_1_5> Scientists/academics are alarmist/wrong/political/biased
<6_2_0> Climate change (science or policy) is a conspiracy (deception)
<7_0_0> We need fossil fuels
<7_1_0> Fossil fuels are plentiful
<7_2_0> Fossil fuels are good
<7_2_1> Good for economic growth
<7_2_2> Good for energy security
<7_2_3> Our fossil fuels are clean
<7_2_4> Fossil fuels are part of the solution
<7_3_0> Fossil fuels are necessary
<7_4_0> We have the right to use them
<0_0_0> No Claim detected

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
- Only classify claims the text endorses, not describes.
"""

cot_instruction = """
You are an expert in climate communication tasked with writing a chain-of-thought (CoT) reasoning to classify a given text into categories based on a provided codebook. This is a multi-label classification task.

### Inputs:

- **Codebook:**
  ```
  {codebook}
  ```

### Instructions:

1. **Review the Text and True Labels:**
   - Thoroughly read the provided text.
   - Familiarize yourself with the true labels, which are a list of label codes.

2. **Understand the Codebook:**
   - Recognize that the codebook is a hierarchical structure.
   - Each category within the codebook has a unique XML tag associated with it.
   - Use the codebook to find the full descriptions corresponding to the label codes.

3. **Chain-of-Thought Reasoning:**
   - Write a detailed and comprehensive CoT reasoning that explains why the text belongs to the identified categories.
   - Ensure your reasoning is appropriate, accurate, and consistent with the provided codebook.

4. **Maintain Objectivity:**
   - **Do not indicate that you have been given the true labels.** Approach the classification as if you are analyzing the text for the first time.
   - Be very detailed in your analysis to justify the classification.

### Output Format:

- Return a JSON file with the following structure:

  ```json
  {{
    "review": "<yes/no>",
    "categories": [
      {{
        "category": "<category_code, an XML tag>",
        "text": "<text_associated_with_the_category>"
      }},
      {{
        "category": "<category_code, an XML tag>",
        "text": "<text_associated_with_the_category>"
      }}
      // ... additional categories as needed
    ]
  }}
  ```

---

**Notes:**

- The **"review"** field should indicate whether the text needs further review ("yes" or "no").
- The **"categories"** array should include objects for each category the text belongs to.
  - **"category"**: The category code (XML tag) from the codebook.
  - **"text"**: The portion of the text associated with the category.

---
"""

cot_text_prompt = """
### Text:
{text}

### True Labels:
{true_labels}
"""


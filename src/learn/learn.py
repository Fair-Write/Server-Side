# import torch

# shape = (2,3,)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

from datasets import load_dataset

# Load your dataset (assuming the JSON file is named 'gender_fair_dataset.json')
dataset = load_dataset("json", data_files="datasets.json")

# Inspect the first entry to ensure it's loaded correctly
print(dataset['train'][0])


prompt = """
**Task:**
Revise each sentence to use gender-fair language, correct grammar, and spelling errors while preserving the original meaning. For each revision, provide a clear and specific rationale explaining why the change was necessary, and classify the type of rationale. If no revision is needed, include the original sentence in the output without any changes.

**Input Sentences:**
{input_text}

**Instructions:**
1. Return the results as structured JSON with the following format:
    ```json
    {
      "suggested_revisions": [
        {
          "original": "original sentence",
          "revision": "revised sentence or original if unchanged",
          "rationale": "Specific reason for the revision, focusing on clarity and gender neutrality. If no change is needed, state 'No changes needed.'",
          "type": "type of rationale (gender neutral, grammar, spelling, none)"
        },
        ...
      ]
    }
    ```
2. Each revision should:
   - Replace gendered terms with gender-neutral language.
   - Correct any grammatical or spelling errors.
   - Maintain the original meaning of the sentence.
3. For each rationale, provide a specific explanation of the change, focusing on how it enhances gender fairness or corrects grammatical issues.
4. Classify the type of rationale under **type** as one of the following:
   - "gender neutral" for gender-related changes
   - "grammar" for grammar corrections
   - "spelling" for spelling corrections
   - "none" for no revision.
5. Output only the JSON, formatted as specified above.
"""
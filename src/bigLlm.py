from transformers import pipeline, AutoTokenizer
import torch

model = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device=device,
    temperature=0.3,     # Lower temperature for more controlled output
    top_k=40,            # Restrict to fewer top token choices
    top_p=0.85,          # Use nucleus sampling for balanced responses
    
)

input_text = "The policwoman halped the woman. He said that shee should folow him."
prompt = f"""
***Input Text:*** "{input_text}"

***Main Task:***
Revise the above text to use gender-fair language.

***Instructions:***
- Maintain the original meaning.
- Correct any grammatical or spelling errors.
- Return only the revised text without additional notes.
"""

sequences = pipe(
    prompt,
    max_new_tokens=100,  
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")
    
print("\n\n")
    
for seq in sequences:
    result_text = seq['generated_text'].strip().split("\n")[0]
    print(f"Result: {result_text}")
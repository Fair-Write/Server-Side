
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from collections import OrderedDict
import torch
import language_tool_python
from fastapi.middleware.cors import CORSMiddleware
import ollama
import spacy
import csv
import re
import csv

# Load models and tools
nlp = spacy.load('en_core_web_sm')
tool = language_tool_python.LanguageTool('en-US')

# Initialize FastAPI
app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Load the model and tokenizer
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id
# if model.config.pad_token_id is None:
#     model.config.pad_token_id = model.config.eos_token_id
# # model.eval()  # Set model to evaluation mode

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# model.save_pretrained("./model")
# tokenizer.save_pretrained("./model")

@app.get("/")
async def read_root():    
    return {"message": "Welcome to our API!"}


class TextRequest(BaseModel):
    prompt: str
    max_length: int = 50      # Default maximum length

@app.post("/generate")
def generate_text(request: TextRequest):
    # Tokenize the input
    inputs = tokenizer(request.prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=request.max_length)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": generated_text}

@app.post("/generate1")
def generate_text(request: TextRequest):
    sequences = pipe(
        request.prompt,
        max_new_tokens=50,
        do_sample=True,
        top_k=10,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    
    generated_text = seq['generated_text']

    return {"generated_text": generated_text}

@app.post("/sentiment")
def generate_text(request: TextRequest):
    prompt = f"""Classify the text into positiveness, just return percentage of positiveness. 
    Text: {request.prompt}
    Sentiment: 
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=request.max_length)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Generated text:", generated_text)

    return {"generated_text": generated_text}


@app.post("/sentiment1")
def generate_text(request: TextRequest):
    prompt = f"""Classify the text into positiveness, just return percentage of positiveness. 
    Text: {request.prompt}
    Sentiment: 
    """

    sequences = pipe(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    top_k=10,
    )
    
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    
    generated_text = seq['generated_text']

    return {"generated_text": generated_text}


@app.post("/tokenize")
def generate_text(request: TextRequest):
    encoding = tokenizer(request.prompt)
    inputs = tokenizer(request.prompt, return_tensors="pt")

    return {"token": encoding, "input": inputs }

@app.post("/gender_fair")
def generate_text(request: TextRequest):
    gender_fair_output = generate_gender_fair_text(request.prompt)
    print("Gender-fair revision:", gender_fair_output)
    return { "output" : gender_fair_output}

def generate_gender_fair_text(input_text):
    prompt = f"""
    **Task:**
    Revise the following text to use gender-fair language and correct any grammatical errors.
    
    **Input Text:** '{input_text}'
    
    **Instructions:**
    1. Maintain the original meaning.
    2. Provide only the revised text, no explanations.
    3. Enclose the revised text in single quotes.
    """
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=120,  # Adjust as necessary
        num_return_sequences=1,
        do_sample=True,  # Use sampling for diversity
        top_k=50,  # Controls diversity
        top_p=0.95,  # Controls diversity
        temperature=1.0  # Adjust for creativity
    )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


class Prompt(BaseModel):
    prompt: str

@app.post("/ollama")
async def generate(prompt: Prompt):
    command = ["ollama", "run", "llama3.2", prompt.prompt]
    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode == 0:
        output = result.stdout.strip()
        print("Raw Output:", output)

        if output.startswith("failed to get console mode for stdout: The handle is invalid.\nfailed to get console mode for stderr: The handle is invalid.\n"):
            return {"output": output.removeprefix("failed to get console mode for stdout: The handle is invalid.\nfailed to get console mode for stderr: The handle is invalid.\n")}

        return {"response": output}
    else:
        return {"error": result.stderr.strip() or "Unknown error occurred."}

@app.post("/ollama-gfl")
async def generate(prompt: Prompt):
    prompt = f"""
    ***Input Text:*** "{prompt.prompt}"

    ***Main Task:***
    Revise the above text to use gender-fair language.

    ***Instructions:***
    - Maintain the original meaning.
    - Correct any grammatical or spelling errors.
    - Return only the revised text without additional notes.
    """

    try:
        output = ollama.generate(model='llama3.2', prompt=prompt)
        return {"output": output["response"]}

    except Exception as e:
        return {"error": str(e)}      


@app.post("/grammar")
async def generate(prompt: Prompt):
    text = prompt.prompt
    # Analyze the text
    matches = tool.check(text)

    # Create a list for corrections
    corrections = []

    # Process each match to extract the necessary details
    for match in matches:
        corrections.append({
            "character_offset": match.offset,
            "character_endset": match.offset + match.errorLength,
            "original_text": text[match.offset:match.offset + match.errorLength],
            "message": match.message,
            "category": match.category,
            "rule_id": match.ruleId,
            "replacements": match.replacements[:5]  # Only get the top 5 replacements
        })

    # Generate revised text with corrections applied
    revised_text = language_tool_python.utils.correct(text, matches)

    # Construct the final result
    result = {
        "original_text": text,
        "revised_text": revised_text,
        "corrections": corrections
    }

    return result

@app.post("/gfl")
async def generate(prompt: Prompt):
    # Load gender-fair language terms from a CSV file
    def load_gfl_terms(csv_file):
        gfl_dict = {}
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                gender_term, fair_term = row
                gfl_dict[gender_term.lower()] = fair_term
        return gfl_dict
    
    def gender_fair_revision(text, gfl_terms):
        # Parse the text with SpaCy
        doc = nlp(text)
        revised_text = text
        corrections = []

        # Words that indicate gender specificity
        gender_indicators = {"lady", "male", "female", "boy", "girl"}
        replaced_indices = set()  # Keep track of indices already replaced

        # Iterate over the tokens and replace gender terms
        offset_correction = 0
        inside_quotes = False
        for i, token in enumerate(doc):
            # Check for quote tokens to toggle inside_quotes flag
            if token.text in ['"', "'"]:
                inside_quotes = not inside_quotes

            if not inside_quotes:
                # Check for gender indicators followed by nouns
                if token.text.lower() in gender_indicators:
                    if i + 1 < len(doc):  # Ensure there's a next word
                        next_token = doc[i + 1]
                        # Proceed only if the next word is a noun
                        if next_token.pos_ in {"NOUN", "PROPN"}:  # NOUN = common noun, PROPN = proper noun
                            original_text = f"{token.text} {next_token.text}"
                            replacement = gfl_terms.get(next_token.text.lower(), next_token.text)

                            # Maintain capitalization for the second word
                            if next_token.text.istitle():
                                replacement = replacement.capitalize()
                            elif next_token.text.isupper():
                                replacement = replacement.upper()

                            # Calculate offsets
                            start = token.idx
                            end = next_token.idx + len(next_token.text)

                            # Adjust the revised text
                            revised_text = (
                                revised_text[:start + offset_correction]
                                + replacement
                                + revised_text[end + offset_correction:]
                            )

                            # Add correction details
                            corrections.append({
                                "original_text": original_text,
                                "replacements": replacement,
                                "character_offset": start,
                                "character_endset": end,
                                "original_character_endset": end
                            })

                            # Adjust offset correction for the next replacement
                            offset_correction += len(replacement) - len(original_text)

                            # Mark indices as replaced
                            replaced_indices.add(next_token.idx)
                            continue  # Skip further checks for this token

                # Single-word replacements
                if token.idx not in replaced_indices and token.text.lower() in gfl_terms:
                    original_text = token.text
                    replacement = gfl_terms[original_text.lower()]

                    # Maintain capitalization
                    if original_text.istitle():
                        replacement = replacement.capitalize()
                    elif original_text.isupper():
                        replacement = replacement.upper()

                    # Calculate offsets
                    start = token.idx
                    end = start + len(original_text)

                    # Adjust the revised text
                    revised_text = (
                        revised_text[:start + offset_correction]
                        + replacement
                        + revised_text[start + offset_correction + len(original_text):]
                    )

                    # Add correction details
                    corrections.append({
                        "original_text": original_text,
                        "replacements": replacement,
                        "character_offset": start,
                        "character_endset": end,
                        "original_character_endset": end
                    })

                    # Adjust offset correction for the next replacement
                    offset_correction += len(replacement) - len(original_text)

                    # Mark index as replaced
                    replaced_indices.add(token.idx)

        # Create the output JSON
        return {
            "original_text": text,
            "revised_text": revised_text,
            "corrections": corrections
        }

    text = prompt.prompt
    # text = """In times of emergency, firemen are the brave ones who risk their lives to save others, while policemen work tirelessly to enforce law and order; on our streets. These men are just naturally inclined towards such roles, given there physical strength and courage. Firemen and policemen undergo rigorous training that prepares them for the challenging situations they face everyday, showing that some jobs simply fit men better. Women might work as policewomen or lady firefighters, but its often a tough fit for them as compared to their male colleagues. In the business world, a successful businessman is admire for his ability to negotiate and lead a team effectively. Many companies prefer male chairmen since they are known for their decisiveness and strategic thinking. Even at lower levels, salesmen is often seen as more persuasive than their female counterparts, as people tend to trust men in these roles. Women on the other hand, usually pursue careers as secretaries or assistants, providing the vital support to their male bosses whom handle the main responsibilities.
    # """
    gfl_terms = load_gfl_terms("gendered_terms.csv")
    return gender_fair_revision(text, gfl_terms) 
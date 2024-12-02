
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
    matches = tool.check(text)

    # Split text into words and their character ranges
    words, word_ranges = split_text_with_punctuation(text)

    corrections = []
    revised_text = list(text)  # Convert text to a list for mutable character edits
    offset_adjustment = 0  # Tracks changes in text length due to replacements

    for match in matches:
        original_text = text[match.offset: match.offset + match.errorLength]
        adjusted_offset = match.offset + offset_adjustment
        word_index = get_word_index_from_offset(word_ranges, adjusted_offset)

        # Apply the first suggested replacement (if available) to the revised text
        if match.replacements:
            replacement = match.replacements[0]
            # Replace text while adjusting offsets
            revised_text[adjusted_offset: adjusted_offset + match.errorLength] = list(replacement)
            offset_adjustment += len(replacement) - match.errorLength

        corrections.append({
            "word_index": word_index,  # Word-based index
            "character_offset": match.offset,  # Original character offset in the text
            "character_endset": match.offset + match.errorLength,  # Original character end offset in the text
            "original_text": original_text,
            "message": match.message,
            "replacements": match.replacements or [""],  # Handle empty suggestions
        })

    revised_text_str = ''.join(revised_text)
    if revised_text_str == text:
        return {}
    return {
        "original_text": text,
        "revised_text": revised_text_str,  # Join the list to form the revised sentence
        "corrections": corrections
    }

def split_text_with_punctuation(text):
    words_with_punctuation = re.findall(r'\S+|\s|[.,!?;(){}\[\]":]', text)
    
    word_ranges = []
    current_offset = 0
    for word in words_with_punctuation:
        start_offset = current_offset
        end_offset = start_offset + len(word)
        word_ranges.append((start_offset, end_offset))
        current_offset = end_offset

    return words_with_punctuation, word_ranges

def get_word_index_from_offset(word_ranges, offset):
    for i, (start, end) in enumerate(word_ranges):
        if start <= offset < end:
            return i
    return -1  # Return -1 if no match is found, which shouldn't happen

@app.post("/gfl")
async def generate(prompt: Prompt):
    def load_gendered_terms(csv_filename): 

        gendered_terms = {}
        try:
            with open(csv_filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        gendered_terms[row[0].lower()] = row[1]
        except Exception as e:
            raise ValueError(f"Error loading gendered terms from {csv_filename}: {e}")
        return gendered_terms

    def adjust_capitalization(original, replacement):
        """Preserve capitalization of the original word/phrase in the replacement."""
        if original.isupper():
            return replacement.upper()
        elif original.istitle():
            return replacement.capitalize()
        return replacement

    def prioritize_terms(terms):
        """Sort gendered terms by phrase length in descending order."""
        return OrderedDict(
            sorted(terms.items(), key=lambda item: len(item[0].split()))
        )

    def is_within_quotes(text, start, end):
        """Check if the match is inside double quotes."""
        # Find the closest quote before and after the match
        before = text[:start]
        after = text[end:]

        # Check if there is an odd number of quotes before the match and after the match
        return before.count('"') % 2 == 1 and after.count('"') % 2 == 1

    def make_gender_fair(text, terms_csv='gendered_terms.csv'):
        def load_gendered_terms(csv_filename): 
            gendered_terms = {}
            try:
                with open(csv_filename, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if len(row) >= 2:
                            gendered_terms[row[0].lower()] = row[1]
            except Exception as e:
                raise ValueError(f"Error loading gendered terms from {csv_filename}: {e}")
            return gendered_terms

    def adjust_capitalization(original, replacement):
        """Preserve capitalization of the original word/phrase in the replacement."""
        if original.isupper():
            return replacement.upper()
        elif original.istitle():
            return replacement.capitalize()
        return replacement

    def prioritize_terms(terms):
        """Sort gendered terms by phrase length in descending order."""
        return OrderedDict(
            sorted(terms.items(), key=lambda item: len(item[0].split()))
        )

    def is_within_quotes(text, start, end):
        """Check if the match is inside double quotes."""
        # Find the closest quote before and after the match
        before = text[:start]
        after = text[end:]

        # Check if there is an odd number of quotes before the match and after the match
        return before.count('"') % 2 == 1 and after.count('"') % 2 == 1

    def make_gender_fair(text, terms_csv='./src/gendered_terms.csv'):
        """Replace gendered terms in the text with gender-neutral terms."""
        # Load and prioritize gendered terms
        gendered_terms = prioritize_terms(load_gendered_terms(terms_csv))

        # Process text using spaCy for tokenization
        doc = nlp(text)

        revised_text = text
        corrections = []

        # Replace exact matches (including hyphenated terms) using regex
        for phrase, replacement in gendered_terms.items():
            # Regex to match full word/phrase boundaries, case-insensitive
            pattern = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)

            # Find all matches and replace them one by one with correct capitalization
            for match in pattern.finditer(revised_text):
                original = match.group(0)  # The matched text

                # Check if the match is inside double quotes
                if is_within_quotes(revised_text, match.start(), match.end()):
                    continue  # Skip if the match is inside quotes

                adjusted_replacement = adjust_capitalization(original, replacement)

                # Replace text
                revised_text = (
                    revised_text[:match.start()] + 
                    adjusted_replacement + 
                    revised_text[match.end():]
                )

                # Identify the word index by checking overlap with tokens
                match_start = match.start()
                match_end = match.end()

                word_index = None
                for i, token in enumerate(doc):
                    if token.idx <= match_start < token.idx + len(token):
                        word_index = i
                        break

                # Track correction details with offsets
                corrections.append({
                    "word_index": word_index,
                    "original_text": original,
                    "replacements": adjusted_replacement,
                    "character_offset": match.start(),
                    "character_endset": match.end()
                })

        return {
            "original_text": text,
            "revised_text": revised_text,
            "corrections": corrections
        }
    
    text = prompt.prompt
    return make_gender_fair(text) 



import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
import language_tool_python
from fastapi.middleware.cors import CORSMiddleware
import ollama



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
    
    # Check for grammar issues using LanguageTool
    matches = tool.check(text)
    
    # Split the text into words and calculate character ranges for each word
    words = text.split()
    word_ranges = []
    current_offset = 0

    for word in words:
        start_offset = current_offset
        end_offset = start_offset + len(word)
        word_ranges.append((start_offset, end_offset))
        current_offset = end_offset + 1  # Account for space after each word

    # Prepare a list to store corrections
    corrections = []

    for match in matches:
        # Extract information about the grammar issue
        original_text = text[match.offset: match.offset + match.errorLength]

        # Find the word index using word_ranges
        word_index = next((i for i, (start, end) in enumerate(word_ranges) if start <= match.offset < end),-1)

        # Construct the correction object
        correction = {
            "word_index": word_index,  # Word-based index
            "character_offset": match.offset,  # Character-based offset
            "original_text": original_text,
            "message": match.message,
            "replacements": match.replacements or ["No suggestions"],  # Handle empty replacements
        }

        corrections.append(correction)

    # Return the corrections in a structured format
    return {
        "original_text": text,
        "corrections": corrections
    }

# @app.get("/other")
# async def read_root():
#     # fetch data to localhost port 1235
#     response = requests.get("http://localhost:1235/v1/models")

#     print(response.status_code)
#     return response.json()


# class BodyPrompt(BaseModel):
#     prompt: str

# @app.post("/send_prompt")
# async def create_prompt(request: BodyPrompt = Body(...)):
#     response = requests.post("http://localhost:1235/v1/completions", json={"prompt": request.prompt})
#     if response.status_code != 200:
#         raise HTTPException(status_code=response.status_code, detail="Failed to get completion")
#     return response.json()
#     # print(request.prompt)
#     # return {"prompt": request.prompt}

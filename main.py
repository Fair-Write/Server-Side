from fastapi import FastAPI
import language_tool_python
from fastapi.middleware.cors import CORSMiddleware
import csv

from main_gfl import load_gfl
from schemas import GrammarBody, GFLBody

# Load models and tools
tool = language_tool_python.LanguageTool('en-US')

# Load your GenderFairLanguage instance
gfl = load_gfl()

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():    
    return {"message": "Welcome to our API!"}


@app.post("/grammar")
async def generate(body: GrammarBody):
    text = body.prompt
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
async def process_text_endpoint(body: GFLBody):
    input_text = body.prompt
    preferred_pronouns = body.pronoun_map
    # Process the text using the GenderFairLanguage instance
    result = gfl.process_text(input_text, preferred_pronouns)
    return result
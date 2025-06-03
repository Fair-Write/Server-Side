from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import language_tool_python
from main_gfl import load_gfl
from csvCrud import GenderTermManager
from schemas import GrammarBody, GFLBody, GenderTermCreate, GenderTermUpdate, GenderTermBulkCreate
from pathlib import Path
from count import read_counter, increment_counter , decrement_counter 

# Load models and tools
tool = language_tool_python.LanguageTool('en-US')

# Initialize the TermManager with the path to your CSV file
genderTermManager = GenderTermManager("gendered_terms.csv")

# Initialize FastAPI
app = FastAPI()

# Initialize the counter file
COUNTER_FILE = "count.csv"

# Add CORS middleware to allow cross-origin requests
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
async def grammar_revision(body: GrammarBody):
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
async def gender_fair_revision(body: GFLBody):
    input_text = body.prompt
    preferred_pronouns = body.pronoun_map
    # Process the text using the GenderFairLanguage instance
    # Load your GenderFairLanguage instance
    gfl = load_gfl()
    result = gfl.process_text(input_text, preferred_pronouns)
    
    # Count request increment
    increment_counter(COUNTER_FILE)
    count = read_counter(COUNTER_FILE)
    
    return {
        "count": count,
        **result,
    }
    
@app.get("/terms")
def list_all_terms():
        return genderTermManager.get_all_term()

@app.get("/terms/{term}")
def get_term(term: str):
        result = genderTermManager.find_term(term)
        if not result:
            raise HTTPException(status_code=404, detail="Term not found")
        return result

@app.put("/terms/{term}")
def update_term(term: str, update_data: GenderTermUpdate):
    try:
        genderTermManager.update(term, update_data.options)
        return {"message": f"Term '{term}' updated"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/terms/{term}", status_code=status.HTTP_204_NO_CONTENT)
def delete_term(term: str):
    try:
        genderTermManager.delete(term)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
@app.post("/terms", status_code=status.HTTP_201_CREATED)
def create_term(term_data: GenderTermCreate):
    try:
        genderTermManager.create(term_data.term, term_data.options)
        return {"message": f"Term '{term_data.term}' created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/count")
def get_count_request():
    current_value = read_counter(COUNTER_FILE)
    return {"count": current_value}

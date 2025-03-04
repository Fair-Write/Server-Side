from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
import language_tool_python
from fastapi.middleware.cors import CORSMiddleware
import spacy
import csv
import re
import csv

# Load models and tools
nlp = spacy.load('en_core_web_sm')
tool = language_tool_python.LanguageTool('en-US')

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


class Prompt(BaseModel):
    prompt: str

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

class PromptGFL(BaseModel):
    prompt: str 
    pronoun_map: dict[str, Literal['gender_fair', 'female', 'male']] = {
        "Alex": "gender_fair",
        "John": "male",
        "Jane": "female"
    }

@app.post("/gfl")
async def generate(prompt: PromptGFL):


    # Load gendered terms from CSV
    def load_gendered_terms(csv_filename):
        """Load gendered terms from a CSV file into a dictionary."""
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

    # Adjust capitalization of replacements
    def adjust_capitalization(original, replacement):
        """Preserve capitalization of the original word/phrase in the replacement."""
        if original.isupper():
            return replacement.upper()
        elif original.istitle():
            return replacement.capitalize()
        return replacement

    # Prioritize terms by length
    def prioritize_terms(terms):
        """Sort gendered terms by phrase length in descending order."""
        return OrderedDict(
            sorted(terms.items(), key=lambda item: len(item[0].split()), reverse=True)
        )

    def is_within_quotes(text, start, end):
        """
        Check if the match is inside quotes (single or double).
        Handles nested quotes and escaped quotes.
        """
        # Slice the text into before and after the match
        before = text[:start]
        after = text[end:]

        # Count unescaped double quotes before the match
        double_quotes_before = before.count('"') - before.count('\\"')
        # Count unescaped single quotes before the match
        single_quotes_before = before.count("'") - before.count("\\'")

        # Count unescaped double quotes after the match
        double_quotes_after = after.count('"') - after.count('\\"')
        # Count unescaped single quotes after the match
        single_quotes_after = after.count("'") - after.count("\\'")

        # Check if the match is inside double quotes
        inside_double_quotes = (double_quotes_before % 2 == 1) and (double_quotes_after % 2 == 1)
        # Check if the match is inside single quotes
        inside_single_quotes = (single_quotes_before % 2 == 1) and (single_quotes_after % 2 == 1)

        # Return True if the match is inside any type of quotes
        return inside_double_quotes or inside_single_quotes

    # Replace gender terms followed by nouns
    def replace_gender_adj_noun_pairs(doc, gendered_terms, original_text):
        """Replace gender terms followed by nouns based on prioritized terms."""
        revised_tokens = []
        corrections = []

        skip_next = False
        gender_adjectives = {"male", "female", "lady", "gentlemen", "boy", "girl", "man", "woman"}

        for i, token in enumerate(doc):
            if skip_next:
                skip_next = False
                continue

            term = token.text.lower()
            next_token = doc[i + 1] if i + 1 < len(doc) else None

            # Check if the current term is a gender adjective
            if term in gender_adjectives and next_token and next_token.pos_ == "NOUN":
                compound_phrase = f"{term} {next_token.text.lower()}"
                replacement = gendered_terms.get(compound_phrase, None)

                # Ensure the match is NOT inside double or single quotes
                if is_within_quotes(original_text, token.idx, next_token.idx + len(next_token.text)):
                    revised_tokens.append(token.text)
                    continue
                
                if replacement:
                    # Replace the compound phrase explicitly if it's in the dictionary
                    corrections.append({
                        "word_index": i,
                        "original_text": compound_phrase,
                        "replacements": replacement,
                        "character_offset": token.idx,
                        "character_endset": next_token.idx + len(next_token.text)
                    })
                    revised_tokens.append(replacement)
                    skip_next = True  # Skip the next token as it was part of the compound phrase
                    continue
                else:
                    # If no explicit replacement, replace with just the noun
                    corrections.append({
                        "word_index": i,
                        "original_text": compound_phrase,
                        "replacements": next_token.text,
                        "character_offset": token.idx,
                        "character_endset": next_token.idx + len(next_token.text)
                    })
                    revised_tokens.append(next_token.text)
                    skip_next = True
                    continue

            revised_tokens.append(token.text)

        revised_text = " ".join(revised_tokens)
        return revised_text, corrections

    def replace_pronouns(text, name_pronoun_map, pronoun_options=None):
        """
        Replace pronouns in a text based on a mapping of names to pronouns.
        Skip pronouns that are inside quotes.
        """
        if pronoun_options is None:
            pronoun_options = {
                "male": {
                    "nsubj": "he",  # Subject pronoun
                    "dobj": "him",  # Object pronoun
                    "poss": "his",  # Possessive adjective
                    "poss_pronoun": "his",  # Possessive pronoun
                    "reflexive": "himself"  # Reflexive pronoun
                },
                "female": {
                    "nsubj": "she",
                    "dobj": "her",
                    "poss": "her",
                    "poss_pronoun": "hers",
                    "reflexive": "herself"
                },
                "gender_fair": {
                    "nsubj": "they",
                    "dobj": "them",
                    "poss": "their",
                    "poss_pronoun": "theirs",
                    "reflexive": "themselves"
                }
            }

        doc = nlp(text)

        # Reverse map for quick lookup of pronouns
        pronoun_reverse_map = {}
        for category, pronouns in pronoun_options.items():
            for role, value in pronouns.items():
                pronoun_reverse_map[value] = (category, role)

        # Process the text and replace pronouns
        name_to_category = {name.lower(): category for name, category in name_pronoun_map.items()}

        def get_pronoun_replacement(token, category):
            if token.text.lower() in pronoun_reverse_map:
                _, role = pronoun_reverse_map[token.text.lower()]
                return pronoun_options[category][role]
            return token.text

        replaced_text = []
        replaced_words = []

        for token in doc:
            # Check if token is a pronoun based on its tag and find replacement if applicable
            if token.pos_ == "PRON":
                # Check if the pronoun is inside quotes
                if is_within_quotes(text, token.idx, token.idx + len(token.text)):
                    # Skip replacement if the pronoun is inside quotes
                    replaced_text.append(token.text_with_ws)
                    continue

                relevant_entity = None
                for ent in doc.ents:
                    if ent.text.lower() in name_to_category and ent.end <= token.i:
                        relevant_entity = ent

                if relevant_entity:
                    category = name_to_category[relevant_entity.text.lower()]
                    replacement = get_pronoun_replacement(token, category)
                    # Only replace if the pronoun is different from the preferred pronoun
                    if replacement.lower() != token.text.lower():
                        replaced_text.append(replacement + token.whitespace_)
                        replaced_words.append({
                            "original_word": token.text,
                            "replaced_word": replacement,
                            "word_index": token.i,
                            "char_offset": token.idx,
                            "char_end_offset": token.idx + len(token.text)
                        })
                    else:
                        replaced_text.append(token.text_with_ws)  # Keep original pronoun if it matches
                else:
                    replaced_text.append(token.text_with_ws)  # Keep original pronoun if no match found
            else:
                replaced_text.append(token.text_with_ws)

        # Ensure proper spacing by joining tokens directly as processed
        return {
            "modified_text": "".join(replaced_text),
            "replaced_words": replaced_words
        }

    def find_word_index(doc, start_char, end_char):
        """
        Find the word index in the spaCy doc based on character offsets.
        """
        for i, token in enumerate(doc):
            if token.idx <= start_char < token.idx + len(token.text):
                return i
        return None

    def main_gfl(text, terms_csv='gendered_terms.csv', name_pronoun_map=None):
        """Replace gendered terms and pronouns in the text."""
        # Load and prioritize gendered terms
        gendered_terms = prioritize_terms(load_gendered_terms(terms_csv))

        # Process text using spaCy for tokenization
        doc = nlp(text)

        # Step 1: Handle gender term followed by noun
        revised_text, noun_corrections = replace_gender_adj_noun_pairs(doc, gendered_terms, text)

        # Step 2: Apply main gendered term replacement logic
        corrections = []

        # Precompile regex patterns for all gendered terms
        patterns = {phrase: re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE) for phrase in gendered_terms}

        # Replace exact matches (including hyphenated terms) using regex
        for phrase, pattern in patterns.items():
            replacement = gendered_terms[phrase]

            # Find all matches and replace them one by one with correct capitalization
            matches = list(pattern.finditer(text))  # Use original text for finding matches
            for match in reversed(matches):  # Process in reverse to avoid offset issues
                original = match.group(0)  # The matched text

                # Check if the match is inside quotes
                if is_within_quotes(text, match.start(), match.end()):
                    continue  # Skip if the match is inside quotes

                adjusted_replacement = adjust_capitalization(original, replacement)

                # Replace text
                revised_text = (
                    revised_text[:match.start()] +
                    adjusted_replacement +
                    revised_text[match.end():]
                )

                # Find the word index in the original doc
                word_index = find_word_index(doc, match.start(), match.end())

                # Track correction details with original offsets
                corrections.append({
                    "word_index": word_index,
                    "original_text": original,
                    "replacements": adjusted_replacement,
                    "character_offset": match.start(),
                    "character_endset": match.end()
                })

        # Step 3: Replace pronouns if a name-pronoun map is provided
        pronoun_corrections = []
        if name_pronoun_map:
            pronoun_result = replace_pronouns(revised_text, name_pronoun_map)
            revised_text = pronoun_result["modified_text"]
            pronoun_corrections = pronoun_result["replaced_words"]

        # Combine all corrections
        all_corrections = noun_corrections + corrections + pronoun_corrections

        return {
            "original_text": text,
            "revised_text": revised_text,
            "corrections": all_corrections
        }

    text = prompt.prompt
    name_pronoun_map = prompt.pronoun_map
    output = main_gfl(text,terms_csv='gendered_terms.csv' ,name_pronoun_map=name_pronoun_map)

    return output
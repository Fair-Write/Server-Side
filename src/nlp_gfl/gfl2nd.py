import csv
import re
from collections import OrderedDict
import json
import spacy

nlp = spacy.load('en_core_web_sm')

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
        sorted(terms.items(), key=lambda item: len(item[0].split()), reverse=True)
    )

def is_within_quotes(text, start, end):
    """Check if the match is inside double quotes."""
    before = text[:start]
    after = text[end:]
    return before.count('"') % 2 == 1 and after.count('"') % 2 == 1

def replace_gender_adj_noun_pairs(doc, gendered_terms):
    """Replace gender terms followed by nouns based on prioritized terms."""
    revised_tokens = []
    corrections = []

    skip_next = False
    for i, token in enumerate(doc):
        if skip_next:
            skip_next = False
            continue

        term = token.text.lower()
        next_token = doc[i + 1] if i + 1 < len(doc) else None

        # Check if the current term is a gender adjective
        if term in {"male", "female", "lady", "gentlemen"} and next_token and next_token.pos_ == "NOUN":
            compound_phrase = f"{term} {next_token.text.lower()}"
            replacement = gendered_terms.get(compound_phrase, None)

            if replacement:
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
                # If the compound phrase isn't explicitly in the terms, use just the noun
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


def main_gfl(text, terms_csv='gendered_terms.csv'):
    """Replace gendered terms in the text with gender-neutral terms."""
    # Load and prioritize gendered terms
    gendered_terms = prioritize_terms(load_gendered_terms(terms_csv))

    # Process text using spaCy for tokenization
    doc = nlp(text)

    # Step 1: Handle gender term followed by noun
    revised_text, noun_corrections = replace_gender_adj_noun_pairs(doc, gendered_terms)

    # Step 2: Apply main gendered term replacement logic
    corrections = []

    # Replace exact matches (including hyphenated terms) using regex
    for phrase, replacement in gendered_terms.items():
        # Regex to match full word/phrase boundaries, case-insensitive
        pattern = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)

        # Find all matches and replace them one by one with correct capitalization
        matches = list(pattern.finditer(revised_text))  # Collect matches first to avoid conflicts
        for match in reversed(matches):  # Process in reverse to avoid offset issues
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

            # Map match offsets to token indices in the original doc
            match_start = match.start()
            match_end = match.end()

            word_index = None
            for i, token in enumerate(doc):
                token_start = token.idx
                token_end = token.idx + len(token)

                # Check if the match fully or partially overlaps this token
                if token_start <= match_start < token_end or token_start < match_end <= token_end:
                    word_index = i
                    break

            # Track correction details with offsets
            corrections.append({
                "word_index": word_index,
                "original_text": original,
                "replacements": adjusted_replacement,
                "character_offset": match.start(),
                "character_endset": match.start() + len(adjusted_replacement)
            })

    # Combine noun corrections and gender term corrections
    all_corrections = noun_corrections + corrections

    return {
        "original_text": text,
        "revised_text": revised_text,
        "corrections": all_corrections
    }

text = """She said the lady policeman was happy and she was a lady chairman."""
output = main_gfl(text)
print(json.dumps(output, indent=4))
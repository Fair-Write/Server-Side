import csv
import re
from collections import OrderedDict
import json
import spacy

nlp = spacy.load('en_core_web_sm')


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

text = "The chairman of the company is a honest"
print(make_gender_fair(text))
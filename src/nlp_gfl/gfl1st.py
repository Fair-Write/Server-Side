import spacy
import csv
import json

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

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

# Gender-fair revision function
def gender_fair_revision(text, gfl_terms, preferred_pronouns):
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
            # Handle names with preferred pronouns
            if token.text in preferred_pronouns:
                pronoun = preferred_pronouns[token.text]
                corrections.append({
                    "original_text": token.text,
                    "replacements": pronoun,
                    "character_offset": token.idx,
                    "character_endset": token.idx + len(token.text),
                    "note": f"Assigned preferred pronoun: {pronoun}"
                })
                continue

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

# Example usage
text = "A lady firefighter with chairman and a girl police officer. John said he prefers they/them pronouns."

gfl_terms = load_gfl_terms("gendered_terms.csv")
preferred_pronouns = {
    "John": "they/them",
    "Mary": "she/her"
}

output = gender_fair_revision(text, gfl_terms, preferred_pronouns)
print(json.dumps(output, indent=4))

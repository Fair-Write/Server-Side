import spacy
import csv

nlp = spacy.load('en_core_web_sm')

def load_gendered_terms(csv_filename):
    gendered_terms = {}
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            gendered_terms[row[0]] = row[1]
    return gendered_terms

def adjust_capitalization(original, replacement):
    """Preserve capitalization of the original word in the replacement."""
    if original.isupper():
        return replacement.upper()
    elif original.istitle():
        return replacement.capitalize()
    else:
        return replacement

def make_gender_fair(text):
    gendered_terms = load_gendered_terms('./src/gendered_terms.csv')

    # Process the corrected text with spaCy
    doc = nlp(text)

    # Replace gendered terms
    revised_tokens = [
        adjust_capitalization(token.text, gendered_terms.get(token.text.lower(), token.text))
        for token in doc
    ]

    return " ".join(revised_tokens)

# Example usage
text = "The Policewoman helped the woman. He said that she should folow him."
revised_text = make_gender_fair(text)
print("Original Text:")
print(text)
print("\nRevised Text:")
print(revised_text)

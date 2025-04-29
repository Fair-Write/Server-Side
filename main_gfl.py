import spacy
import csv
import re
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

# Constants
DEFAULT_PRONOUNS = {
    "male": {"subject": "he", "object": "him", "possessive": "his", "reflexive": "himself"},
    "female": {"subject": "she", "object": "her", "possessive": "her", "reflexive": "herself"},
    "gender_fair": {"subject": "they", "object": "them", "possessive": "their", "reflexive": "themselves"}
}

GENDER_ADJECTIVES = {"male", "female", "lady", "gentlemen", "boy", "girl", "man", "woman"}
GENDER_PAIRS = {
    ("girl", "boy"): "children",
    ("boy", "girl"): "children",
    ("woman", "man"): "people",
    ("man", "woman"): "people",
    ("women", "men"): "people",
    ("men", "women"): "people"
}
nlp = spacy.load('en_core_web_sm')

class GenderFairLanguage:
    def __init__(self, terms_csv: str = 'gendered_terms.csv'):
        self.gendered_terms = self._load_and_prioritize_terms(terms_csv)
        
    def _load_and_prioritize_terms(self, csv_filename: str) -> OrderedDict:
        """Load and prioritize gendered terms from CSV."""
        gendered_terms = {}
        try:
            with open(csv_filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        gendered_terms[row[0].lower()] = row[1]
        except Exception as e:
            raise ValueError(f"Error loading gendered terms: {e}")
            
        return OrderedDict(
            sorted(gendered_terms.items(), key=lambda item: len(item[0].split()), reverse=True)
        )

    @staticmethod
    def _adjust_capitalization(original: str, replacement: str) -> str:
        """Preserve capitalization patterns in replacements."""
        if original.isupper():
            return replacement.upper()
        elif original.istitle():
            return replacement.capitalize()
        return replacement.lower()

    def _is_within_quotes(self, doc, start_idx: int, end_idx: int) -> bool:
        """Check if text is within quotes using spaCy tokens."""
        quotes_before = [t for t in doc[:start_idx] if t.text in ('"', "'")]
        quotes_after = [t for t in doc[end_idx:] if t.text in ('"', "'")]
        return (len(quotes_before) % 2 == 1) and (len(quotes_after) % 2 == 1)

    def _process_text_replacements(self, text: str, name_pronoun_map: Dict) -> Tuple[str, List[Dict]]:
        """Main text processing with accurate offset tracking."""
        doc = nlp(text)
        corrections = []
        
        # First collect all potential corrections without modifying text
        corrections.extend(self._find_gender_pairs(text, doc))
        corrections.extend(self._find_redundant_pairs(text, doc))
        corrections.extend(self._find_adjective_noun_pairs(text, doc))
        corrections.extend(self._find_individual_terms(text, doc))
        corrections.extend(self._find_pronoun_replacements(text, doc, name_pronoun_map))
        
        # Remove overlapping corrections, keeping the longest matches
        corrections = self._filter_overlapping_corrections(corrections)
        
        # Apply corrections in reverse order
        revised_text = text
        for correction in sorted(corrections, key=lambda x: -x['character_offset']):
            revised_text = (
                revised_text[:correction['character_offset']] +
                correction['replacements'] +
                revised_text[correction['character_endset']:]
            )
        
        return revised_text, corrections

    def _find_gender_pairs(self, text: str, doc) -> List[Dict]:
        """Find specific gender pairs to replace with more inclusive terms."""
        corrections = []
        
        for (term1, term2), replacement in GENDER_PAIRS.items():
            # Pattern for "term1 and term2" or "term1 or term2"
            pattern = rf'\b({term1})\s+(and|or)\s+({term2})\b'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in reversed(matches):
                if self._is_within_quotes(doc, match.start(), match.end()):
                    continue
                
                # Determine capitalization from first word
                first_word = match.group(1)
                adjusted_replacement = replacement.capitalize() if first_word[0].isupper() else replacement
                
                corrections.append({
                    "word_index": next((i for i, t in enumerate(doc) 
                                      if t.idx <= match.start() < t.idx + len(t.text)), None),
                    "original_text": match.group(0),
                    "replacements": adjusted_replacement,
                    "character_offset": match.start(),
                    "character_endset": match.end()
                })
        
        return corrections

    def _find_redundant_pairs(self, text: str, doc) -> List[Dict]:
        """Find redundant gendered pairs with proper capitalization."""
        redundant_patterns = [
            (r'\b([Gg]irls?|[Bb]oys?)\s+(and|or)\s+([Gg]irls?|[Bb]oys?)\b', 'children'),
            (r'\b([Mm]en|[Ww]omen)\s+(and|or)\s+([Mm]en|[Ww]omen)\b', 'people'),
            (r'\b([Hh]e|[Ss]he)\s+(and|or)\s+([Hh]e|[Ss]he)\b', 'they')
        ]
        
        corrections = []
        
        for pattern, replacement in redundant_patterns:
            matches = list(re.finditer(pattern, text))
            for match in reversed(matches):
                if self._is_within_quotes(doc, match.start(), match.end()):
                    continue
                
                # Determine capitalization from first word
                first_word = match.group(1)
                adjusted_replacement = replacement.capitalize() if first_word[0].isupper() else replacement
                
                corrections.append({
                    "word_index": next((i for i, t in enumerate(doc) 
                                      if t.idx <= match.start() < t.idx + len(t.text)), None),
                    "original_text": match.group(0),
                    "replacements": adjusted_replacement,
                    "character_offset": match.start(),
                    "character_endset": match.end()
                })
        
        return corrections

    def _find_adjective_noun_pairs(self, text: str, doc) -> List[Dict]:
        """Find gender adjective-noun pairs."""
        corrections = []
        
        for i in range(len(doc) - 1):
            token = doc[i]
            next_token = doc[i + 1]
            
            if (token.text.lower() in GENDER_ADJECTIVES and 
                next_token.pos_ == "NOUN" and
                not self._is_within_quotes(doc, token.idx, next_token.idx + len(next_token.text))):
                
                noun_replacement = self.gendered_terms.get(next_token.text.lower(), next_token.text)
                replacement = self._adjust_capitalization(next_token.text, noun_replacement)
                
                corrections.append({
                    "word_index": i,
                    "original_text": f"{token.text} {next_token.text}",
                    "replacements": replacement,
                    "character_offset": token.idx,
                    "character_endset": next_token.idx + len(next_token.text)
                })
        
        return corrections

    def _find_individual_terms(self, text: str, doc) -> List[Dict]:
        """Find individual gendered terms."""
        corrections = []
        
        for term, replacement in self.gendered_terms.items():
            if ' ' in term:  # Skip multi-word terms (handled elsewhere)
                continue
                
            matches = list(re.finditer(rf'\b{re.escape(term)}\b', text, re.IGNORECASE))
            for match in reversed(matches):
                if self._is_within_quotes(doc, match.start(), match.end()):
                    continue
                
                adjusted = self._adjust_capitalization(match.group(0), replacement)
                corrections.append({
                    "word_index": next((i for i, t in enumerate(doc) 
                                      if t.idx <= match.start() < t.idx + len(t.text)), None),
                    "original_text": match.group(0),
                    "replacements": adjusted,
                    "character_offset": match.start(),
                    "character_endset": match.end()
                })
        
        return corrections

    def _find_pronoun_replacements(self, text: str, doc, name_pronoun_map: Dict) -> List[Dict]:
        """Find pronoun replacements with fallback to gender-fair language."""
        corrections = []
        name_to_category = {name.lower(): category for name, category in (name_pronoun_map or {}).items()}
        
        # First collect all person entities
        person_entities = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
        
        # Pronouns that are already gender-neutral and shouldn't be replaced
        NEUTRAL_PRONOUNS = {"i", "me", "my", "mine", "myself",
                            "we", "us", "our", "ours", "ourselves",
                            "you", "your", "yours", "yourself", "yourselves",
                            "they", "them", "their", "theirs", "themselves"}
        
        for token in doc:
            if (token.tag_ in ["PRP", "PRP$"] and 
                token.text.lower() not in NEUTRAL_PRONOUNS and
                not self._is_within_quotes(doc, token.idx, token.idx + len(token.text))):
                
                # Find most recent named entity that could be referent
                referent = None
                for ent in reversed(person_entities[:token.i]):
                    if ent.end <= token.i:  # Entity appears before pronoun
                        referent = ent.text
                        break
                    
                # Determine pronoun role
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    role = "subject"
                elif token.dep_ in ["dobj", "iobj", "pobj"]:
                    role = "object"
                elif token.dep_ == "poss" or token.tag_ == "PRP$":
                    role = "possessive"
                elif token.dep_ == "reflexive":
                    role = "reflexive"
                else:
                    continue
                
                # Use specified pronoun if referent is in name_pronoun_map,
                # otherwise default to gender-fair
                if referent and referent.lower() in name_to_category:
                    category = name_to_category[referent.lower()]
                    new_pronoun = DEFAULT_PRONOUNS[category][role]
                else:
                    new_pronoun = DEFAULT_PRONOUNS["gender_fair"][role]
                
                # Preserve original capitalization
                if token.text.istitle():
                    new_pronoun = new_pronoun.capitalize()
                elif token.text.isupper():
                    new_pronoun = new_pronoun.upper()
                
                if new_pronoun.lower() != token.text.lower():
                    corrections.append({
                        "word_index": token.i,
                        "original_text": token.text,
                        "replacements": new_pronoun,
                        "character_offset": token.idx,
                        "character_endset": token.idx + len(token.text)
                    })
        
        return corrections

    def _filter_overlapping_corrections(self, corrections: List[Dict]) -> List[Dict]:
        """Filter out overlapping corrections, keeping the longest matches."""
        if not corrections:
            return []
            
        # Sort by start position then by length (longest first)
        sorted_corrections = sorted(corrections, key=lambda x: (x['character_offset'], -x['character_endset']))
        
        filtered = []
        last_end = -1
        
        for correction in sorted_corrections:
            if correction['character_offset'] >= last_end:
                filtered.append(correction)
                last_end = correction['character_endset']
        
        return filtered

    def process_text(self, text: str, name_pronoun_map: Optional[Dict] = None) -> Dict:
        """Main processing method that returns original and revised text with corrections."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        revised_text, corrections = self._process_text_replacements(text, name_pronoun_map or {})
        
        return {
            "original_text": text,
            "revised_text": revised_text,
            "corrections": sorted(
                [c for c in corrections if c['character_offset'] >= 0],
                key=lambda x: x['character_offset']
            )
        }

# Export
def load_gfl(csv_path: str = 'gendered_terms.csv') -> GenderFairLanguage:
    return GenderFairLanguage(csv_path)
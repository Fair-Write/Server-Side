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
    ("son", "daughter"): "children",
    ("daughter", "son"): "children",
    ("woman", "man"): "people",
    ("man", "woman"): "people",
    ("women", "men"): "people",
    ("men", "women"): "people",
    ("he", "she"): "they",
    ("she", "he"): "they",
    ("his", "her"): "their",
    ("her", "his"): "their",
    ("him", "her"): "them",
    ("her", "him"): "them",
    ("himself", "herself"): "themselves",
    ("herself", "himself"): "themselves",
    ("husband", "wife"): "spouse",
    ("wife", "husband"): "spouse",
    ("boyfriend", "girlfriend"): "partner",
    ("girlfriend", "boyfriend"): "partner",
    ("brother", "sister"): "sibling",
    ("sister", "brother"): "sibling",
    ("father", "mother"): "parent",
    ("mother", "father"): "parent",
    ("uncle", "aunt"): "relative",
    ("aunt", "uncle"): "relative",
    ("nephew", "niece"): "relative",
    ("niece", "nephew"): "relative",
    ("ladies", "gentlemen"): "everyone",
    ("gentlemen", "ladies"): "everyone"
}

nlp = spacy.load('en_core_web_sm')

class GenderFairLanguage:
    def __init__(self, terms_csv: str = 'gendered_terms.csv'):
        self.gendered_terms = self._load_and_prioritize_terms(terms_csv)
        
    def _load_and_prioritize_terms(self, csv_filename: str) -> OrderedDict:
        """Load and prioritize gendered terms from CSV with multiple replacements."""
        gendered_terms = OrderedDict()
        try:
            with open(csv_filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        term = row[0].strip().lower()
                        replacements = [repl.strip() for repl in row[1:] if repl.strip()]
                        if term and replacements:
                            gendered_terms[term] = replacements
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
        """Check if text is within double quotes in the original text."""
        text = doc.text
        before = len(re.findall(r'(?<!\\)"', text[:start_idx]))
        after = len(re.findall(r'(?<!\\)"', text[end_idx:]))
        return (before % 2 == 1) and (after % 2 == 1)

    def _process_text_replacements(self, text: str, name_pronoun_map: Dict) -> Tuple[str, List[Dict]]:
        """Main text processing with accurate offset tracking."""
        doc = nlp(text)
        corrections = []
        
        corrections.extend(self._find_gender_pairs(text, doc))
        corrections.extend(self._find_redundant_pairs(text, doc))
        corrections.extend(self._find_adjective_noun_pairs(text, doc))
        corrections.extend(self._find_individual_terms(text, doc))
        corrections.extend(self._find_pronoun_replacements(text, doc, name_pronoun_map))
        
        corrections = self._filter_overlapping_corrections(corrections)
        
        revised_text = text
        for correction in sorted(corrections, key=lambda x: -x['character_offset']):
            replacement_str = correction['replacements'][0]
            revised_text = (
                revised_text[:correction['character_offset']] +
                replacement_str +
                revised_text[correction['character_endset']:]
            )
        
        return revised_text, corrections

    def _find_gender_pairs(self, text: str, doc) -> List[Dict]:
        """Find specific gender pairs to replace with more inclusive terms."""
        corrections = []

        for (term1, term2), replacement in GENDER_PAIRS.items():
            term1_variants = [term1, term1 + 's', term1 + 'es']
            term2_variants = [term2, term2 + 's', term2 + 'es']

            for t1 in term1_variants:
                for t2 in term2_variants:
                    pattern = rf'\b({t1})\s+(and|or)\s+({t2})\b'
                    matches = list(re.finditer(pattern, text, re.IGNORECASE))

                    for match in reversed(matches):
                        if self._is_within_quotes(doc, match.start(), match.end()):
                            continue
                        
                        if match.group(2).lower() == "or":
                            actual_replacement = replacement[:-1] if replacement.endswith('s') else replacement[:-2] if replacement.endswith('es') else replacement
                        else:
                            actual_replacement = replacement

                        first_word = match.group(1)
                        adjusted_replacement = self._adjust_capitalization(first_word, actual_replacement)

                        corrections.append({
                            "word_index": next((i for i, t in enumerate(doc) if t.idx <= match.start() < t.idx + len(t.text)), None),
                            "original_text": match.group(0),
                            "replacements": [adjusted_replacement],
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
                
                first_word = match.group(1)
                adjusted = replacement.capitalize() if first_word[0].isupper() else replacement
                
                corrections.append({
                    "word_index": next((i for i, t in enumerate(doc) if t.idx <= match.start() < t.idx + len(t.text)), None),
                    "original_text": match.group(0),
                    "replacements": [adjusted],
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
                
                noun_replacements = self.gendered_terms.get(next_token.text.lower(), [next_token.text])
                adjusted_replacements = [self._adjust_capitalization(next_token.text, repl) for repl in noun_replacements]
                
                corrections.append({
                    "word_index": i,
                    "original_text": f"{token.text} {next_token.text}",
                    "replacements": adjusted_replacements,
                    "character_offset": token.idx,
                    "character_endset": next_token.idx + len(next_token.text)
                })
        
        return corrections

    def _find_individual_terms(self, text: str, doc) -> List[Dict]:
        """Find individual gendered terms."""
        corrections = []
        
        for term, replacements_list in self.gendered_terms.items():
            if ' ' in term:
                continue
            
            for match in reversed(list(re.finditer(rf'\b{re.escape(term)}\b', text, re.IGNORECASE))):
                if self._is_within_quotes(doc, match.start(), match.end()):
                    continue
                
                original_text = match.group(0)
                adjusted_replacements = [self._adjust_capitalization(original_text, repl) for repl in replacements_list]
                
                corrections.append({
                    "word_index": next((i for i, t in enumerate(doc) if t.idx <= match.start() < t.idx + len(t.text)), None),
                    "original_text": original_text,
                    "replacements": adjusted_replacements,
                    "character_offset": match.start(),
                    "character_endset": match.end()
                })
        
        return corrections

    def _find_pronoun_replacements(self, text: str, doc, name_pronoun_map: Dict) -> List[Dict]:
        """Find pronoun replacements with fallback to gender-fair language."""
        corrections = []
        name_to_category = {name.lower(): category for name, category in (name_pronoun_map or {}).items()}

        person_entities = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
        NEUTRAL_PRONOUNS = {"i", "me", "my", "mine", "myself",
                            "we", "us", "our", "ours", "ourselves",
                            "you", "your", "yours", "yourself", "yourselves",
                            "they", "them", "their", "theirs", "themselves", "it", "its", "itself"}

        for token in doc:
            if (token.tag_ in ["PRP", "PRP$"] and 
                token.text.lower() not in NEUTRAL_PRONOUNS and
                not self._is_within_quotes(doc, token.idx, token.idx + len(token.text))):

                referent = None
                for ent in reversed(person_entities[:token.i]):
                    if ent.end <= token.i:
                        referent = ent.text
                        break
                
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
                
                if referent and referent.lower() in name_to_category:
                    category = name_to_category[referent.lower()]
                    new_pronoun = DEFAULT_PRONOUNS[category][role]
                else:
                    new_pronoun = DEFAULT_PRONOUNS["gender_fair"][role]

                if token.text.istitle():
                    new_pronoun = new_pronoun.capitalize()
                elif token.text.isupper():
                    new_pronoun = new_pronoun.upper()

                if new_pronoun.lower() != token.text.lower():
                    corrections.append({
                        "word_index": token.i,
                        "original_text": token.text,
                        "replacements": [new_pronoun],
                        "character_offset": token.idx,
                        "character_endset": token.idx + len(token.text)
                    })

        return corrections

    def _filter_overlapping_corrections(self, corrections: List[Dict]) -> List[Dict]:
        """Filter out overlapping corrections, keeping the longest matches."""
        if not corrections:
            return []
            
        sorted_corrections = sorted(corrections, key=lambda x: (x['character_offset'], -x['character_endset']))
        filtered = []
        last_end = -1
        
        for c in sorted_corrections:
            if c['character_offset'] >= last_end:
                filtered.append(c)
                last_end = c['character_endset']
        
        return filtered

    def process_text(self, text: str, name_pronoun_map: Optional[Dict] = None) -> Dict:
        """Main processing method that returns original and revised text with corrections."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        revised_text, corrections = self._process_text_replacements(text, name_pronoun_map or {})
        
        return {
            "original_text": text,
            "revised_text": revised_text,
            "corrections": sorted([
                c for c in corrections if c['character_offset'] >= 0
            ], key=lambda x: x['character_offset'])
        }

def load_gfl(csv_path: str = 'gendered_terms.csv') -> GenderFairLanguage:
    return GenderFairLanguage(csv_path)
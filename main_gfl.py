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
    ("girl", "boy"): ["children", "kids", "students", "youth", "young people"],
    ("son", "daughter"): ["children", "kids", "offspring"],
    ("woman", "man"): ["people", "individuals", "persons"],
    ("women", "men"): ["people", "individuals", "persons"],
    ("he", "she"): ["they"],
    ("his", "her"): ["their"],
    ("him", "her"): ["them"],
    ("himself", "herself"): ["themselves"],
    ("husband", "wife"): ["spouse", "partner"],
    ("boyfriend", "girlfriend"): ["partner"],
    ("brother", "sister"): ["sibling"],
    ("father", "mother"): ["parent"],
    ("uncle", "aunt"): ["relative"],
    ("nephew", "niece"): ["relative"],
    ("ladies", "gentlemen"): ["everyone", "all"]
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
        """Main text processing with enhanced title detection."""
        doc = nlp(text)
        
        # Enhanced title detection in token stream
        title_mapping = {'mr': 'male', 'ms': 'female', 'mx': 'gender_fair'}
        i = 0
        while i < len(doc):
            token = doc[i]
            # Look for titles with optional period and proper noun following
            base_title = token.text.lower().rstrip('.')
            if base_title in title_mapping:
                # Find all consecutive proper nouns after title
                name_parts = [token.text]
                j = i + 1
                while j < len(doc) and (doc[j].pos_ == 'PROPN' or doc[j].text in ['-']):
                    name_parts.append(doc[j].text)
                    j += 1
                
                if len(name_parts) > 1:  # Found at least title + 1 name part
                    # Map full name and name without title
                    full_name = ' '.join(name_parts).lower()
                    category = title_mapping[base_title]
                    name_pronoun_map[full_name] = category
                    
                    # Add name without title (multiple name parts)
                    name_without_title = ' '.join(name_parts[1:]).lower()
                    name_pronoun_map[name_without_title] = category
                    
                    # Skip processed tokens
                    i = j - 1
            i += 1

        # Original entity-based processing (supplements token-based detection)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                parts = ent.text.strip().split()
                if parts:
                    base_title = parts[0].lower().rstrip('.')
                    if base_title in title_mapping:
                        full_name = ent.text.strip().lower()
                        name_pronoun_map[full_name] = title_mapping[base_title]

        # Rest of processing remains the same
        corrections = []
        corrections.extend(self._find_gender_pairs(text, doc))
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

        for (term1, term2), replacements in GENDER_PAIRS.items():
            for t1, t2 in [(term1, term2), (term2, term1)]:
                term1_variants = [t1, t1 + 's', t1 + 'es']
                term2_variants = [t2, t2 + 's', t2 + 'es']

                for v1 in term1_variants:
                    for v2 in term2_variants:
                        pattern = rf'\b({v1})\s+(and|or)\s+({v2})\b'
                        matches = list(re.finditer(pattern, text, re.IGNORECASE))

                        for match in reversed(matches):
                            if self._is_within_quotes(doc, match.start(), match.end()):
                                continue

                            is_or = match.group(2).lower() == "or"

                            # Handle singular/plural for "or" cases
                            if is_or:
                                adjusted_replacements = []
                                for repl in replacements:
                                    if repl.endswith('s'):
                                        adjusted = repl[:-1]  # children -> child
                                    elif repl.endswith('es'):
                                        adjusted = repl[:-2]  # ladies -> lady
                                    else:
                                        adjusted = repl
                                    adjusted_replacements.append(adjusted)
                            else:
                                adjusted_replacements = replacements

                            first_word = match.group(1)
                            adjusted_replacements = [
                                self._adjust_capitalization(first_word, repl)
                                for repl in adjusted_replacements
                            ]

                            corrections.append({
                                "word_index": next((i for i, t in enumerate(doc) if t.idx <= match.start() < t.idx + len(t.text)), None),
                                "original_text": match.group(0),
                                "replacements": adjusted_replacements,
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
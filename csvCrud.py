# term_manager.py
from dataclasses import dataclass
from typing import List, Optional, Dict
import csv


@dataclass
class GenderTerm:
    term: str
    options: List[str]


class GenderTermManager:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> List[GenderTerm]:
        terms = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    terms.append(GenderTerm(row[0].strip(), [opt.strip() for opt in row[1:] if opt.strip()]))
        return terms

    def save(self, terms: List[GenderTerm]):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for entry in terms:
                writer.writerow([entry.term, *entry.options])

    def create(self, term: str, options: List[str]):
        all_terms = self.load()
        if any(entry.term == term for entry in all_terms):
            raise ValueError(f"Term '{term}' already exists.")
            # update existing term instead
        all_terms.append(GenderTerm(term, options))
        self.save(all_terms)

    def find_term(self, term: str) -> Optional[Dict[str, List[str]]]:
        for entry in self.load():
            if entry.term == term:
                return {entry.term: entry.options}
        return None

    def update(self, term: str, new_options: List[str]):
        terms = self.load()
        for entry in terms:
            if entry.term == term:
                entry.options = new_options
                self.save(terms)
                return
        raise KeyError(f"Term '{term}' not found.")

    def delete(self, term: str):
        terms = self.load()
        new_terms = [entry for entry in terms if entry.term != term]
        if len(new_terms) == len(terms):
            raise KeyError(f"Term '{term}' not found.")
        self.save(new_terms)

    def create_bulk(self, term_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
        all_terms = self.load()
        existing_terms = {entry.term for entry in all_terms}
        new_entries = []

        for term, options in term_dict.items():
            if term not in existing_terms:
                new_entries.append(GenderTerm(term, options))

        self.save(all_terms + new_entries)
        return {entry.term: entry.options for entry in new_entries}


    def get_all_term(self) -> Dict[str, List[str]]:
        return {entry.term: entry.options for entry in self.load()}

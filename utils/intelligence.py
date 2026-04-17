import re
from typing import List, Dict

class LegalIntelligence:
    STATES = ["Delhi", "Haryana", "Punjab", "Uttar Pradesh", "UP", "NCT of Delhi"]
    ENV_KEYWORDS = [
        "deforestation", "tree felling", "forest conservation", "environment", 
        "green belt", "ecologically sensitive", "biodiversity", "reforestation",
        "illegal logging", "timber", "encroachment", "National Green Tribunal", "NGT"
    ]
    LEGAL_TERMS = [
        "section", "article", "act", "provision", "petitioner", "respondent",
        "appellant", "writ", "plaintiff", "defendant", "judgment", "decree",
        "ordinance", "statute", "precedent", "affidavit", "suo motu"
    ]

    @staticmethod
    def detect_entities(text: str) -> Dict[str, List[str]]:
        """
        Detects states, environmental keywords, and legal terms in the text.
        """
        detected = {
            "states": [],
            "environmental_impact": [],
            "legal_terms": []
        }

        # Use regex for case-insensitive matching
        for state in LegalIntelligence.STATES:
            if re.search(r'\b' + re.escape(state) + r'\b', text, re.I):
                detected["states"].append(state)

        for kw in LegalIntelligence.ENV_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text, re.I):
                detected["environmental_impact"].append(kw)

        for term in LegalIntelligence.LEGAL_TERMS:
            if re.search(r'\b' + re.escape(term) + r'\b', text, re.I):
                detected["legal_terms"].append(term)

        # De-duplicate
        detected["states"] = list(set(detected["states"]))
        detected["environmental_impact"] = list(set(detected["environmental_impact"]))
        detected["legal_terms"] = list(set(detected["legal_terms"]))

        return detected

    @staticmethod
    def extract_principles(text: str) -> List[str]:
        """
        Tries to extract sentences that look like legal principles (e.g., "The principle of...").
        """
        principles = []
        # Pattern: "The principle of [X]" or "[X] principle" or "held that [X]"
        patterns = [
            r'(?i)the\s+principle\s+of\b[^.!?]*[.!?]',
            r'(?i)\bheld\s+that\b[^.!?]*[.!?]',
            r'(?i)\bis\s+well\s+settled\s+that\b[^.!?]*[.!?]'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                principles.append(match.group().strip())
        
        return list(set(principles))[:5] # Return top 5 unique principles

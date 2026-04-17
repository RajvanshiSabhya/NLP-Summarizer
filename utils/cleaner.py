import re

class LegalCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans the extracted legal text by removing noise, excessive whitespace, 
        and common header/footer patterns.
        """
        if not text:
            return ""

        # Remove page numbers and headers like "Page X of Y" or "Page X"
        text = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
        
        # Remove common court header noise (dates, case numbers in headers)
        # This is generic and might need tuning for specific court formats
        text = re.sub(r'\d{2}\.\d{2}\.\d{4}', '', text) # Remove dates like 16.04.2024
        
        # Remove multiple newlines and normalize spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove specific citation artifacts if they are noise
        # (e.g., [2024] INSC 123 -> keeping it might be useful for domain intelligence,
        # but for summarization we want the core text)
        
        return text.strip()

    @staticmethod
    def segment_legal_sections(text: str) -> dict:
        """
        Segments the text into logical parts based on common legal headings.
        """
        sections = {
            "facts": "",
            "arguments": "",
            "judgment": "",
            "reasoning": "",
            "issues": ""
        }
        
        # Define keywords for segmentation
        patterns = {
            "facts": r'(?i)(background|facts|brief facts|factual layout)',
            "issues": r'(?i)(issues|points for consideration|questions of law)',
            "arguments": r'(?i)(submissions|arguments|contentions|pleadings)',
            "reasoning": r'(?i)(reasoning|discussion|merits|consideration)',
            "judgment": r'(?i)(order|conclusion|judgment|final order|held)'
        }
        
        # This is a naive segmentation. A better way would be finding indices of these keywords
        # and splitting the text accordingly.
        
        parts = {}
        last_found = None
        current_text = text
        
        # Extract indices of found sections
        indices = []
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                indices.append((match.start(), key))
        
        # Sort indices by position
        indices.sort()
        
        if not indices:
            # If no sections found, return everything as facts as a fallback
            sections["facts"] = text
            return sections

        for i in range(len(indices)):
            start_idx, section_name = indices[i]
            end_idx = indices[i+1][0] if i+1 < len(indices) else len(text)
            sections[section_name] = text[start_idx:end_idx].strip()
            
        return sections

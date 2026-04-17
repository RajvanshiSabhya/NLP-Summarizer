import fitz  # PyMuPDF
from typing import List, Dict
import os

class PDFParser:
    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extracts raw text from a PDF file using PyMuPDF.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
            
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    @staticmethod
    def extract_metadata(file_path: str) -> Dict[str, str]:
        """
        Extracts basic metadata from the PDF.
        """
        doc = fitz.open(file_path)
        metadata = doc.metadata
        doc.close()
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creationDate": metadata.get("creationDate", ""),
            "modDate": metadata.get("modDate", "")
        }

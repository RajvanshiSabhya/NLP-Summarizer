import os
import shutil
import logging
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn

# Import custom utilities
from utils.pdf_parser import PDFParser
from utils.cleaner import LegalCleaner
from utils.summarizer import AbstractiveSummarizer
from utils.extractor import ExtractiveSummarizer
from utils.intelligence import LegalIntelligence

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="⚖️ Legal Document Summarizer API",
    description="""
    ## Advanced Legal NLP Pipeline
    This API processes Indian Legal PDFs (Court Judgments, environmental policies) and returns a structured summary.
    
    ### Features:
    * **PDF Extraction**: Extracts structured text from multi-page documents.
    * **Legal Intelligence**: Detects states, years, and legal principles.
    * **Abstractive Summarization**: Uses BART/T5 for human-like summaries of specific legal sections.
    * **Extractive Summarization**: Highlights the most important sentences for quick reading.
    """,
    version="2.0.0"
)

# Deployment & CORS Configuration
IS_HF = "SPACE_ID" in os.environ
TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp" if IS_HF else "data")
PORT = int(os.environ.get("PORT", 7860 if IS_HF else 8000))

# Ensure local data directory exists if needed
if not IS_HF and not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

raw_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,https://vanrakshakcm.vercel.app",
)
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https://.*\.hf\.space$|^https://.*\.vercel\.app$|^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
async def root():
    """
    ### API Health Check
    Returns the status of the API and model loading.
    """
    return {
        "status": "online",
        "message": "⚖️ Legal Document Summarizer API is running",
        "version": "2.0.0",
        "models_loaded": True
    }

# Initialize models (load on startup)
logger.info("Loading NLP models... this may take a moment.")
abs_summarizer = AbstractiveSummarizer()
ext_summarizer = ExtractiveSummarizer()
logger.info("Models loaded successfully.")


class SummaryOutput(BaseModel):
    case_title: str
    court: str
    year: str
    facts: str
    legal_issues: str
    arguments: str
    judgment: str
    reasoning: str
    environmental_impact: str
    detected_states: List[str]
    legal_principles: List[str]
    comprehensive_summary: str

@app.post("/summarize", response_model=SummaryOutput, tags=["Summarization"])
async def summarize_legal_doc(file: UploadFile = File(..., description="The Legal PDF file to summarize")):
    """
    ### Process a Legal PDF Document
    
    1. **Upload** a PDF file (e.g., Narinder_Singh_vs_Divesh_Bhutani).
    2. **NLP Engine** cleans text, segments sections, and runs intelligence extraction.
    3. **Returns** a structured JSON containing facts, logic, and final verdict.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file temporarily using a writable path
    temp_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Extract Text
        logger.info(f"Extracting text from {file.filename}")
        raw_text = PDFParser.extract_text(temp_path)
        metadata = PDFParser.extract_metadata(temp_path)
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="The PDF appears to be empty or corrupted.")

        # 2. Clean Text
        clean_text = LegalCleaner.clean_text(raw_text)

        # 3. Segment Text
        sections = LegalCleaner.segment_legal_sections(clean_text)

        # 4. Domain Intelligence
        intelligence = LegalIntelligence.detect_entities(clean_text)
        principles = LegalIntelligence.extract_principles(clean_text)

        # 5. Summarize Sections (Abstractive)
        # We summarize key sections using BART
        summary_sections = {}
        for section_name, content in sections.items():
            if content and len(content.split()) > 30:
                logger.info(f"Summarizing section: {section_name}")
                summary_sections[section_name] = abs_summarizer.summarize(content, max_length=150)
            else:
                summary_sections[section_name] = content or "No specific information found in this section."

        # 6. Generate Extractive Summary and Construct a Comprehensive Summary
        logger.info("Generating extractive highlights and aggregating summary")
        ext_summary = ext_summarizer.summarize(clean_text, num_sentences=5)
        
        # We build a structured summary using the internal results (Simple Text)
        summary_parts = []
        
        if summary_sections.get("facts") and summary_sections["facts"] != "No specific information found in this section.":
            summary_parts.append(f"FACTS:\n{summary_sections['facts']}")
            
        if summary_sections.get("issues") and summary_sections["issues"] != "No specific information found in this section.":
            summary_parts.append(f"LEGAL ISSUES:\n{summary_sections['issues']}")
            
        if summary_sections.get("judgment") and summary_sections["judgment"] != "No specific information found in this section.":
            summary_parts.append(f"FINAL JUDGMENT:\n{summary_sections['judgment']}")
            
        # Add the extractive highlight as a conclusion
        if ext_summary:
            summary_parts.append(f"KEY HIGHLIGHTS:\n{ext_summary}")

        full_summary = "\n\n".join(summary_parts) if summary_parts else ext_summary or "No summary could be generated."

        # 7. Construct Result
        result = SummaryOutput(
            case_title=metadata.get("title") or file.filename,
            court="Supreme Court of India" if "SUPREME COURT" in clean_text.upper() else "Unknown Court",
            year=re.search(r'\b(19|20)\d{2}\b', clean_text).group() if re.search(r'\b(19|20)\d{2}\b', clean_text) else "Unknown",
            facts=summary_sections.get("facts", ""),
            legal_issues=summary_sections.get("issues", ""),
            arguments=summary_sections.get("arguments", ""),
            judgment=summary_sections.get("judgment", ""),
            reasoning=summary_sections.get("reasoning", ""),
            environmental_impact=", ".join(intelligence["environmental_impact"]) if intelligence["environmental_impact"] else "No specific environmental impact keywords detected.",
            detected_states=intelligence["states"],
            legal_principles=principles,
            comprehensive_summary=full_summary
        )

        return result

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

import re # Needed for the year regex in main.py

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

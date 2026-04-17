---
title: Legal Summarizer
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Legal Document Summarizer (North Indian Focus)

A production-ready NLP system designed to summarize legal documents, specifically court judgments and policies related to environmental and deforestation cases in North India.

## Features
- **Input Processing**: Extract and clean text from PDF documents using PyMuPDF.
- **Section Segmentation**: Automatically identifies Facts, Arguments, Judgment, and Reasoning.
- **Abstractive Summarization**: Uses `facebook/bart-large-cnn` to generate concise, human-like summaries.
- **Extractive Summarization**: Identifies key sentences using BERT-based ranking.
- **Domain Intelligence**: 
    - Detects North Indian states (Delhi, Haryana, Punjab, UP).
    - Highlights environmental impact keywords (Deforestation, Tree felling, etc.).
    - Extracts core legal principles.
- **Structured Output**: Returns a detailed JSON object tailored for legal assistants.

## Tech Stack
- **Backend**: FastAPI
- **NLP**: Transformers (Hugging Face), Sentence-Transformers, NLTK
- **Vector Math**: Scikit-learn, NumPy
- **Containerization**: Docker

## Setup Instructions

### Local Setup
1. Clone the repository.
2. Install Python 3.10+.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.

### Docker Setup
1. Build the image:
   ```bash
   docker build -t legal-summarizer .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 legal-summarizer
   ```

## API Documentation

### POST `/summarize`
**Input**: multipart/form-data
- `file`: PDF Document

**Output**: JSON
```json
{
  "case_title": "Delhi High Court Order - 2024",
  "court": "Delhi High Court",
  "year": "2024",
  "facts": "...",
  "legal_issues": "...",
  "arguments": "...",
  "judgment": "...",
  "reasoning": "...",
  "final_verdict": "...",
  "environmental_impact": "deforestation, tree felling",
  "detected_states": ["Delhi"],
  "legal_principles": ["The principle of absolute liability..."],
  "extractive_summary": "..."
}
```

## Example Usage (cURL)
```bash
curl -X 'POST' \
  'http://localhost:8000/summarize' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_legal_document.pdf;type=application/pdf'
```

## Project Structure
- `main.py`: FastAPI entry point.
- `utils/`: Core logic modules.
  - `pdf_parser.py`: PDF extraction.
  - `cleaner.py`: Text preprocessing.
  - `summarizer.py`: BART summarization.
  - `extractor.py`: BERT extractive ranking.
  - `intelligence.py`: Domain entity detection.
- `data/`: Temporary storage for processing.
- `models/`: Fine-tuned model directory.

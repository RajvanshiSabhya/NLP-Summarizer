# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models to cache them in the Docker image
# This ensures faster startup and offline capability
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('facebook/bart-large-cnn'); \
    AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')"
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -m nltk.downloader punkt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (7860 is default for Hugging Face)
EXPOSE 7860

# Command to run the application (PORT is usually set by HF, but we default to 7860)
CMD ["python", "main.py"]

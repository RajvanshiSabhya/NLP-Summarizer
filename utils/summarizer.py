from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List

class AbstractiveSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate_summary(self, text: str, max_length: int = 250, min_length: int = 40) -> str:
        """
        Internal helper to generate summary for a single chunk.
        """
        try:
            inputs = self.tokenizer(text, max_length=1024, return_tensors="pt", truncation=True).to(self.device)
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                num_beams=4, 
                max_length=max_length, 
                min_length=min_length, 
                early_stopping=True
            )
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in manual generation: {e}")
            return text[:500] + "..."

    def summarize(self, text: str, max_length: int = 150, min_length: int = 50, chunk_size: int = 800, overlap: int = 100) -> str:
        """
        Summarizes long text using a sliding window approach.
        """
        if not text or len(text.split()) < 20:
            return text

        tokens = self.tokenizer.encode(text, truncation=False)
        total_tokens = len(tokens)
        
        # If text is small enough, summarize directly
        if total_tokens <= 1024:
            return self.generate_summary(text, max_length=max_length, min_length=min_length)

        # Chunk the text
        summaries = []
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunk_summary = self.generate_summary(chunk_text, max_length=max_length, min_length=min_length)
            summaries.append(chunk_summary)
            
            start += (chunk_size - overlap)
            if start >= total_tokens:
                break

        # Merge summaries
        merged_summary = " ".join(summaries)
        
        # If merged summary is still too long, summarize it again
        if len(self.tokenizer.encode(merged_summary)) > 1024:
            return self.summarize(merged_summary, max_length=max_length, min_length=min_length)
        
        return merged_summary

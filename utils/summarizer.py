from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List

class AbstractiveSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Explicitly set model max length to suppress warnings and ensure truncation
        self.tokenizer.model_max_length = 1024
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate_summary(self, text: str, max_length: int = 250, min_length: int = 40) -> str:
        """
        Internal helper to generate summary for a single chunk.
        """
        try:
            # Always use truncation and clear max_length to avoid indexing errors
            inputs = self.tokenizer(
                text, 
                max_length=1024, 
                return_tensors="pt", 
                truncation=True
            ).to(self.device)
            
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                num_beams=4, 
                max_length=max_length, 
                min_length=min_length, 
                early_stopping=True
            )
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in summary generation: {e}")
            return text[:500] + "..."

    def summarize(self, text: str, max_length: int = 150, min_length: int = 50, chunk_size: int = 700, overlap: int = 100) -> str:
        """
        Summarizes long text using a sliding window approach with strict token limits.
        """
        if not text or len(text.split()) < 20:
            return text

        # Use truncation=True and max_length to avoid the "1707 > 1024" warning
        # We just want to check if the total length is over the limit
        full_tokens = self.tokenizer.encode(text, truncation=True, max_length=1024)
        
        # If text is small enough, summarize directly
        # Check against decoded text length roughly or use a separate non-truncating encode for size check
        # To avoid the warning, we can use a higher max_length or ignore the warning
        
        # Proper way to check length without warning:
        with torch.no_grad():
            token_count = len(self.tokenizer.encode(text, add_special_tokens=False))

        if token_count <= 1000: # Leave some buffer for special tokens
            return self.generate_summary(text, max_length=max_length, min_length=min_length)

        # Chunk the text using tokens for accuracy
        all_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(all_tokens)
        
        summaries = []
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = all_tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunk_summary = self.generate_summary(chunk_text, max_length=max_length, min_length=min_length)
            summaries.append(chunk_summary)
            
            start += (chunk_size - overlap)
            if start >= total_tokens - overlap: # Avoid tiny tail chunks
                break

        # Merge summaries
        merged_summary = " ".join(summaries)
        
        # Recursive check: if merged summary is still too long, summarize it again
        # We use a slightly smaller limit for the recursive call to ensure progress
        merged_token_count = len(self.tokenizer.encode(merged_summary, add_special_tokens=False))
        if merged_token_count > 1000:
            return self.summarize(merged_summary, max_length=max_length, min_length=min_length, chunk_size=600)
        
        return merged_summary

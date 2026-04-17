import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from typing import List

# Download sentence tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ExtractiveSummarizer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def summarize(self, text: str, num_sentences: int = 5) -> str:
        """
        Generates an extractive summary using sentence embeddings and 
        centrality (centroid) ranking.
        """
        if not text:
            return ""

        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Compute embeddings
        embeddings = self.model.encode(sentences)
        
        # Calculate the document centroid
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        
        # Calculate similarity of each sentence to the centroid
        similarities = cosine_similarity(embeddings, centroid).flatten()
        
        # Get the indices of the most similar sentences
        ranked_indices = np.argsort(similarities)[::-1]
        
        # Select top N sentences and sort them by original order
        top_indices = sorted(ranked_indices[:num_sentences])
        
        summary = " ".join([sentences[i] for i in top_indices])
        return summary

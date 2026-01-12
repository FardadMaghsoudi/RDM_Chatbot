from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def clean_text(text):
    """Clean up excessive whitespace and line breaks from scraped text."""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    # Join lines, preserving paragraph breaks (double newlines)
    cleaned = '\n'.join(lines)
    # Normalize paragraph breaks
    cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
    return cleaned.strip()

def split_into_sentences(text):
    """Split text into sentences while preserving sentence boundaries."""
    # Simple sentence splitter - handles common abbreviations
    # Splits on periods, exclamation marks, and question marks followed by space/newline
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def split_text(text, chunk_size=1500, overlap=200):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def split_text_by_sentences(text, target_chunk_size=1500, overlap_size=200):
    """
    Split text into chunks by complete sentences, preserving sentence boundaries.
    
    Args:
        text: The text to chunk
        target_chunk_size: Target size in characters (default: 1200)
        overlap_size: Overlap between chunks in characters (default: 200)
    
    Returns:
        List of text chunks
    """
    # Clean the text first
    text = clean_text(text)
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed target size and we have content
        if current_length + sentence_length > target_chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            # Keep sentences from the end that fit within overlap_size
            overlap_chunk = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap_size:
                    overlap_chunk.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            
            current_chunk = overlap_chunk
            current_length = overlap_length
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

class SimpleVectorStore:
    def __init__(self, texts):
        self.texts = texts
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(texts)

    def similarity_search(self, query, k=4):
        chunked_query = split_text_by_sentences(query)
        query_vec = self.model.encode(chunked_query)
        sims = cosine_similarity(query_vec, self.embeddings)
        aggregated = np.max(sims, axis=0)
        top_k_idx = np.argsort(aggregated)[::-1][:k]
        return [self.texts[i] for i in top_k_idx] 
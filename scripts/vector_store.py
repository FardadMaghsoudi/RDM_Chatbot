from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import html
import unicodedata
import faiss
from collections import Counter
from cleantext import clean

def clean_text(text):
    cleaned_text = clean(
            text,
            fix_unicode=True,
            lower=True,
            no_urls=False,  # Keep URLs for reference
            no_emails=False,  # Keep emails for contact info
            #no_punct=True,  # Remove for better search matching
            normalize_whitespace=True,
            no_line_breaks=True
        )
    return cleaned_text

def split_into_sentences(text):
    """Split text into sentences while preserving sentence boundaries."""
    # Simple sentence splitter - handles common abbreviations
    # Splits on periods, exclamation marks, and question marks followed by space/newline
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def split_text(text, chunk_size=1000, overlap=200):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def split_text_by_sentences(text, target_chunk_size=2000, overlap_size=200):
    """
    Split text into chunks by complete sentences, preserving sentence boundaries.
    
    Args:
        text: The text to chunk
        target_chunk_size: Target size in characters (default: 2000)
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
        self.model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

    def similarity_search(self, query, k=5):
        query_vec = self.model.encode([query], normalize_embeddings=True)
        # D represents distances (similarity scores), I represents indices
        D, I = self.index.search(query_vec, k)
        # Retrieve texts based on the indices returned by Faiss
        return [self.texts[i] for i in I[0]]

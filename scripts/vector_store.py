from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import html
import unicodedata
from collections import Counter

def clean_text(text):
    """Clean up excessive whitespace and line breaks from scraped text."""
    # Normalize unicode (fixes different fonts)
    # Converts distinct font characters (e.g., 𝐇, 𝑓, 𝐱) into standard ASCII (H, f, x).
    # Also fixes combined accents and some invisible formatting chars.
    text = unicodedata.normalize('NFKC', text)

    # HTML decoding (web specific)
    # Converts entities like &amp;, &nbsp;, &quot; into actual characters
    text = html.unescape(text)
   
    # Fix Bullet Points: â<97><8b> is the byte sequence for ●
    text = text.replace('â<97><8b>', ' ')

    # Fix the other bullet point: â<97><8f> (Filled Circle)
    text = text.replace('â<97><8f>', ' ')

    # Fix Smart Quotes: â<80><99> is the byte sequence for ’ (apostrophe)
    text = text.replace('â<80><99>', "'")
   
    # Anything else
    text = re.sub(r'[●•■◆▪○]', ' ', text)

    # Remove hex byte markers like <80>, <94>, <9c>
    text = re.sub(r'<[0-9a-fA-F]{2}>', '', text)

    # Remove caret notation for control chars like ^D, ^A, ^@
    text = re.sub(r'\^[A-Z@\[\]\\\^_]', '', text)

    # Remove residual garbage patterns often left after hex removal
    # Example: "X." or "X|" or "Xt" followed by remaining binary noise
    text = re.sub(r'(?<!\w)X(?!\w)', '', text)

    # Remove artifacts (?X?, ?Xt, ?Xd)
    # These appear to be failed icon conversions acting as breaks.
    # We replace them with a newline to prevent merging sentences.
    text = re.sub(r'\?X\S?', '\n', text)

    # Fix missing spaces
    # Case A: Lowercase immediately followed by Uppercase (e.g., "PolicyThe")
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    # Case B: Number immediately followed by Uppercase (e.g., "2020The")
    text = re.sub(r'(?<=[0-9])(?=[A-Z])', ' ', text)
    # Case C: Letter immediately followed by Number (e.g., "implementation1")
    # (Common for footnote markers; we add a space to separate them)
    text = re.sub(r'(?<=[a-z])(?=[0-9])', ' ', text)

    # Fix "wide" text (e.g., "M a n a g e m e n t" -> "Management")
    # Look for single letters surrounded by word boundaries and separated by spaces
    # Note: We restrict this to sequences of 3+ characters to avoid merging "I a m" -> "Iam"
    text = re.sub(r'\b(?:[A-Za-z]\s){2,}[A-Za-z]\b', 
                  lambda m: m.group().replace(" ", ""), text)

    # De-hyphenation: Join words split by a hyphen and a newline (e.g., "facul-\n ty")
    # Finds a word char, a hyphen, optional whitespace/newlines, and another word char
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # Clean up specific non-standard whitespace (common in PDFs)
    # Replace non-breaking spaces (\xa0) and tabs with standard spaces
    text = text.replace(u'\xa0', ' ').replace(u'\u200b', '')

    # Standardize paragraph breaks
    # Replace multiple newlines with a unique marker to preserve paragraphs
    text = re.sub(r'\n\s*\n+', '<PARAGRAPH_BREAK>', text)
    
    # Join lines within paragraphs (Fixes "breaks between letters" caused by hard wraps)
    # This treats single newlines as spaces, which is usually correct for PDF prose
    text = text.replace('\n', ' ')
    
    # Restore paragraph breaks
    text = text.replace('<PARAGRAPH_BREAK>', '\n\n')
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

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

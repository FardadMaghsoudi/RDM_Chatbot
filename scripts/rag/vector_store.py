from sentence_transformers import SentenceTransformer
import re
import unicodedata
import faiss

# Splits on periods, exclamation marks and question marks followed by whitespace and a capital letter
SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
# Paragraph breaks (blank lines): web pages separate paragraphs/list items with them
PARAGRAPH_BREAK = re.compile(r'\n\s*\n')
# Separator inserted between pages by pdf_utils.load_prdw_pdf_text; not content
PAGE_BREAK_MARKER = re.compile(r'^-+ PAGE BREAK -+$')

# Bullet glyphs from symbol fonts are extracted as private-use characters (e.g. U+F0B7)
PRIVATE_USE_CHARS = re.compile('[\ue000-\uf8ff]')

def clean_text(text):
    """
    Normalise chunk text without removing content: unicode normalisation (e.g. ligatures,
    non-breaking spaces), symbol-font bullets and collapsing whitespace. Punctuation and stopwords
    are kept, because the chunks are given to the LLM as context and the sentence splitter needs
    the punctuation.
    """
    text = unicodedata.normalize("NFKC", text)
    # Some PDFs (e.g. the EEMCS policy) use soft hyphens instead of spaces between words
    text = text.replace("\u00ad", " ")
    text = PRIVATE_USE_CHARS.sub("•", text)
    return re.sub(r'\s+', ' ', text).strip()

def split_into_sentences(text):
    """Split text into cleaned sentences. Paragraph breaks (blank lines) also end a sentence."""
    sentences = []
    for paragraph in PARAGRAPH_BREAK.split(text):
        paragraph = clean_text(paragraph)
        if not paragraph or PAGE_BREAK_MARKER.match(paragraph):
            continue
        sentences.extend(s.strip() for s in SENTENCE_END.split(paragraph) if s.strip())
    return sentences

def split_long_sentence(sentence, max_length):
    """Split a sentence longer than max_length into pieces at word boundaries."""
    pieces = []
    current = ""
    for word in sentence.split(" "):
        if current and len(current) + 1 + len(word) > max_length:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces

def split_text(text, chunk_size=1000, overlap=200):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def split_text_by_sentences(text, target_chunk_size=2000, overlap_size=200, min_chunk_size=50):
    """
    Split text into chunks by complete sentences, preserving sentence boundaries.
    
    Args:
        text: The text to chunk
        target_chunk_size: Target size in characters (default: 2000)
        overlap_size: Overlap between chunks in characters (default: 200)
        min_chunk_size: Chunks shorter than this are dropped, e.g. pages that only contain
            "Filter by:" or "Contact" (default: 50)
    
    Returns:
        List of text chunks
    """
    # Split into cleaned sentences; sentences longer than a chunk are split at word boundaries
    sentences = []
    for sentence in split_into_sentences(text):
        if len(sentence) > target_chunk_size:
            sentences.extend(split_long_sentence(sentence, target_chunk_size))
        else:
            sentences.append(sentence)
    
    chunks = []
    current_chunk = []
    current_length = 0
    overlap_count = 0  # number of sentences at the start of current_chunk repeated from the previous chunk
    
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
            overlap_count = len(overlap_chunk)
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it has content
    if current_chunk:
        last_chunk = ' '.join(current_chunk)
        if len(last_chunk) >= min_chunk_size:
            chunks.append(last_chunk)
        elif chunks:
            # Too short to stand on its own: append the new sentences to the previous chunk
            chunks[-1] = ' '.join([chunks[-1], *current_chunk[overlap_count:]])
        # else: the whole text is shorter than min_chunk_size (e.g. only "Filter by:"), so drop it

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

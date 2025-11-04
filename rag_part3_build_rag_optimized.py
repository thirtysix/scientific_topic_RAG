#!/usr/bin/env python3
"""
RAG System - Part 3: RAG Construction (OPTIMIZED VERSION v0.02)
Takes literature.json file and builds a complete RAG system with embeddings and vector database.

OPTIMIZATIONS v0.02:
- REDUCED mega-batch size from 1000 to 300 chunks (prevents GPU OOM at 50%)
- REDUCED batch size from 48 to 24 (lower memory per batch)
- FP16 half-precision support (reduces memory usage by ~50%)
- Model offloading to CPU between mega-batches (frees GPU memory)
- Enhanced memory monitoring with before/after stats
- More aggressive garbage collection after each mega-batch
- Improved GPU cache clearing with synchronization
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# ============================================================================
# 🎯 RAG CONSTRUCTION CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Text processing settings - Hybrid approach with multiple granularities
# Small chunks for precise retrieval
SMALL_CHUNK_SIZE = 100                   # ~2-3 sentences for fine-grained matching
SMALL_CHUNK_OVERLAP = 25                 # Overlap for small chunks

# Medium chunks for balanced context
MEDIUM_CHUNK_SIZE = 250                  # ~1 paragraph or section
MEDIUM_CHUNK_OVERLAP = 50                # Overlap for medium chunks

# Large chunks (kept for compatibility)
CHUNK_SIZE = 512                         # Full abstract or multiple paragraphs
CHUNK_OVERLAP = 50                       # Overlap between chunks
MAX_CHUNK_LENGTH = 1000                  # Maximum chunk length

# Chunking strategy - Choose ONE approach
CHUNKING_STRATEGY = "small_chunks_only"   # Options: "hybrid", "small_chunks_only", "semantic_only"

# Semantic segmentation settings (used when CHUNKING_STRATEGY = "hybrid" or "semantic_only")
ENABLE_SEMANTIC_SEGMENTATION = False      # Split abstracts by sections
ENABLE_KEY_SENTENCE_EXTRACTION = True    # Extract important sentences
ENABLE_MULTI_GRANULARITY = True          # Create multiple chunk sizes

# Small chunks only settings (used when CHUNKING_STRATEGY = "small_chunks_only")
SMALL_CHUNKS_ONLY_SIZE = 100             # Size for small-chunks-only approach
SMALL_CHUNKS_ONLY_OVERLAP = 25           # Overlap for small-chunks-only approach

# Text content settings
INCLUDE_TITLE_IN_CHUNKS = False           # Whether to include paper titles in chunk text

# Debug settings
DEBUG_CHUNK_GENERATION = True            # Show detailed chunk generation per paper
DEBUG_MAX_PAPERS = 1                    # Max papers to show debug info for (0 = all)
DEBUG_SHOW_FULL_CHUNKS = True           # Show complete chunk text (warning: very verbose!)

# Embedding settings
# Options available:
# 1. "sentence-transformers/all-mpnet-base-v2" - High quality general purpose (768 dims)
# 2. "allenai/scibert_scivocab_uncased" - SciBERT trained on scientific text (768 dims)
# 3. "dmis-lab/biobert-base-cased-v1.2" - BioBERT trained on PubMed (768 dims)
# 4. "sentence-transformers/all-MiniLM-L6-v2" - Fast general purpose (384 dims)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # High quality embeddings
EMBEDDING_BATCH_SIZE = 24                # Batch size for embedding generation (reduced for GPU memory)
EMBEDDING_MEGA_BATCH_SIZE = 300          # Process chunks in mega-batches to avoid memory issues (reduced from 1000)
EMBEDDING_DEVICE = "cpu"
EMBEDDING_DEVICE = "cuda" # "cpu" or "cuda" if GPU available
EMBEDDING_USE_FP16 = True                # Use half precision (FP16) to reduce memory usage by ~50%
EMBEDDING_OFFLOAD_MODEL = True           # Offload model to CPU between mega-batches to free GPU memory
EMBEDDING_MONITOR_MEMORY = True          # Monitor GPU memory usage during processing

# Vector database settings
VECTOR_DB_TYPE = "chroma"                # "chroma", "faiss", or "pinecone"
SIMILARITY_THRESHOLD = 0.7               # Minimum similarity for retrieval
MAX_RETRIEVAL_RESULTS = 10               # Maximum documents to retrieve

# RAG generation settings  
GENERATION_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # LLM for generation
GENERATION_TEMPERATURE = 0.7             # Temperature for generation
MAX_GENERATION_TOKENS = 1024            # Maximum tokens in generated response

# ============================================================================
# 🔧 IMPLEMENTATION
# ============================================================================

def extract_paper_texts(literature_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and prepare text from papers for chunking"""
    
    print("📄 Extracting text from papers...")
    
    papers = literature_data.get('papers', [])
    if not papers:
        raise ValueError("No papers found in literature data")
    
    documents = []
    for paper in papers:
        # Combine title and abstract for full text
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        # Skip papers without abstracts
        if not abstract or abstract.lower() in ['', 'n/a', 'not available']:
            continue
            
        # Choose whether to include title
        if INCLUDE_TITLE_IN_CHUNKS:
            full_text = f"{title}. {abstract}"
        else:
            full_text = abstract  # Abstract only for better semantic matching
        
        # Clean metadata for ChromaDB (no None values)
        clean_metadata = {
            'pmid': str(paper.get('pmid') or ''),
            'title': str(title or ''),
            'authors': ', '.join(paper.get('authors') or []),
            'journal': str(paper.get('journal') or ''),
            'year': str(paper.get('year') or ''),
            'doi': str(paper.get('doi') or ''),
            'relevance_score': float(paper.get('relevance_score') or 0.0),
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid') or ''}"
        }
        
        document = {
            'text': full_text,
            'metadata': clean_metadata
        }
        documents.append(document)
    
    print(f"   Extracted text from {len(documents)} papers")
    return documents


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split documents into chunks for embedding - using sentence-aware chunking"""

    print("✂️  Chunking documents (sentence-aware)...")

    # Try to import NLTK for sentence tokenization
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("   Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
    except ImportError:
        print("⚠️  NLTK not installed, falling back to simple sentence splitting")
        nltk = None

    chunks = []
    for doc_idx, document in enumerate(documents):
        text = document['text']
        metadata = document['metadata'].copy()

        # Clean text: remove double newlines, normalize whitespace
        text = text.replace('\\n\\n', ' ')  # Handle escaped newlines
        text = text.replace('\n\n', '. ')    # Handle actual newlines
        text = text.replace('\\n', ' ')      # Handle escaped single newlines
        text = text.replace('\n', ' ')       # Handle actual single newlines
        text = ' '.join(text.split())        # Normalize whitespace

        # Get sentences
        if nltk:
            sentences = nltk.sent_tokenize(text)
        else:
            # Fallback: simple sentence splitting
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        # Build chunks from complete sentences
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            # If adding this sentence exceeds chunk size and we have content
            if current_word_count + sentence_word_count > CHUNK_SIZE and current_chunk:
                # Save the current chunk
                chunk_text = ' '.join(current_chunk)

                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'doc_index': doc_idx,
                    'chunk_index': len(chunks),
                    'sentence_count': len(current_chunk),
                    'word_count': current_word_count
                })

                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata,
                    'chunk_type': 'content'
                })

                # Start new chunk with overlap
                if CHUNK_OVERLAP > 0:
                    # Keep last few sentences for context overlap
                    overlap_sentences = []
                    overlap_words = 0

                    for sent in reversed(current_chunk):
                        sent_words = len(sent.split())
                        if overlap_words + sent_words <= CHUNK_OVERLAP:
                            overlap_sentences.insert(0, sent)
                            overlap_words += sent_words
                        else:
                            break

                    current_chunk = overlap_sentences
                    current_word_count = overlap_words
                else:
                    current_chunk = []
                    current_word_count = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        # Don't forget the last chunk
        if current_chunk and current_word_count >= 50:  # Minimum 50 words
            chunk_text = ' '.join(current_chunk)

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'doc_index': doc_idx,
                'chunk_index': len(chunks),
                'sentence_count': len(current_chunk),
                'word_count': current_word_count
            })

            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata,
                'chunk_type': 'content'
            })

    print(f"   Created {len(chunks)} sentence-aware chunks from {len(documents)} documents")
    return chunks


def create_small_chunks_only(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create only small chunks for faster processing and focused retrieval"""

    print(f"✂️  Creating small chunks only ({SMALL_CHUNKS_ONLY_SIZE} words, {SMALL_CHUNKS_ONLY_OVERLAP} overlap)...")
    if DEBUG_CHUNK_GENERATION:
        print(f"   Debug mode: showing details for first {DEBUG_MAX_PAPERS if DEBUG_MAX_PAPERS > 0 else 'all'} papers")

    all_chunks = []

    for doc_idx, document in enumerate(documents):
        text = document['text']
        metadata = document['metadata'].copy()

        # Debug output for first few papers
        show_debug = DEBUG_CHUNK_GENERATION and (DEBUG_MAX_PAPERS == 0 or doc_idx < DEBUG_MAX_PAPERS)

        if show_debug:
            print(f"\n📋 Paper {doc_idx + 1}: PMID {metadata.get('pmid', 'N/A')}")
            print(f"   Title: {metadata.get('title', 'N/A')[:80]}...")
            print(f"   Original text length: {len(text)} chars")

        # Clean text
        text = text.replace('\\n\\n', ' ')
        text = text.replace('\n\n', '. ')
        text = text.replace('\\n', ' ')
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())

        if show_debug:
            print(f"   Cleaned text length: {len(text)} chars")

        # Get sentences for chunking
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(text)
        except:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if show_debug:
            print(f"   📏 Total sentences: {len(sentences)}")

        # Create small chunks using sentence boundaries
        current_chunk = []
        current_word_count = 0
        chunks_created = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            # If adding this sentence exceeds chunk size and we have content
            if current_word_count + sentence_word_count > SMALL_CHUNKS_ONLY_SIZE and current_chunk:
                # Save the current chunk
                chunk_text = ' '.join(current_chunk)

                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'doc_index': doc_idx,
                    'chunk_index': len(all_chunks),
                    'chunk_type': 'small_chunk',
                    'sentence_count': len(current_chunk),
                    'word_count': current_word_count
                })

                all_chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata,
                    'chunk_type': 'small_chunk'
                })
                chunks_created += 1

                if show_debug and DEBUG_SHOW_FULL_CHUNKS:
                    print(f"     Chunk {chunks_created}: {chunk_text}")
                elif show_debug:
                    print(f"     Chunk {chunks_created}: {chunk_text[:80]}... ({current_word_count} words)")

                # Start new chunk with overlap
                if SMALL_CHUNKS_ONLY_OVERLAP > 0:
                    # Keep last few sentences for context overlap
                    overlap_sentences = []
                    overlap_words = 0

                    for sent in reversed(current_chunk):
                        sent_words = len(sent.split())
                        if overlap_words + sent_words <= SMALL_CHUNKS_ONLY_OVERLAP:
                            overlap_sentences.insert(0, sent)
                            overlap_words += sent_words
                        else:
                            break

                    current_chunk = overlap_sentences
                    current_word_count = overlap_words
                else:
                    current_chunk = []
                    current_word_count = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        # Don't forget the last chunk
        if current_chunk and current_word_count >= 20:  # Minimum 20 words
            chunk_text = ' '.join(current_chunk)

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'doc_index': doc_idx,
                'chunk_index': len(all_chunks),
                'chunk_type': 'small_chunk',
                'sentence_count': len(current_chunk),
                'word_count': current_word_count
            })

            all_chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata,
                'chunk_type': 'small_chunk'
            })
            chunks_created += 1

            if show_debug and DEBUG_SHOW_FULL_CHUNKS:
                print(f"     Chunk {chunks_created}: {chunk_text}")
            elif show_debug:
                print(f"     Chunk {chunks_created}: {chunk_text[:80]}... ({current_word_count} words)")

        if show_debug:
            print(f"   📊 Total chunks for this paper: {chunks_created}")
            print("   " + "="*60)

    print(f"   Created {len(all_chunks)} small chunks from {len(documents)} documents")
    return all_chunks


def segment_abstract_by_sections(text: str) -> Dict[str, str]:
    """Segment abstract into semantic sections based on common patterns"""
    import re

    sections = {}

    # Common section patterns in scientific abstracts
    section_patterns = {
        'background': r'(background|introduction|context|objective|aim|purpose)',
        'methods': r'(method|approach|material|procedure|technique|experiment)',
        'results': r'(result|finding|outcome|observation|discover)',
        'conclusions': r'(conclusion|significance|implication|summary|future)'
    }

    # Try to find section markers in the text
    sentences = text.split('. ')
    current_section = 'main'
    current_text = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        # Check if sentence starts a new section
        section_found = False
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, sentence_lower[:100]):  # Check first 100 chars
                # Save previous section if it has content
                if current_text:
                    sections[current_section] = '. '.join(current_text) + '.'
                    current_text = []

                current_section = section_name
                section_found = True
                break

        current_text.append(sentence)

    # Save last section
    if current_text:
        sections[current_section] = '. '.join(current_text) + '.'

    # If no sections found, treat whole text as one section
    if len(sections) <= 1:
        sections = {'full_abstract': text}

    return sections


def extract_key_sentences(text: str, max_sentences: int = 5) -> List[str]:
    """Extract key sentences that likely contain important findings"""
    import re

    sentences = text.split('. ')
    key_sentences = []

    # Patterns indicating important sentences
    importance_patterns = [
        r'significant',
        r'demonstrate',
        r'reveal',
        r'show[s]?\s+that',
        r'conclude',
        r'suggest',
        r'indicate',
        r'found that',
        r'discovered',
        r'novel',
        r'first time',
        r'important',
        r'critical',
        r'essential',
        r'we found',
        r'our results',
        r'these findings'
    ]

    # Score each sentence
    scored_sentences = []
    for sentence in sentences:
        if len(sentence.split()) < 10:  # Skip very short sentences
            continue

        score = 0
        sentence_lower = sentence.lower()

        # Check for importance indicators
        for pattern in importance_patterns:
            if re.search(pattern, sentence_lower):
                score += 2

        # Bonus for sentences with numbers (often results)
        if re.search(r'\d+\.?\d*\s*%', sentence):
            score += 1
        if re.search(r'p\s*[<=]\s*0\.\d+', sentence_lower):
            score += 2
        if re.search(r'n\s*=\s*\d+', sentence_lower):
            score += 1

        # Store sentence with score
        if score > 0:
            scored_sentences.append((score, sentence))

    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    key_sentences = [sent for _, sent in scored_sentences[:max_sentences]]

    return key_sentences


def create_hybrid_chunks(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create multiple types of chunks using hybrid approach"""

    print("🔬 Creating hybrid chunks (semantic + multi-granularity)...")
    if DEBUG_CHUNK_GENERATION:
        print(f"   Debug mode: showing details for first {DEBUG_MAX_PAPERS if DEBUG_MAX_PAPERS > 0 else 'all'} papers")

    all_chunks = []

    for doc_idx, document in enumerate(documents):
        text = document['text']
        metadata = document['metadata'].copy()

        # Debug output for first few papers
        show_debug = DEBUG_CHUNK_GENERATION and (DEBUG_MAX_PAPERS == 0 or doc_idx < DEBUG_MAX_PAPERS)

        if show_debug:
            print(f"\n📋 Paper {doc_idx + 1}: PMID {metadata.get('pmid', 'N/A')}")
            print(f"   Title: {metadata.get('title', 'N/A')[:80]}...")
            print(f"   Original text length: {len(text)} chars")

        # Clean text
        text = text.replace('\\n\\n', ' ')
        text = text.replace('\n\n', '. ')
        text = text.replace('\\n', ' ')
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())

        if show_debug:
            print(f"   Cleaned text length: {len(text)} chars")
            print(f"   Text preview: {text[:150]}...")

        # 1. Semantic segmentation
        sections_created = 0
        if ENABLE_SEMANTIC_SEGMENTATION:
            sections = segment_abstract_by_sections(text)

            if show_debug:
                print(f"   🔍 Semantic sections found: {list(sections.keys())}")

            for section_name, section_text in sections.items():
                if len(section_text.split()) >= 20:  # Minimum 20 words
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'doc_index': doc_idx,
                        'chunk_index': len(all_chunks),
                        'chunk_type': 'semantic_section',
                        'section_name': section_name,
                        'word_count': len(section_text.split())
                    })

                    all_chunks.append({
                        'text': section_text,
                        'metadata': chunk_metadata,
                        'chunk_type': 'semantic_section'
                    })
                    sections_created += 1

                    if show_debug:
                        print(f"     ✓ {section_name}: {len(section_text.split())} words")
                        if DEBUG_SHOW_FULL_CHUNKS:
                            print(f"       Full text: {section_text}")
                        else:
                            print(f"       Preview: {section_text[:100]}...")
                elif show_debug:
                    print(f"     ✗ {section_name}: {len(section_text.split())} words (too short)")

            if show_debug:
                print(f"   → Created {sections_created} semantic section chunks")

        # 2. Key sentence extraction
        key_sentences_created = 0
        if ENABLE_KEY_SENTENCE_EXTRACTION:
            key_sentences = extract_key_sentences(text)

            if show_debug:
                print(f"   🎯 Key sentences found: {len(key_sentences)}")

            for sent_idx, sentence in enumerate(key_sentences):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'doc_index': doc_idx,
                    'chunk_index': len(all_chunks),
                    'chunk_type': 'key_sentence',
                    'sentence_rank': sent_idx + 1,
                    'word_count': len(sentence.split())
                })

                all_chunks.append({
                    'text': sentence,
                    'metadata': chunk_metadata,
                    'chunk_type': 'key_sentence'
                })
                key_sentences_created += 1

                if show_debug:
                    if DEBUG_SHOW_FULL_CHUNKS:
                        print(f"     {sent_idx + 1}. {sentence}")
                    else:
                        print(f"     {sent_idx + 1}. {sentence[:80]}...")

            if show_debug:
                print(f"   → Created {key_sentences_created} key sentence chunks")

        # 3. Multi-granularity chunks
        small_chunks_created = 0
        medium_chunks_created = 0
        if ENABLE_MULTI_GRANULARITY:
            # Get sentences for granular chunking
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                sentences = nltk.sent_tokenize(text)
            except:
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
                sentences = [s.strip() for s in sentences if s.strip()]

            if show_debug:
                print(f"   📏 Total sentences for chunking: {len(sentences)}")

            # Create small chunks
            small_chunks = create_chunks_by_size(
                sentences, SMALL_CHUNK_SIZE, SMALL_CHUNK_OVERLAP,
                metadata, doc_idx, 'small', all_chunks
            )
            all_chunks.extend(small_chunks)
            small_chunks_created = len(small_chunks)

            # Create medium chunks
            medium_chunks = create_chunks_by_size(
                sentences, MEDIUM_CHUNK_SIZE, MEDIUM_CHUNK_OVERLAP,
                metadata, doc_idx, 'medium', all_chunks
            )
            all_chunks.extend(medium_chunks)
            medium_chunks_created = len(medium_chunks)

            if show_debug:
                print(f"   → Created {small_chunks_created} small chunks ({SMALL_CHUNK_SIZE} words each)")
                if DEBUG_SHOW_FULL_CHUNKS and small_chunks_created > 0:
                    for i, chunk in enumerate(small_chunks):
                        print(f"     Small chunk {i+1}: {chunk['text']}")

                print(f"   → Created {medium_chunks_created} medium chunks ({MEDIUM_CHUNK_SIZE} words each)")
                if DEBUG_SHOW_FULL_CHUNKS and medium_chunks_created > 0:
                    for i, chunk in enumerate(medium_chunks):
                        print(f"     Medium chunk {i+1}: {chunk['text']}")

        # Summary for this paper
        if show_debug:
            total_paper_chunks = sections_created + key_sentences_created + small_chunks_created + medium_chunks_created
            print(f"   📊 Total chunks for this paper: {total_paper_chunks}")
            print("   " + "="*60)

    print(f"   Created {len(all_chunks)} hybrid chunks:")
    chunk_types = {}
    for chunk in all_chunks:
        ct = chunk['chunk_type']
        chunk_types[ct] = chunk_types.get(ct, 0) + 1
    for ct, count in chunk_types.items():
        print(f"     - {ct}: {count}")

    return all_chunks


def create_chunks_by_size(sentences: List[str], chunk_size: int, overlap: int,
                          metadata: Dict, doc_idx: int, size_category: str,
                          existing_chunks: List) -> List[Dict[str, Any]]:
    """Helper function to create chunks of specific size"""

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'doc_index': doc_idx,
                'chunk_index': len(existing_chunks) + len(chunks),
                'chunk_type': f'{size_category}_chunk',
                'chunk_size_category': size_category,
                'word_count': current_word_count
            })

            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata,
                'chunk_type': f'{size_category}_chunk'
            })

            # Handle overlap
            if overlap > 0:
                overlap_sentences = []
                overlap_words = 0

                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_words + sent_words <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_words += sent_words
                    else:
                        break

                current_chunk = overlap_sentences
                current_word_count = overlap_words
            else:
                current_chunk = []
                current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    # Don't forget last chunk
    if current_chunk and current_word_count >= 20:  # Minimum 20 words
        chunk_text = ' '.join(current_chunk)

        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'doc_index': doc_idx,
            'chunk_index': len(existing_chunks) + len(chunks),
            'chunk_type': f'{size_category}_chunk',
            'chunk_size_category': size_category,
            'word_count': current_word_count
        })

        chunks.append({
            'text': chunk_text,
            'metadata': chunk_metadata,
            'chunk_type': f'{size_category}_chunk'
        })

    return chunks


def create_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for chunks using domain-specific or general models"""

    print("🔢 Generating embeddings...")

    embedding_model_used = EMBEDDING_MODEL
    model = None

    try:
        # Try to import sentence-transformers
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Try to load the preferred model
        try:
            print(f"   Attempting to load: {EMBEDDING_MODEL}")
            model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
            embedding_model_used = EMBEDDING_MODEL
        except Exception as e:
            print(f"   ⚠️  Could not load {EMBEDDING_MODEL}: {e}")

            # Fallback to general purpose model
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"   Falling back to: {fallback_model}")
            model = SentenceTransformer(fallback_model, device=EMBEDDING_DEVICE)
            embedding_model_used = fallback_model

        print(f"   ✅ Using model: {embedding_model_used}")
        print(f"   Device: {EMBEDDING_DEVICE}")

        # Enable FP16 if using CUDA and option is enabled
        if EMBEDDING_DEVICE == "cuda" and EMBEDDING_USE_FP16:
            try:
                import torch
                model = model.half()  # Convert model to FP16
                print(f"   ✅ Enabled FP16 precision (reduces memory by ~50%)")
            except Exception as e:
                print(f"   ⚠️  Could not enable FP16: {e}")

        # Print memory optimization settings
        print(f"   📊 Memory optimizations:")
        print(f"      - Batch size: {EMBEDDING_BATCH_SIZE}")
        print(f"      - Mega-batch size: {EMBEDDING_MEGA_BATCH_SIZE}")
        print(f"      - FP16 precision: {EMBEDDING_USE_FP16 and EMBEDDING_DEVICE == 'cuda'}")
        print(f"      - Model offloading: {EMBEDDING_OFFLOAD_MODEL}")
        print(f"      - Memory monitoring: {EMBEDDING_MONITOR_MEMORY}")

        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]

        # For scientific models, we might need to truncate or handle longer sequences
        max_seq_length = model.max_seq_length if hasattr(model, 'max_seq_length') else 512

        # Truncate texts if necessary
        truncated_texts = []
        for text in texts:
            words = text.split()
            if len(words) > max_seq_length:
                # Truncate to max sequence length
                truncated_text = ' '.join(words[:max_seq_length])
                truncated_texts.append(truncated_text)
            else:
                truncated_texts.append(text)

        # Generate embeddings in mega-batches to avoid memory issues
        print(f"   Processing {len(truncated_texts)} chunks in mega-batches of {EMBEDDING_MEGA_BATCH_SIZE}...")
        print(f"   (Using batch size of {EMBEDDING_BATCH_SIZE} within each mega-batch)")

        total_processed = 0
        embedding_dim = None

        # Process in mega-batches to avoid memory accumulation
        for mega_batch_idx, mega_batch_start in enumerate(range(0, len(truncated_texts), EMBEDDING_MEGA_BATCH_SIZE)):
            mega_batch_end = min(mega_batch_start + EMBEDDING_MEGA_BATCH_SIZE, len(truncated_texts))
            mega_batch_texts = truncated_texts[mega_batch_start:mega_batch_end]

            print(f"   Processing mega-batch {mega_batch_idx + 1}/{(len(truncated_texts)-1)//EMBEDDING_MEGA_BATCH_SIZE + 1} "
                  f"(chunks {mega_batch_start+1}-{mega_batch_end})...")

            # Monitor memory before processing
            if EMBEDDING_DEVICE == "cuda" and EMBEDDING_MONITOR_MEMORY:
                try:
                    import torch
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"   💾 GPU Memory - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
                except:
                    pass

            # Clear any lingering GPU cache before processing
            if EMBEDDING_DEVICE == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Ensure model is on GPU and in correct precision
                    if hasattr(model, 'to'):
                        model = model.to(EMBEDDING_DEVICE)
                        if EMBEDDING_USE_FP16:
                            model = model.half()
                except:
                    pass

            # Generate embeddings for this mega-batch
            mega_batch_embeddings = model.encode(
                mega_batch_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True  # Ensure we get numpy arrays
            )

            # Store embedding dimensions from first batch
            if embedding_dim is None and len(mega_batch_embeddings) > 0:
                embedding_dim = len(mega_batch_embeddings[0])

            # Immediately add embeddings to their corresponding chunks
            for local_idx, global_idx in enumerate(range(mega_batch_start, mega_batch_end)):
                chunks[global_idx]['embedding'] = mega_batch_embeddings[local_idx].tolist()
                chunks[global_idx]['embedding_model'] = embedding_model_used

            total_processed += len(mega_batch_embeddings)

            # Aggressive memory cleanup
            del mega_batch_embeddings
            del mega_batch_texts

            # Force garbage collection after every mega-batch
            import gc
            gc.collect()

            # More aggressive GPU cache clearing and model offloading
            if EMBEDDING_DEVICE == "cuda":
                try:
                    import torch

                    # Offload model to CPU between mega-batches if enabled
                    if EMBEDDING_OFFLOAD_MODEL and mega_batch_idx < ((len(truncated_texts)-1)//EMBEDDING_MEGA_BATCH_SIZE):
                        print(f"   🔄 Offloading model to CPU to free GPU memory...")
                        model = model.cpu()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    else:
                        # Just clear cache without offloading
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Monitor memory after cleanup
                    if EMBEDDING_MONITOR_MEMORY:
                        mem_allocated = torch.cuda.memory_allocated() / 1024**3
                        mem_reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"   💾 After cleanup - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
                except Exception as e:
                    print(f"   ⚠️  Memory cleanup warning: {e}")

        print(f"   Generated {total_processed} embeddings")
        print(f"   Embedding dimensions: {embedding_dim}")
        return chunks

    except ImportError:
        print("❌ sentence-transformers not installed. Please install with:")
        print("   pip install sentence-transformers transformers")
        print("   For SciBERT support also install: pip install torch")
        raise
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        raise


def build_vector_database(chunks: List[Dict[str, Any]], output_dir: Path) -> str:
    """Build vector database from chunks"""
    
    print(f"🗄️  Building {VECTOR_DB_TYPE.upper()} vector database...")
    
    if VECTOR_DB_TYPE.lower() == "chroma":
        return build_chroma_db(chunks, output_dir)
    elif VECTOR_DB_TYPE.lower() == "faiss":
        return build_faiss_db(chunks, output_dir)
    else:
        raise ValueError(f"Unsupported vector database type: {VECTOR_DB_TYPE}")


def build_chroma_db(chunks: List[Dict[str, Any]], output_dir: Path) -> str:
    """Build ChromaDB vector database"""
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create ChromaDB client
        db_path = output_dir / "chroma_db"
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        collection_name = "rag_papers"
        try:
            collection = client.get_collection(collection_name)
            client.delete_collection(collection_name)  # Clear existing
        except:
            pass
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "RAG literature collection"}
        )
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        embeddings = [chunk['embedding'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            collection.add(
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"   ChromaDB created at: {db_path}")
        return str(db_path)
        
    except ImportError:
        print("❌ chromadb not installed. Please install with:")
        print("   pip install chromadb")
        raise


def build_faiss_db(chunks: List[Dict[str, Any]], output_dir: Path) -> str:
    """Build FAISS vector database"""
    
    try:
        import faiss
        import numpy as np
        import pickle
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        # Save FAISS index
        faiss_path = output_dir / "faiss_index.index"
        faiss.write_index(index, str(faiss_path))
        
        # Save metadata separately
        metadata_path = output_dir / "faiss_metadata.pkl"
        metadata = [chunk['metadata'] for chunk in chunks]
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save documents separately
        documents_path = output_dir / "faiss_documents.pkl"
        documents = [chunk['text'] for chunk in chunks]
        with open(documents_path, 'wb') as f:
            pickle.dump(documents, f)
        
        print(f"   FAISS index created at: {faiss_path}")
        return str(faiss_path)
        
    except ImportError:
        print("❌ faiss-cpu not installed. Please install with:")
        print("   pip install faiss-cpu")
        raise


def create_rag_config(output_dir: Path, vector_db_path: str, literature_file: Path,
                     chunks_with_embeddings: List[Dict[str, Any]] = None) -> Path:
    """Create RAG system configuration file"""

    # Determine which embedding model was actually used
    actual_embedding_model = EMBEDDING_MODEL
    if chunks_with_embeddings and len(chunks_with_embeddings) > 0:
        actual_embedding_model = chunks_with_embeddings[0].get('embedding_model', EMBEDDING_MODEL)

    config = {
        "rag_system_info": {
            "created_at": datetime.now().isoformat(),
            "version": "3.1",
            "description": "Multi-stage RAG system with sentence-aware chunking and scientific embeddings"
        },
        "data_sources": {
            "literature_file": literature_file.name,
            "vector_database": vector_db_path,
            "vector_db_type": VECTOR_DB_TYPE
        },
        "text_processing": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "max_chunk_length": MAX_CHUNK_LENGTH,
            "chunking_method": "sentence-aware",
            "text_cleaning": "normalize_whitespace"
        },
        "embeddings": {
            "model_configured": EMBEDDING_MODEL,
            "model_used": actual_embedding_model,
            "device": EMBEDDING_DEVICE,
            "batch_size": EMBEDDING_BATCH_SIZE,
            "embedding_dimensions": len(chunks_with_embeddings[0]['embedding']) if chunks_with_embeddings else None
        },
        "retrieval": {
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "max_results": MAX_RETRIEVAL_RESULTS
        },
        "generation": {
            "model": GENERATION_MODEL,
            "temperature": GENERATION_TEMPERATURE,
            "max_tokens": MAX_GENERATION_TOKENS
        }
    }

    config_file = output_dir / "rag_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    return config_file


def create_usage_example(output_dir: Path) -> Path:
    """Create example usage script"""
    
    example_code = '''#!/usr/bin/env python3
"""
Example RAG System Usage
This script shows how to query the built RAG system.
"""

import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

def load_rag_system(config_path: str):
    """Load the RAG system configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def query_rag(question: str, config_path: str):
    """Query the RAG system"""
    
    # Load configuration
    config = load_rag_system(config_path)
    
    # Load embedding model
    embedding_model = SentenceTransformer(config['embeddings']['model'])
    
    # Load vector database
    if config['data_sources']['vector_db_type'] == 'chroma':
        client = chromadb.PersistentClient(path=config['data_sources']['vector_database'])
        collection = client.get_collection("rag_papers")
        
        # Generate query embedding
        query_embedding = embedding_model.encode([question]).tolist()
        
        # Search for similar documents
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=config['retrieval']['max_results']
        )
        
        # Format results
        retrieved_docs = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            retrieved_docs.append({
                'text': doc,
                'score': results['distances'][0][i],
                'pmid': metadata.get('pmid', ''),
                'title': metadata.get('title', ''),
                'journal': metadata.get('journal', ''),
                'year': metadata.get('year', '')
            })
        
        return retrieved_docs
    
    else:
        raise NotImplementedError(f"Vector DB type {config['data_sources']['vector_db_type']} not implemented in example")

if __name__ == "__main__":
    # Example usage
    question = "How do transcription factors regulate gene expression during development?"
    
    config_path = "rag_config.json"
    results = query_rag(question, config_path)
    
    print(f"Question: {question}")
    print(f"\\nTop {len(results)} relevant papers:")
    
    for i, result in enumerate(results):
        print(f"\\n{i+1}. PMID: {result['pmid']} | {result['title']} ({result['year']})")
        print(f"   Journal: {result['journal']}")
        print(f"   Relevance: {result['score']:.3f}")
        print(f"   Text: {result['text'][:200]}...")
'''
    
    example_file = output_dir / "example_usage.py"
    with open(example_file, 'w') as f:
        f.write(example_code)
    
    return example_file


def build_rag_system(literature_file_path: str):
    """Build complete RAG system from literature file"""
    
    literature_file = Path(literature_file_path)
    if not literature_file.exists():
        print(f"❌ Literature file not found: {literature_file_path}")
        return False
    
    print("🏗️  RAG System - Part 3: RAG Construction")
    print("=" * 60)
    print(f"Literature file: {literature_file.name}")
    
    try:
        # Load literature data
        with open(literature_file, 'r') as f:
            literature_data = json.load(f)
        
        # Create output directory
        project_dir = literature_file.parent
        rag_dir = project_dir / "rag_system"
        rag_dir.mkdir(exist_ok=True)
        
        print(f"📁 RAG output directory: {rag_dir}")
        
        # Step 1: Extract text from papers
        documents = extract_paper_texts(literature_data)
        
        # Step 2: Chunk documents (using selected strategy)
        if CHUNKING_STRATEGY == "small_chunks_only":
            chunks = create_small_chunks_only(documents)
        elif CHUNKING_STRATEGY == "hybrid":
            chunks = create_hybrid_chunks(documents)
        elif CHUNKING_STRATEGY == "semantic_only":
            # Create semantic sections only (no multi-granularity)
            ENABLE_MULTI_GRANULARITY = False
            chunks = create_hybrid_chunks(documents)
        else:
            # Default fallback
            chunks = chunk_documents(documents)
        
        # Step 3: Generate embeddings
        chunks_with_embeddings = create_embeddings(chunks)
        
        # Step 4: Build vector database
        vector_db_path = build_vector_database(chunks_with_embeddings, rag_dir)
        
        # Step 5: Create configuration
        config_file = create_rag_config(rag_dir, vector_db_path, literature_file, chunks_with_embeddings)
        
        # Step 6: Create usage example
        example_file = create_usage_example(rag_dir)
        
        # Save processed chunks for reference
        chunks_file = rag_dir / "processed_chunks.json"
        with open(chunks_file, 'w') as f:
            json.dump(chunks_with_embeddings, f, indent=2)
            
        # Create additional formats for maximum compatibility
        project_name = Path(literature_file_path).stem.replace('_literature', '')
        
        # Hybrid reranking format
        hybrid_reranking_file = create_hybrid_reranking_format(
            chunks_with_embeddings, literature_data, rag_dir, project_name
        )
        
        # Simple JSON formats
        simple_json_file = create_simple_json_format(
            chunks_with_embeddings, literature_data, rag_dir, project_name
        )
        
        # CSV format
        csv_file = create_csv_format(
            chunks_with_embeddings, literature_data, rag_dir, project_name
        )
        
        # Plain text format
        txt_file = create_plaintext_format(
            chunks_with_embeddings, literature_data, rag_dir, project_name
        )
        
        # OpenAI compatible format
        openai_file = create_openai_compatible_format(
            chunks_with_embeddings, literature_data, rag_dir, project_name
        )
        
        # LangChain compatible format
        langchain_file = create_langchain_format(
            chunks_with_embeddings, literature_data, rag_dir, project_name
        )
        
        # LM Studio integration script
        lm_studio_script = create_lm_studio_integration_script(rag_dir, project_name)
        
        print(f"\\n✅ RAG system construction completed!")
        print(f"\\n📁 RAG system files:")
        print(f"   🔧 Configuration: {config_file.name}")
        print(f"   🗄️  Vector database: {Path(vector_db_path).name}")
        print(f"   📄 Processed chunks: {chunks_file.name}")
        print(f"   💡 Usage example: {example_file.name}")
        print(f"")
        print(f"   🔗 Multiple Format Outputs:")
        print(f"   • Hybrid reranking: {Path(hybrid_reranking_file).name}")
        print(f"   • Simple JSON: {Path(simple_json_file).name}")
        print(f"   • Simple JSON (no embeddings): {project_name}_simple_no_embeddings.json")
        print(f"   • CSV format: {Path(csv_file).name}")  
        print(f"   • Plain text: {Path(txt_file).name}")
        print(f"   • OpenAI compatible: {Path(openai_file).name}")
        print(f"   • LangChain compatible: {Path(langchain_file).name}")
        print(f"   • LM Studio integration: {Path(lm_studio_script).name}")
        
        print(f"\\n🚀 Next Steps:")
        print(f"1. Install required packages: pip install sentence-transformers chromadb")
        print(f"2. Test the system: cd {rag_dir} && python {example_file.name}")
        print(f"3. Integrate into your application using the configuration in {config_file.name}")
        
        print(f"\\n📊 RAG System Statistics:")
        metadata = literature_data.get('metadata', {})
        print(f"   📚 Source papers: {metadata.get('total_papers', 0)}")
        print(f"   📄 Processed documents: {len(documents)}")
        print(f"   ✂️  Text chunks: {len(chunks_with_embeddings)}")
        print(f"   🔢 Vector dimensions: {len(chunks_with_embeddings[0]['embedding']) if chunks_with_embeddings else 'N/A'}")
        print(f"   🗄️  Database type: {VECTOR_DB_TYPE}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_hybrid_reranking_format(chunks_with_embeddings: List[Dict[str, Any]], 
                                   literature_data: Dict[str, Any], 
                                   rag_dir: Path, 
                                   project_name: str) -> str:
    """Create hybrid reranking format output compatible with hybrid_retriever.py"""
    
    print("🔗 Creating hybrid reranking format...")
    
    # Build vocabulary from all chunk texts
    vocabulary = {}
    word_freq = {}
    
    # Collect all words and count frequencies
    for chunk in chunks_with_embeddings:
        words = chunk['text'].lower().split()
        for word in words:
            # Simple preprocessing
            word = word.strip('.,!?;:"()[]{}').lower()
            if len(word) > 2:  # Only words longer than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Select top words for vocabulary (similar to TF-IDF approach)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for idx, (word, freq) in enumerate(sorted_words[:500]):  # Top 500 words
        vocabulary[word] = idx
    
    # Create hybrid reranking chunks format
    hybrid_chunks = []
    
    for i, chunk in enumerate(chunks_with_embeddings):
        # Extract metadata
        metadata = chunk['metadata']
        
        # Create search tags from content
        words = chunk['text'].lower().split()
        search_tags = []
        for word in words:
            word = word.strip('.,!?;:"()[]{}').lower()
            if word in vocabulary:
                search_tags.append(word)
        
        # Limit search tags
        search_tags = list(set(search_tags))[:20]  # Top 20 unique tags
        
        chunk_type = chunk.get('chunk_type', 'content')  # Default to 'content' if not specified
        hybrid_chunk = {
            "chunk_id": f"{metadata.get('pmid', 'unknown')}_{chunk_type}_{i}",
            "content": chunk['text'],
            "chunk_type": chunk_type,
            "source_pmid": metadata.get('pmid', ''),
            "source_title": metadata.get('title', ''),
            "source_authors": metadata.get('authors', '').split(', ') if metadata.get('authors') else [],
            "source_journal": metadata.get('journal', ''),
            "source_year": metadata.get('year', ''),
            "keywords": [],  # Could be populated from content analysis
            "wnt_genes_mentioned": [],  # Domain-specific - could be extracted
            "pathway_components": [],  # Domain-specific
            "disease_context": [],  # Could be extracted from content
            "relevance_score": float(metadata.get('relevance_score', 0.0)),
            "word_count": len(chunk['text'].split()),
            "char_count": len(chunk['text']),
            "search_tags": search_tags,
            "composite_score": float(metadata.get('relevance_score', 0.0)),
            "embedding_ready": True,
            "embedding": chunk['embedding']
        }
        
        hybrid_chunks.append(hybrid_chunk)
    
    # Create metadata section
    metadata_section = {
        "total_chunks": len(hybrid_chunks),
        "creation_date": datetime.now().isoformat(),
        "processing_version": "3.0_generalized_rag_pipeline",
        "chunk_types": ["title", "abstract", "content"],
        "original_papers": literature_data.get('metadata', {}).get('total_papers', 0),
        "expanded_papers": 0,
        "duplicates_removed": 0,
        "expansion_method": "direct_literature_fetch",
        "domain_filtering": ["general_biology"],
        "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embeddings_method": "SentenceTransformer",
        "embeddings_dimensions": len(chunks_with_embeddings[0]['embedding']) if chunks_with_embeddings else 384,
        "embeddings_created": datetime.now().isoformat(),
        "total_chunks_with_embeddings": len(hybrid_chunks),
        "vocabulary_size": len(vocabulary)
    }
    
    # Create final hybrid format
    hybrid_format = {
        "metadata": metadata_section,
        "vocabulary": vocabulary,
        "chunks": hybrid_chunks
    }
    
    # Save hybrid reranking format
    hybrid_file = rag_dir / f"{project_name}_hybrid_reranking.json"
    with open(hybrid_file, 'w') as f:
        json.dump(hybrid_format, f, indent=2)
    
    print(f"   💫 Hybrid reranking format: {hybrid_file.name}")
    
    return str(hybrid_file)


def create_simple_json_format(chunks_with_embeddings: List[Dict[str, Any]], 
                             literature_data: Dict[str, Any], 
                             rag_dir: Path, 
                             project_name: str) -> str:
    """Create simple JSON format for easy parsing by any tool"""
    
    print("📄 Creating simple JSON format...")
    
    simple_chunks = []
    for i, chunk in enumerate(chunks_with_embeddings):
        metadata = chunk['metadata']
        simple_chunk = {
            "id": f"chunk_{i}",
            "text": chunk['text'],
            "title": metadata.get('title', ''),
            "authors": metadata.get('authors', ''),
            "journal": metadata.get('journal', ''),
            "year": metadata.get('year', ''),
            "pmid": metadata.get('pmid', ''),
            "relevance_score": float(metadata.get('relevance_score', 0.0)),
            "url": metadata.get('url', ''),
            "word_count": len(chunk['text'].split()),
            "embedding": chunk['embedding']  # Optional - can be removed for smaller files
        }
        simple_chunks.append(simple_chunk)
    
    simple_format = {
        "format": "simple_rag_chunks",
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "total_chunks": len(simple_chunks),
        "total_papers": literature_data.get('metadata', {}).get('total_papers', 0),
        "chunks": simple_chunks
    }
    
    # Save with and without embeddings
    simple_file = rag_dir / f"{project_name}_simple.json"
    with open(simple_file, 'w') as f:
        json.dump(simple_format, f, indent=2)
    
    # Create lightweight version without embeddings
    simple_format_light = simple_format.copy()
    for chunk in simple_format_light['chunks']:
        del chunk['embedding']
    
    simple_file_light = rag_dir / f"{project_name}_simple_no_embeddings.json"
    with open(simple_file_light, 'w') as f:
        json.dump(simple_format_light, f, indent=2)
    
    print(f"   📄 Simple format: {simple_file.name}")
    print(f"   💡 Lightweight format: {simple_file_light.name}")
    
    return str(simple_file)


def create_csv_format(chunks_with_embeddings: List[Dict[str, Any]], 
                     literature_data: Dict[str, Any], 
                     rag_dir: Path, 
                     project_name: str) -> str:
    """Create CSV format for spreadsheet compatibility"""
    
    print("📊 Creating CSV format...")
    
    import csv
    
    csv_file = rag_dir / f"{project_name}_chunks.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'chunk_id', 'text', 'title', 'authors', 'journal', 'year', 'pmid', 
            'relevance_score', 'url', 'word_count'
        ])
        
        # Data rows
        for i, chunk in enumerate(chunks_with_embeddings):
            metadata = chunk['metadata']
            writer.writerow([
                f"chunk_{i}",
                chunk['text'],
                metadata.get('title', ''),
                metadata.get('authors', ''),
                metadata.get('journal', ''),
                metadata.get('year', ''),
                metadata.get('pmid', ''),
                float(metadata.get('relevance_score', 0.0)),
                metadata.get('url', ''),
                len(chunk['text'].split())
            ])
    
    print(f"   📊 CSV format: {csv_file.name}")
    return str(csv_file)


def create_plaintext_format(chunks_with_embeddings: List[Dict[str, Any]], 
                           literature_data: Dict[str, Any], 
                           rag_dir: Path, 
                           project_name: str) -> str:
    """Create plain text format for basic tools and manual inspection"""
    
    print("📝 Creating plain text format...")
    
    txt_file = rag_dir / f"{project_name}_chunks.txt"
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"RAG CHUNKS - {project_name.upper()}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total chunks: {len(chunks_with_embeddings)}\n")
        f.write(f"Total papers: {literature_data.get('metadata', {}).get('total_papers', 0)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, chunk in enumerate(chunks_with_embeddings):
            metadata = chunk['metadata']
            
            f.write(f"CHUNK {i+1}\n")
            f.write("-" * 40 + "\n")
            f.write(f"ID: chunk_{i}\n")
            f.write(f"Title: {metadata.get('title', 'N/A')}\n")
            f.write(f"Authors: {metadata.get('authors', 'N/A')}\n")
            f.write(f"Journal: {metadata.get('journal', 'N/A')}\n")
            f.write(f"Year: {metadata.get('year', 'N/A')}\n")
            f.write(f"PMID: {metadata.get('pmid', 'N/A')}\n")
            f.write(f"Relevance: {metadata.get('relevance_score', 0.0):.3f}\n")
            f.write(f"Words: {len(chunk['text'].split())}\n")
            f.write(f"URL: {metadata.get('url', 'N/A')}\n")
            f.write("\nCONTENT:\n")
            f.write(chunk['text'])
            f.write("\n\n" + "=" * 80 + "\n\n")
    
    print(f"   📝 Plain text format: {txt_file.name}")
    return str(txt_file)


def create_openai_compatible_format(chunks_with_embeddings: List[Dict[str, Any]], 
                                  literature_data: Dict[str, Any], 
                                  rag_dir: Path, 
                                  project_name: str) -> str:
    """Create OpenAI/GPT-compatible format for easy integration"""
    
    print("🤖 Creating OpenAI-compatible format...")
    
    # Format compatible with OpenAI's retrieval systems
    openai_chunks = []
    for i, chunk in enumerate(chunks_with_embeddings):
        metadata = chunk['metadata']
        
        # OpenAI-style document format
        openai_chunk = {
            "id": f"{project_name}_chunk_{i}",
            "object": "document",
            "content": chunk['text'],
            "metadata": {
                "title": metadata.get('title', ''),
                "authors": metadata.get('authors', ''),
                "journal": metadata.get('journal', ''),
                "year": metadata.get('year', ''),
                "pmid": metadata.get('pmid', ''),
                "relevance_score": float(metadata.get('relevance_score', 0.0)),
                "url": metadata.get('url', ''),
                "source": "pubmed_literature"
            },
            "embedding": chunk['embedding']
        }
        openai_chunks.append(openai_chunk)
    
    openai_format = {
        "object": "document_collection",
        "data": openai_chunks,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "usage": {
            "total_documents": len(openai_chunks),
            "total_tokens": sum(len(chunk['content'].split()) for chunk in openai_chunks)
        }
    }
    
    openai_file = rag_dir / f"{project_name}_openai_format.json"
    with open(openai_file, 'w') as f:
        json.dump(openai_format, f, indent=2)
    
    print(f"   🤖 OpenAI format: {openai_file.name}")
    return str(openai_file)


def create_langchain_format(chunks_with_embeddings: List[Dict[str, Any]], 
                          literature_data: Dict[str, Any], 
                          rag_dir: Path, 
                          project_name: str) -> str:
    """Create LangChain-compatible format"""
    
    print("🦜 Creating LangChain-compatible format...")
    
    langchain_docs = []
    for i, chunk in enumerate(chunks_with_embeddings):
        metadata = chunk['metadata']
        
        # LangChain Document format
        langchain_doc = {
            "page_content": chunk['text'],
            "metadata": {
                "source": f"pmid_{metadata.get('pmid', 'unknown')}",
                "title": metadata.get('title', ''),
                "authors": metadata.get('authors', ''),
                "journal": metadata.get('journal', ''),
                "year": metadata.get('year', ''),
                "pmid": metadata.get('pmid', ''),
                "relevance_score": float(metadata.get('relevance_score', 0.0)),
                "url": metadata.get('url', ''),
                "chunk_id": f"chunk_{i}",
                "chunk_index": i
            }
        }
        langchain_docs.append(langchain_doc)
    
    langchain_format = {
        "format": "langchain_documents",
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "documents": langchain_docs,
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": len(chunks_with_embeddings[0]['embedding']) if chunks_with_embeddings else 384,
            "vectors": [chunk['embedding'] for chunk in chunks_with_embeddings]
        }
    }
    
    langchain_file = rag_dir / f"{project_name}_langchain_format.json"
    with open(langchain_file, 'w') as f:
        json.dump(langchain_format, f, indent=2)
    
    print(f"   🦜 LangChain format: {langchain_file.name}")
    return str(langchain_file)


def create_lm_studio_integration_script(rag_dir: Path, project_name: str) -> str:
    """Create ready-to-use integration script for LM Studio"""
    
    print("🎬 Creating LM Studio integration script...")
    
    script_content = f'''#!/usr/bin/env python3
"""
LM Studio RAG Integration Script
Ready-to-use script for integrating RAG with LM Studio API
"""

import json
import requests
from typing import List, Dict, Any

class LMStudioRAG:
    """Simple RAG integration for LM Studio"""
    
    def __init__(self, chunks_file: str = "{project_name}_simple_no_embeddings.json", 
                 lm_studio_url: str = "http://localhost:1234/v1/chat/completions"):
        self.lm_studio_url = lm_studio_url
        self.chunks = self.load_chunks(chunks_file)
        
    def load_chunks(self, chunks_file: str) -> List[Dict]:
        """Load RAG chunks from JSON file"""
        with open(chunks_file, 'r') as f:
            data = json.load(f)
        return data['chunks']
    
    def search_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based search (can be enhanced with embeddings)"""
        query_words = set(query.lower().split())
        
        # Score chunks by keyword overlap
        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(chunk['text'].lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def create_context(self, chunks: List[Dict]) -> str:
        """Create context string from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"Source {{i}} ({{chunk['title'][:50]}}...):\\n{{chunk['text']}}\\n")
        return "\\n".join(context_parts)
    
    def query_lm_studio(self, prompt: str, model: str = "local-model") -> str:
        """Query LM Studio API"""
        payload = {{
            "model": model,
            "messages": [
                {{"role": "user", "content": prompt}}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }}
        
        try:
            response = requests.post(self.lm_studio_url, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error querying LM Studio: {{e}}"
    
    def rag_query(self, question: str, top_k: int = 3) -> str:
        """Complete RAG pipeline: retrieve + generate"""
        
        # Step 1: Retrieve relevant chunks
        relevant_chunks = self.search_chunks(question, top_k)
        
        if not relevant_chunks:
            return self.query_lm_studio(f"Question: {{question}}")
        
        # Step 2: Create context
        context = self.create_context(relevant_chunks)
        
        # Step 3: Create prompt with context
        prompt = f"""Context from research literature:
{{context}}

Question: {{question}}

Please answer the question based on the provided context. If the context doesn't contain enough information, please say so."""
        
        # Step 4: Query LM Studio
        return self.query_lm_studio(prompt)

# Usage Example
if __name__ == "__main__":
    # Initialize RAG system
    rag = LMStudioRAG()
    
    # Example queries
    test_questions = [
        "How do transcription factors regulate gene expression during development?",
        "What role does epigenetic regulation play in evolution?",
        "What are the mechanisms of gene regulatory networks?"
    ]
    
    print("🎬 LM Studio RAG Integration Test")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\\n❓ Question: {{question}}")
        print("🔍 Searching for relevant chunks...")
        
        # Just show retrieval for demo (without calling LM Studio)
        rag_instance = LMStudioRAG()
        relevant_chunks = rag_instance.search_chunks(question, top_k=2)
        
        if relevant_chunks:
            print(f"✅ Found {{len(relevant_chunks)}} relevant chunks:")
            for i, chunk in enumerate(relevant_chunks, 1):
                print(f"   {{i}}. {{chunk['title'][:60]}}... ({{chunk['year']}})")
        else:
            print("❌ No relevant chunks found")
            
        # Uncomment to actually query LM Studio:
        # answer = rag.rag_query(question)
        # print(f"🤖 Answer: {{answer}}")
        print("-" * 50)
'''
    
    script_file = rag_dir / f"{project_name}_lm_studio_integration.py"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    print(f"   🎬 LM Studio script: {script_file.name}")
    return str(script_file)


def main():
    """Main function"""
    
    if len(sys.argv) != 2:
        print("Usage: python rag_part3_build_rag.py <literature_file.json>")
        print("Example: python rag_part3_build_rag.py gene_reg_evolution_literature.json")
        return 1
    
    literature_file = sys.argv[1]
    
    # Build RAG system
    success = build_rag_system(literature_file)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

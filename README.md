# Generalized RAG Pipeline for Scientific Literature

A comprehensive, modular system for creating domain-specific Retrieval-Augmented Generation (RAG) systems from scientific literature. Build powerful RAG systems for any research domain using PubMed, LLMs, and vector databases.

## Overview

This pipeline provides a **3-part modular workflow** that separates query generation, literature fetching, and RAG construction into independent, editable stages. The system is domain-agnostic and can be configured for any scientific field including biology, medicine, computer science, and more.

### Key Features

- **Modular 3-Stage Pipeline**: Generate queries → Fetch literature → Build RAG system
- **Domain Agnostic**: Works with any scientific domain via LLM-powered query generation
- **Manual Control**: Edit and refine queries, filters, and configurations between stages
- **Multiple Output Formats**: 8+ formats including OpenAI, LangChain, ChromaDB, FAISS
- **Advanced Retrieval**: Hybrid search combining BM25 and semantic embeddings
- **GPU Optimized**: FP16 support, batch processing, and memory management
- **Comprehensive Logging**: Complete debugging URLs, violation detection, and summaries

## Architecture

```
Stage 1: Query Generation          Stage 2: Literature Fetching      Stage 3: RAG Construction
        ↓                                    ↓                                 ↓
    LLM-based                           PubMed API                      Text Chunking
  query generation                    + filtering                       + Embeddings
        ↓                                    ↓                                 ↓
  query.json (editable!)             literature.json                    Vector Database
                                                                         (ChromaDB/FAISS)
```

## Prerequisites

- Python 3.8 or higher
- GPU recommended but not required (CPU mode available)
- API keys for:
  - DeepInfra (for LLM query generation) - Required
  - NCBI (for PubMed rate limits) - Optional but recommended
  - OpenAI or Anthropic - Optional

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/thirtysix/generalized_rag_pipeline_semantic_sections.git
cd generalized_rag_pipeline_semantic_sections
```

### 2. Create Virtual Environment

It's **strongly recommended** to use a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows
```

You'll know the virtual environment is activated when you see `(venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Core dependencies** (automatically installed from requirements.txt):

**Essential Packages:**
- `sentence-transformers>=2.2.0` - Embedding generation for RAG
- `transformers>=4.30.0` - Transformer models
- `torch>=2.0.0` - PyTorch (required by sentence-transformers)
- `chromadb>=0.4.0` - Vector database (default)
- `requests>=2.31.0` - HTTP requests for PubMed API
- `beautifulsoup4>=4.12.0` - HTML parsing for literature
- `python-dotenv>=1.0.0` - Environment variable management

**Optional but Recommended:**
- `faiss-cpu>=1.7.4` - Alternative vector database (faster for large datasets)
- `rank-bm25>=0.2.2` - Hybrid retrieval support
- `nltk>=3.8` - Text processing utilities

**For GPU Support (Recommended for faster embeddings):**
```bash
# If you have CUDA-capable GPU, install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Minimal Installation** (if you want to test without all dependencies):
```bash
pip install sentence-transformers chromadb requests beautifulsoup4 python-dotenv
```

### 4. Configure Environment Variables

Copy the sample environment file and add your API keys:

```bash
cp .env.sample .env
nano .env  # Edit with your API keys
```

Required variables:
```bash
DEEPINFRA_API_KEY=your_deepinfra_api_key_here  # Required for query generation
EMAIL=your.email@example.com                    # Required for PubMed API
NCBI_API_KEY=your_ncbi_api_key_here            # Optional, for higher rate limits
```

### 5. Verify Installation

Test that everything is installed correctly:

```bash
# Test Python imports
python -c "import sentence_transformers, chromadb, requests; print('All core packages imported successfully!')"

# Check if GPU is available (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If you see "All core packages imported successfully!" you're ready to go!

## Quick Start

### Complete Workflow Example

```bash
# Step 1: Generate queries using LLM
python rag_part1_generate_queries.py
# → Creates: queries/dyrk1b_queries_20251104_123456.json

# Step 2: (Optional) Manually edit the query file to refine search terms
nano queries/dyrk1b_queries_20251104_123456.json

# Step 3: Fetch literature from PubMed
python rag_part2_fetch_literature.py queries/dyrk1b_queries_20251104_123456.json
# → Creates: results/dyrk1b_20251104_123456/dyrk1b_literature.json

# Step 4: Build RAG system with embeddings
python rag_part3_build_rag_optimized.py results/dyrk1b_20251104_123456/dyrk1b_literature.json
# → Creates: results/dyrk1b_20251104_123456/rag_system/

# Step 5: Use your RAG system
cd results/dyrk1b_20251104_123456/rag_system/
python example_usage.py
```

## Detailed Usage

### Part 1: Query Generation

**Script**: `rag_part1_generate_queries.py`

Generates comprehensive search queries using an LLM, organized into categories:
- Core terms (main concepts)
- Entity terms (specific proteins, genes, diseases)
- Method terms (experimental techniques)
- Context terms (broader research areas)
- Synonyms (alternative terminology)
- Exclude terms (filter out unwanted papers)

**Configuration** (edit in script):
```python
PROJECT_NAME = "dyrk1b"
RESEARCH_TOPIC = "DYRK1B protein kinase function and regulation"
RESEARCH_DOMAIN = "biology"
INCLUDE_TERMS = ["kinase activity", "signal transduction"]
EXCLUDE_TERMS = ["clinical trials", "patents"]
LLM_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
```

**Output**:
- Timestamped query JSON file in `queries/` directory
- Raw LLM response saved to `data/` for debugging

**Manual Editing**: You can manually edit the generated JSON file to add/remove search terms before proceeding to Part 2.

### Part 2: Literature Fetching

**Script**: `rag_part2_fetch_literature.py`

Fetches papers from PubMed using the generated queries, with intelligent relevance scoring and filtering.

**Usage**:
```bash
python rag_part2_fetch_literature.py queries/your_query_file.json
```

**Configuration** (edit in script):
```python
MAX_TOTAL_PAPERS = 15000              # Total paper limit
MAX_PAPERS_PER_QUERY = 3000           # Per-category limit
MIN_RELEVANCE_THRESHOLD = 0.02        # Minimum relevance score
INCLUDE_RECENT_YEARS = 25             # Publication date range
USE_NOT_OPERATOR = False              # PubMed NOT filtering
```

**Key Features**:
- Relevance scoring based on term frequency
- Per-category paper limits to prevent single-category domination
- Exclude term violation detection
- Complete debugging URLs for manual PubMed query testing
- Timestamped project directories

**Output**:
- `results/{project}_{timestamp}/` directory containing:
  - `{project}_literature.json` - Complete literature database
  - `literature_fetch_summary.txt` - Detailed summary with URLs
  - `{project}_queries.json` - Copy of query configuration
  - Scripts and metadata for reproducibility

### Part 3: RAG System Construction

**Script**: `rag_part3_build_rag_optimized.py`

Builds a complete RAG system with embeddings and vector database. Optimized for GPU with FP16 support and memory management.

**Usage**:
```bash
python rag_part3_build_rag_optimized.py results/project_timestamp/project_literature.json
```

**Configuration** (edit in script):
```python
# Chunking strategy
CHUNKING_STRATEGY = "small_chunks_only"  # Options: "hybrid", "small_chunks_only", "semantic_only"

# Multi-granularity chunk sizes
SMALL_CHUNK_SIZE = 100              # ~2-3 sentences for fine-grained matching
SMALL_CHUNK_OVERLAP = 25
MEDIUM_CHUNK_SIZE = 250             # ~1 paragraph or section
MEDIUM_CHUNK_OVERLAP = 50
CHUNK_SIZE = 512                    # Large chunks (full abstract)
CHUNK_OVERLAP = 50

# Text content settings
INCLUDE_TITLE_IN_CHUNKS = False     # Include title in chunk text

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Default: high quality
# Alternatives: "allenai/scibert_scivocab_uncased" (scientific), "dmis-lab/biobert-base-cased-v1.2" (biomedical)

# Vector database
VECTOR_DB_TYPE = "chroma"           # Options: "chroma", "faiss", "pinecone"

# GPU optimization settings
EMBEDDING_DEVICE = "cuda"           # "cpu" or "cuda" for GPU
EMBEDDING_BATCH_SIZE = 24           # Batch size per embedding call
EMBEDDING_MEGA_BATCH_SIZE = 300     # Process in mega-batches to avoid memory issues
EMBEDDING_USE_FP16 = True           # Half-precision (FP16) for 2x memory savings
EMBEDDING_OFFLOAD_MODEL = True      # Offload model to CPU between mega-batches
EMBEDDING_MONITOR_MEMORY = True     # Monitor GPU memory usage
```

**Chunking Strategies**:
- `small_chunks_only`: Small chunks (100 tokens) for maximum precision (default)
- `hybrid`: Multi-granularity - small (100), medium (250), large (512) chunks
- `semantic_only`: Sentence-based semantic segmentation

**Embedding Models**:
- `sentence-transformers/all-mpnet-base-v2` - High quality general purpose (768 dims, default)
- `allenai/scibert_scivocab_uncased` - Scientific literature optimized (768 dims)
- `dmis-lab/biobert-base-cased-v1.2` - Biomedical/clinical papers (768 dims)
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, smaller model (384 dims)

**GPU Memory Optimization**:
- **FP16 Mode**: Reduces memory by ~50% with minimal accuracy loss
- **Mega-batching**: Processes 300 chunks at a time to prevent OOM errors
- **Model Offloading**: Moves model to CPU between batches to free GPU memory
- **Memory Monitoring**: Tracks GPU usage throughout processing

**Output Formats** (8+ formats in `rag_system/` directory):
1. `rag_config.json` - System configuration
2. `chroma_db/` or `faiss_index/` - Vector database
3. `processed_chunks.json` - All chunks with full embeddings
4. `*_simple_no_embeddings.json` - Lightweight format without embeddings
5. `*_chunks.csv` - Spreadsheet format
6. `*_chunks.txt` - Plain text format
7. `*_openai_format.json` - OpenAI API compatible
8. `*_langchain_format.json` - LangChain compatible
9. `*_hybrid_reranking.json` - Advanced retrieval with BM25
10. `example_usage.py` - Working example script
11. `*_lm_studio_integration.py` - LM Studio integration

## Directory Structure

```
generalized_rag_pipeline_semantic_sections/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .env.sample                           # Environment template
├── .gitignore                            # Git exclusions
├── LICENSE                               # MIT License
│
├── rag_part1_generate_queries.py         # Part 1: Query generation
├── rag_part2_fetch_literature.py         # Part 2: Literature fetching
├── rag_part2_fetch_literature_mesh.py    # Alternative: MeSH-based fetching
├── rag_part3_build_rag_optimized.py      # Part 3: RAG construction (optimized)
├── run_with_env.py                       # Environment loader utility
├── run_hybrid_rag.sh                     # Shell script wrapper
│
├── core/                                 # Core modules
│   ├── query_generator.py                # LLM query generation
│   └── literature_fetcher.py             # PubMed API interaction
│
├── config/                               # Configuration
│   └── pipeline_config.py                # Dataclass-based configs
│
├── utils/                                # Utilities
│   └── logger.py                         # Logging utilities
│
├── queries/                              # Generated query files
│   ├── .gitkeep
│   └── *.json                           # Query configurations
│
├── data/                                 # Generated data (gitignored)
│   └── llm_raw_response_*.txt           # Raw LLM responses
│
├── results/                              # Generated RAG systems (gitignored)
│   └── {project}_{timestamp}/
│       ├── {project}_literature.json
│       ├── literature_fetch_summary.txt
│       └── rag_system/
│           ├── chroma_db/ or faiss_index/
│           ├── *.json (multiple formats)
│           └── example_usage.py
│
├── cache/                                # Vector store cache (gitignored)
├── temp/                                 # Temporary files (gitignored)
│
├── old/                                  # Archived development files
│   ├── README_original.md
│   ├── create_my_rag*.py                # Legacy single scripts
│   └── test_*.py                        # Development utilities
│
└── venv/                                 # Virtual environment (gitignored)
```

## Configuration System

The pipeline uses a comprehensive configuration system (`config/pipeline_config.py`) with dataclasses for:

- **LiteratureFetchConfig**: PubMed API settings, rate limits, filtering
- **ProcessingConfig**: Text chunking, preprocessing options
- **EmbeddingConfig**: Model selection, batch sizes, device settings
- **VectorStoreConfig**: Database type, persistence, collection settings
- **PipelineConfig**: Master configuration combining all components

Domain-specific presets available for biology, medicine, and AI research.

## Advanced Features

### Hybrid Retrieval

Combines keyword-based (BM25) and semantic search for optimal results:
```python
# Available in hybrid_reranking.json output format
hybrid_results = retriever.search(
    query="How do checkpoint inhibitors work?",
    semantic_weight=0.7,
    bm25_weight=0.3
)
```

### GPU Optimization

- **FP16 Mode**: 2x memory savings with minimal accuracy loss
- **Batch Processing**: Configurable batch sizes for large datasets
- **Memory Monitoring**: Automatic tracking and warnings
- **Model Offloading**: CPU offload support for large models

### Quality Assurance

- **Relevance Scoring**: Per-paper relevance based on query term frequency
- **Violation Detection**: Flags papers containing excluded terms
- **Complete URLs**: Browser-testable PubMed queries for debugging
- **Summary Reports**: Detailed statistics and configuration logs

## Output Formats

### For RAG Applications
- **ChromaDB/FAISS**: Production vector databases
- **OpenAI Format**: Compatible with OpenAI's retrieval APIs
- **LangChain Format**: Ready for LangChain integration

### For Analysis
- **CSV**: Load into Excel/Pandas for analysis
- **TXT**: Plain text for manual review
- **JSON (no embeddings)**: Lightweight format for inspection

### For Integration
- **LM Studio**: Direct integration script
- **Hybrid Reranking**: Advanced retrieval with BM25 + semantic

## Troubleshooting

### No Papers Found

**Symptoms**: PubMed queries return 0 results

**Solutions**:
1. Check the detailed summary URLs - test manually in browser
2. Reduce filtering: Set `USE_NOT_OPERATOR = False`
3. Increase `MAX_PAPERS_PER_QUERY` to 5000
4. Broaden search terms in query JSON file

### Clinical/Irrelevant Papers Getting Through

**Symptoms**: Exclude term violations in summary report

**Solutions**:
1. Review violation report in `literature_fetch_summary.txt`
2. Add more specific exclude terms to query JSON
3. Use positive filtering (include terms) instead of NOT operator
4. Manually edit query JSON between Part 1 and Part 2

### GPU Memory Errors

**Symptoms**: CUDA out of memory errors during embedding

**Solutions**:
1. Reduce `BATCH_SIZE` (try 16, 8, or 4)
2. Enable `USE_FP16 = True` for 2x memory savings
3. Use smaller embedding model (all-MiniLM-L6-v2)
4. Switch to CPU mode: Set device to "cpu" in script

### Import Errors

**Symptoms**: Module not found errors

**Solutions**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# For GPU support, ensure CUDA is installed
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Slow Embedding Generation

**Symptoms**: Part 3 takes hours to complete

**Solutions**:
1. Use GPU instead of CPU (20-50x faster)
2. Increase `BATCH_SIZE` if GPU memory allows
3. Use faster model like all-MiniLM-L6-v2
4. Reduce number of papers fetched in Part 2

## Use Cases

### Biomedical Research
```python
# Configure for biomedical literature
RESEARCH_DOMAIN = "biomedicine"
EMBEDDING_MODEL = "dmis-lab/biobert-v1.1"
INCLUDE_RECENT_YEARS = 10  # Recent papers only
```

### Literature Review
```python
# Broad literature survey
MAX_TOTAL_PAPERS = 5000
CHUNKING_STRATEGY = "hybrid"  # Balance precision and context
```

### Hypothesis Generation
```python
# Find novel connections
ENABLE_CITATION_EXPANSION = True
USE_HIGH_IMPACT_FILTERING = True
```

## Performance Considerations

### Dataset Size vs. Quality
- **Small (100-500 papers)**: High precision, focused domain
- **Medium (500-5000 papers)**: Balanced coverage
- **Large (5000+ papers)**: Comprehensive but slower

### Embedding Time Estimates
- **CPU**: ~1-2 papers/second
- **GPU (consumer)**: ~20-50 papers/second
- **GPU (datacenter)**: ~100+ papers/second

### Storage Requirements
- **Literature JSON**: ~2-5 KB per paper
- **Vector DB (ChromaDB)**: ~10-20 KB per chunk
- **All Formats**: ~50-100 MB per 1000 papers

## Contributing

Contributions are welcome! Areas for improvement:
1. Additional embedding models
2. More vector database backends
3. Enhanced query generation prompts
4. Domain-specific configurations
5. Performance optimizations

## License

MIT License - see LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{generalized_rag_pipeline,
  title={Generalized RAG Pipeline for Scientific Literature},
  author={Harlan Barker},
  year={2025},
  url={https://github.com/thirtysix/scientific_topic_RAG}
}
```

## Acknowledgments

Built with:
- [Sentence Transformers](https://www.sbert.net/) - Embedding generation
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) - Literature API
- [DeepInfra](https://deepinfra.com/) - LLM API

---

**Questions or issues?** Please open an issue on GitHub or contact the maintainers.

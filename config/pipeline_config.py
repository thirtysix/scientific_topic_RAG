"""
Generalized RAG Pipeline Configuration

Defines configuration classes and settings for the RAG pipeline components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
from pathlib import Path


@dataclass
class QueryGenerationConfig:
    """Configuration for LLM-based query generation"""
    api_provider: str = "deepinfra"  # deepinfra, openai, anthropic
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 120


@dataclass
class LiteratureFetchConfig:
    """Configuration for literature fetching from PubMed"""
    email: str = "rag-pipeline@example.com"
    tool_name: str = "GeneralizedRAG"
    api_key: Optional[str] = None
    rate_limit_delay: float = 0.34  # ~3 requests/second
    max_per_query: int = 500
    max_total_papers: int = 5000
    min_relevance_score: float = 0.0
    date_range_years: Optional[int] = None
    include_preprints: bool = False
    require_abstract: bool = True
    timeout: int = 30
    
    # Publication filtering options
    exclude_clinical_trials: bool = True
    exclude_case_reports: bool = True
    exclude_editorials: bool = True
    exclude_letters: bool = True
    exclude_news: bool = True
    focus_basic_research: bool = True
    languages: List[str] = field(default_factory=lambda: ["English"])
    
    # MeSH term filtering
    use_mesh_terms: bool = False
    exclude_clinical_mesh: bool = False
    basic_research_mesh: List[str] = field(default_factory=lambda: [
        "Gene Expression Regulation",
        "Transcription Factors", 
        "Epigenesis, Genetic",
        "Gene Regulatory Networks"
    ])
    
    # Journal filtering (optional)
    exclude_clinical_journals: bool = False
    preferred_journals: List[str] = field(default_factory=list)


@dataclass
class ProcessingConfig:
    """Configuration for text processing and chunking"""
    chunk_size: int = 400
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 1000
    chunk_strategies: List[str] = field(default_factory=lambda: ["abstract", "title", "combined"])
    extract_entities: bool = True
    entity_types: List[str] = field(default_factory=lambda: ["genes", "proteins", "diseases", "drugs"])
    filter_low_quality: bool = True
    quality_threshold: float = 0.5


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "all-MiniLM-L6-v2"
    model_type: str = "sentence-transformers"  # sentence-transformers, openai, custom
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class VectorStoreConfig:
    """Configuration for vector database"""
    backend: str = "chromadb"  # chromadb, faiss, pinecone
    persist_directory: Optional[str] = None
    collection_name: str = "rag_collection"
    distance_metric: str = "cosine"  # cosine, euclidean, ip
    index_params: Dict[str, Any] = field(default_factory=dict)
    
    # Backend-specific configs
    chromadb_settings: Dict[str, Any] = field(default_factory=dict)
    pinecone_settings: Dict[str, Any] = field(default_factory=dict)
    faiss_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpansionConfig:
    """Configuration for corpus expansion"""
    enable_citation_expansion: bool = True
    citation_hops: int = 2
    max_citations_per_paper: int = 100
    citation_score_threshold: float = 0.1
    enable_high_impact_extraction: bool = True
    min_citation_count: int = 10
    min_impact_factor: float = 2.0
    extract_discussions: bool = True
    discussion_min_length: int = 100


@dataclass
class RetrievalConfig:
    """Configuration for retrieval and reranking"""
    enable_hybrid_retrieval: bool = True
    bm25_weight: float = 0.5
    semantic_weight: float = 0.5
    top_k_candidates: int = 100
    final_top_k: int = 10
    reranking_strategy: str = "hybrid"  # bm25, semantic, hybrid, neural
    reranker_model: Optional[str] = None
    enable_query_expansion: bool = True
    expansion_terms: int = 5


@dataclass
class ValidationConfig:
    """Configuration for validation and quality checks"""
    validate_inputs: bool = True
    validate_outputs: bool = True
    min_corpus_size: int = 100
    max_corpus_size: int = 100000
    quality_metrics: List[str] = field(default_factory=lambda: [
        "coverage", "relevance", "diversity", "quality"
    ])
    benchmark_queries: Optional[List[str]] = None


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_progress_bars: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all components"""
    
    # Component configs
    query_generation: QueryGenerationConfig = field(default_factory=QueryGenerationConfig)
    literature_fetch: LiteratureFetchConfig = field(default_factory=LiteratureFetchConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Pipeline settings
    pipeline_name: str = "generalized_rag"
    output_directory: str = "data"
    cache_directory: str = "cache"
    temp_directory: str = "temp"
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Environment settings
    use_gpu: bool = True
    gpu_memory_limit: Optional[float] = None
    random_seed: int = 42
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Set up directories
        for dir_name in [self.output_directory, self.cache_directory, self.temp_directory]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        # Load API keys from environment if not provided
        if not self.query_generation.api_key:
            self.query_generation.api_key = os.getenv('LLM_API_KEY')
        
        if not self.literature_fetch.api_key:
            self.literature_fetch.api_key = os.getenv('NCBI_API_KEY')
        
        # Set default persist directory for vector store
        if not self.vector_store.persist_directory:
            self.vector_store.persist_directory = f"{self.cache_directory}/vector_store"
        
        # Set default log file
        if not self.logging.log_file:
            self.logging.log_file = f"{self.output_directory}/pipeline.log"
    
    @classmethod
    def for_domain(cls, domain: str, **overrides) -> 'PipelineConfig':
        """Create domain-specific configuration"""
        config = cls()
        
        # Apply domain-specific defaults
        domain_configs = {
            'biology': {
                'embedding.model_name': 'dmis-lab/biobert-base-cased-v1.2',
                'processing.entity_types': ['genes', 'proteins', 'pathways', 'organisms'],
                'expansion.min_citation_count': 5,
            },
            'medicine': {
                'embedding.model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                'processing.entity_types': ['diseases', 'drugs', 'treatments', 'symptoms'],
                'expansion.min_impact_factor': 3.0,
            },
            'ai': {
                'embedding.model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'processing.entity_types': ['algorithms', 'methods', 'datasets', 'models'],
                'expansion.enable_citation_expansion': False,
            },
        }
        
        # Apply domain defaults
        if domain.lower() in domain_configs:
            domain_settings = domain_configs[domain.lower()]
            for key, value in domain_settings.items():
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        
        # Apply user overrides
        for key, value in overrides.items():
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        def _asdict_recursive(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    result[key] = _asdict_recursive(value)
                return result
            elif isinstance(obj, (list, tuple)):
                return [_asdict_recursive(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: _asdict_recursive(value) for key, value in obj.items()}
            else:
                return obj
        
        return _asdict_recursive(self)
    
    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct nested dataclass objects
        # This is simplified - in practice you'd want more robust deserialization
        config = cls()
        # Apply loaded data (implementation would be more complex)
        return config


# Default configurations for common domains
DEFAULT_BIOLOGY_CONFIG = PipelineConfig.for_domain('biology')
DEFAULT_MEDICINE_CONFIG = PipelineConfig.for_domain('medicine')
DEFAULT_AI_CONFIG = PipelineConfig.for_domain('ai')

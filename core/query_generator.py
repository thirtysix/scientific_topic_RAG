"""
LLM-based Query Generator for RAG Systems

Generates comprehensive search terms for filtering scientific literature using LLM assistance.
Adapted from the original generate_query.py with enhanced flexibility and domain agnosticism.
"""

import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

try:
    from ..config.pipeline_config import QueryGenerationConfig
    from ..utils.logger import get_logger
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.pipeline_config import QueryGenerationConfig
    from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Container for generated query results"""
    topic: str
    search_config: Dict[str, Any]
    metadata: Dict[str, Any]
    raw_response: str
    timestamp: str


class QueryGenerator:
    """Generate domain-agnostic search query configurations using LLM assistance"""
    
    def __init__(self, config: Optional[QueryGenerationConfig] = None):
        """
        Initialize the query generator
        
        Args:
            config: Query generation configuration
        """
        self.config = config or QueryGenerationConfig()
        self._setup_client()
        
    def _setup_client(self):
        """Setup API client based on configuration"""
        if self.config.api_provider == "deepinfra":
            self.base_url = "https://api.deepinfra.com/v1/openai/chat/completions"
        elif self.config.api_provider == "openai":
            self.base_url = "https://api.openai.com/v1/chat/completions"
        elif self.config.api_provider == "anthropic":
            # Would need different client setup for Anthropic
            raise NotImplementedError("Anthropic integration not yet implemented")
        else:
            raise ValueError(f"Unknown API provider: {self.config.api_provider}")
    
    def create_domain_agnostic_prompt(self, 
                                    topic: str, 
                                    domain: Optional[str] = None,
                                    include_terms: Optional[List[str]] = None, 
                                    exclude_terms: Optional[List[str]] = None,
                                    custom_instructions: Optional[str] = None) -> str:
        """Create a domain-agnostic prompt for the LLM"""
        
        include_terms = include_terms or []
        exclude_terms = exclude_terms or []
        
        # Domain-specific guidance
        domain_guidance = ""
        if domain:
            domain_templates = {
                "biology": "Focus on biological processes, molecular mechanisms, genes, proteins, pathways, and cellular components.",
                "medicine": "Focus on diseases, treatments, drugs, clinical applications, symptoms, and medical procedures.", 
                "ai": "Focus on algorithms, methods, models, datasets, frameworks, and computational approaches.",
                "physics": "Focus on physical phenomena, theories, equations, experiments, and fundamental concepts.",
                "chemistry": "Focus on chemical compounds, reactions, synthesis, analysis, and molecular structures.",
                "engineering": "Focus on systems, designs, methods, technologies, applications, and performance metrics."
            }
            domain_guidance = domain_templates.get(domain.lower(), 
                f"Focus on core concepts, methods, and applications relevant to {domain}.")
        
        prompt = f"""You are generating comprehensive search terms to filter scientific literature for RAG (Retrieval-Augmented Generation) applications.

TOPIC: {topic}
{f"DOMAIN: {domain}" if domain else ""}

USER PROVIDED TERMS:
Include: {include_terms if include_terms else 'None specified'}
Exclude: {exclude_terms if exclude_terms else 'None specified'}

{domain_guidance if domain_guidance else ""}

{custom_instructions if custom_instructions else ""}

INSTRUCTIONS:
Generate comprehensive but focused search terms for scientific literature filtering. The goal is to capture relevant papers while avoiding noise from unrelated content.

For the given topic, include terms that would capture:
1. Core concepts and terminology
2. Key methods and techniques  
3. Important entities (genes, proteins, compounds, algorithms, etc.)
4. Related processes and mechanisms
5. Application domains and contexts
6. Common abbreviations and synonyms
7. Alternative spellings and variants

Also identify terms to exclude that might introduce irrelevant content.

RESPOND ONLY with a JSON object in this exact format:
{{
  "topic": "{topic}",
  "domain": "{domain if domain else 'general'}",
  "search_config": {{
    "core_terms": ["primary concepts and terminology"],
    "entity_terms": ["specific entities like genes, algorithms, compounds"],
    "method_terms": ["techniques, approaches, methodologies"],
    "context_terms": ["application domains, use cases, related fields"],
    "synonym_terms": ["alternative names, abbreviations, variants"],
    "exclude_terms": ["terms to exclude to reduce noise"]
  }},
  "query_strategies": {{
    "broad_queries": ["general queries for comprehensive coverage"],
    "focused_queries": ["specific targeted queries"],
    "exclusion_patterns": ["patterns to exclude irrelevant content"]
  }},
  "metadata": {{
    "confidence": 0.9,
    "estimated_papers": 1000,
    "domain_specificity": "medium",
    "generated_by": "LLM query generator",
    "model_used": "{self.config.model_name}",
    "timestamp": "{datetime.now().isoformat()}"
  }}
}}

Ensure the JSON is valid and well-formatted."""
        
        return prompt
    
    def query_llm(self, prompt: str) -> str:
        """Send query to LLM API and get response"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            logger.info(f"Querying {self.config.api_provider} API...")
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            response_data = response.json()
            llm_response = response_data["choices"][0]["message"]["content"].strip()
            
            logger.info("LLM query completed successfully")
            return llm_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise Exception(f"API request failed: {e}")
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise Exception(f"Unexpected API response format: {e}")
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the LLM response"""
        
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            query_config = json.loads(json_str)
            
            # Validate required structure
            required_keys = ['topic', 'search_config']
            for key in required_keys:
                if key not in query_config:
                    raise ValueError(f"Missing required key: {key}")
            
            # Validate search_config structure
            search_config = query_config['search_config']
            expected_keys = ['core_terms', 'entity_terms', 'method_terms', 
                           'context_terms', 'synonym_terms', 'exclude_terms']
            
            for key in expected_keys:
                if key not in search_config:
                    search_config[key] = []
            
            # Ensure query_strategies exists
            if 'query_strategies' not in query_config:
                query_config['query_strategies'] = {
                    'broad_queries': [],
                    'focused_queries': [], 
                    'exclusion_patterns': []
                }
            
            logger.info("LLM response parsed and validated successfully")
            return query_config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in LLM response: {e}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
    
    def generate_queries(self, 
                        topic: str,
                        domain: Optional[str] = None,
                        include_terms: Optional[List[str]] = None,
                        exclude_terms: Optional[List[str]] = None,
                        custom_instructions: Optional[str] = None,
                        save_raw_response: bool = True) -> QueryResult:
        """
        Generate comprehensive query configuration for the given topic
        
        Args:
            topic: The research topic to generate queries for
            domain: Optional domain specification (biology, medicine, ai, etc.)
            include_terms: Additional terms the user wants to include
            exclude_terms: Terms the user wants to exclude
            custom_instructions: Additional custom instructions for the LLM
            save_raw_response: Whether to save the raw LLM response
            
        Returns:
            QueryResult containing the query configuration
        """
        
        logger.info(f"Generating queries for topic: '{topic}'")
        if domain:
            logger.info(f"Domain: {domain}")
        if include_terms:
            logger.info(f"Include terms: {include_terms}")
        if exclude_terms:
            logger.info(f"Exclude terms: {exclude_terms}")
        
        # Create prompt
        prompt = self.create_domain_agnostic_prompt(
            topic, domain, include_terms, exclude_terms, custom_instructions
        )
        
        # Query LLM
        raw_response = self.query_llm(prompt)
        
        # Parse response
        query_config = self.parse_llm_response(raw_response)
        
        # Create result
        timestamp = datetime.now().isoformat()
        result = QueryResult(
            topic=topic,
            search_config=query_config,
            metadata={
                'domain': domain,
                'include_terms': include_terms,
                'exclude_terms': exclude_terms,
                'custom_instructions': custom_instructions,
                'generation_timestamp': timestamp,
                'config': self.config.__dict__
            },
            raw_response=raw_response,
            timestamp=timestamp
        )
        
        # Save raw response if requested
        if save_raw_response:
            self._save_raw_response(result)
        
        logger.info("Query generation completed successfully")
        self._log_query_stats(result)
        
        return result
    
    def _save_raw_response(self, result: QueryResult, data_dir: str = "data") -> Path:
        """Save raw LLM response for debugging/analysis"""
        # Ensure data directory exists
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() or c in ' -_' else '' for c in result.topic)
        safe_topic = safe_topic.replace(' ', '_').lower()
        
        filename = f"llm_raw_response_{safe_topic}_{timestamp}.txt"
        filepath = data_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Topic: {result.topic}\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Provider: {self.config.api_provider}\n")
            f.write(f"Domain: {result.metadata.get('domain', 'N/A')}\n")
            f.write(f"Include terms: {result.metadata.get('include_terms', [])}\n")
            f.write(f"Exclude terms: {result.metadata.get('exclude_terms', [])}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"PROMPT SENT TO LLM:\n")
            f.write(f"(Note: Actual prompt may be longer, this is reconstructed)\n")
            f.write(f"Topic: {result.topic}\n")
            f.write(f"Domain: {result.metadata.get('domain', 'general')}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"RAW LLM RESPONSE:\n{result.raw_response}\n")
        
        logger.info(f"Raw response saved to: {filepath}")
        return filepath
    
    def _log_query_stats(self, result: QueryResult):
        """Log statistics about generated queries"""
        config = result.search_config.get('search_config', {})
        
        stats = {
            'core_terms': len(config.get('core_terms', [])),
            'entity_terms': len(config.get('entity_terms', [])),
            'method_terms': len(config.get('method_terms', [])),
            'context_terms': len(config.get('context_terms', [])),
            'synonym_terms': len(config.get('synonym_terms', [])),
            'exclude_terms': len(config.get('exclude_terms', []))
        }
        
        logger.info("Generated query statistics:")
        for term_type, count in stats.items():
            logger.info(f"  {term_type}: {count}")
    
    def save_query_config(self, result: QueryResult, output_file: str, data_dir: str = "data") -> Path:
        """Save the query configuration to a JSON file"""
        
        # If output_file is just a filename, put it in data directory
        output_path = Path(output_file)
        if not output_path.is_absolute() and output_path.parent == Path('.'):
            data_path = Path(data_dir)
            data_path.mkdir(parents=True, exist_ok=True)
            output_path = data_path / output_file
        
        # Create complete config structure
        complete_config = {
            'topic': result.topic,
            'search_config': result.search_config.get('search_config', {}),
            'query_strategies': result.search_config.get('query_strategies', {}),
            'metadata': {
                **result.search_config.get('metadata', {}),
                **result.metadata
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(complete_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Query configuration saved to: {output_path}")
        return output_path
    
    def generate_multiple_domains(self, 
                                 topic: str,
                                 domains: List[str],
                                 **kwargs) -> Dict[str, QueryResult]:
        """Generate query configurations for multiple domains"""
        results = {}
        
        for domain in domains:
            logger.info(f"Generating queries for domain: {domain}")
            try:
                result = self.generate_queries(topic, domain=domain, **kwargs)
                results[domain] = result
            except Exception as e:
                logger.error(f"Failed to generate queries for domain {domain}: {e}")
                continue
        
        return results
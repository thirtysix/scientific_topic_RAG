"""
Generalized Literature Fetcher

Configurable PubMed literature retrieval system that uses generated query configurations
to fetch relevant scientific papers from any domain.
"""

import time
import json
import re
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from ..config.pipeline_config import LiteratureFetchConfig
    from ..utils.logger import PipelineLogger
    from .query_generator import QueryResult
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.pipeline_config import LiteratureFetchConfig
    from utils.logger import PipelineLogger
    from core.query_generator import QueryResult


@dataclass
class PaperResult:
    """Container for a single paper result"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: int
    doi: Optional[str]
    pmc_id: Optional[str]
    relevance_score: float
    query_source: str


class LiteratureFetcher:
    """Configurable literature fetcher using PubMed API"""
    
    def __init__(self, config: Optional[LiteratureFetchConfig] = None):
        """
        Initialize the literature fetcher
        
        Args:
            config: Fetch configuration
        """
        self.config = config or LiteratureFetchConfig()
        self.logger = PipelineLogger("LiteratureFetcher")
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Setup requests session with appropriate headers"""
        self.session.headers.update({
            'User-Agent': f'{self.config.tool_name} (mailto:{self.config.email})',
            'Accept': 'application/json'
        })
    
    def build_pubmed_queries(self, query_config: QueryResult) -> List[Dict[str, str]]:
        """
        Build PubMed search queries from generated query configuration

        Args:
            query_config: Generated query configuration

        Returns:
            List of PubMed query dictionaries
        """
        queries = []
        search_config = query_config.search_config.get('search_config', {})

        # Core concept queries - these define the main topic
        core_terms = search_config.get('core_terms', [])
        if core_terms:
            core_query = '(' + ' OR '.join([f'"{term}"[Title/Abstract]' for term in core_terms[:10]]) + ')'
            queries.append({
                'query': core_query,
                'type': 'core_concepts',
                'description': 'Core concepts and terminology'
            })

            # Build a focused core query for combining with other categories
            # Use the most important core terms (first 3-5) for focused searches
            focused_core = '(' + ' OR '.join([f'"{term}"[Title/Abstract]' for term in core_terms[:3]]) + ')'
        else:
            focused_core = None

        # Entity-based queries - entities are usually specific to the topic, so can stand alone
        entity_terms = search_config.get('entity_terms', [])
        if entity_terms:
            entity_query = '(' + ' OR '.join([f'"{term}"[Title/Abstract]' for term in entity_terms[:10]]) + ')'
            queries.append({
                'query': entity_query,
                'type': 'entities',
                'description': 'Key entities and components'
            })

        # Method-based queries - MUST be combined with core topic to avoid generic results
        method_terms = search_config.get('method_terms', [])
        if method_terms and focused_core:
            method_query = '(' + ' OR '.join([f'"{term}"[Title/Abstract]' for term in method_terms[:10]]) + ')'
            # Combine with core topic to keep it focused
            combined_method_query = f'{focused_core} AND {method_query}'
            queries.append({
                'query': combined_method_query,
                'type': 'methods',
                'description': 'Methods and techniques for topic'
            })

        # Context-based queries - also combine with core for focus
        context_terms = search_config.get('context_terms', [])
        if context_terms and focused_core:
            context_query = '(' + ' OR '.join([f'"{term}"[Title/Abstract]' for term in context_terms[:10]]) + ')'
            # Combine with core topic for relevance
            combined_context_query = f'{focused_core} AND {context_query}'
            queries.append({
                'query': combined_context_query,
                'type': 'context',
                'description': 'Application contexts for topic'
            })

        # Combined broad query - mix of core and entity terms (not methods to avoid dilution)
        # This provides a broad but still topic-focused search
        combined_terms = (core_terms[:5] + entity_terms[:10])[:15]
        if combined_terms:
            broad_query = '(' + ' OR '.join([f'"{term}"[Title/Abstract]' for term in combined_terms]) + ')'
            queries.append({
                'query': broad_query,
                'type': 'combined',
                'description': 'Combined broad search'
            })
        
        # v003: Replace NOT operator with positive basic research terms
        exclude_terms = search_config.get('exclude_terms', [])
        if exclude_terms and hasattr(self.config, 'use_not_operator') and getattr(self.config, 'use_not_operator', True):
            # Only use NOT operator if explicitly enabled (disabled by default in v003)
            exclusion_clause = ' AND NOT (' + ' OR '.join([f'"{term}"[Title/Abstract]' for term in exclude_terms[:10]]) + ')'
            for query_dict in queries:
                query_dict['query'] += exclusion_clause
        
        # Add basic research focus terms (positive approach)
        if hasattr(self.config, 'use_basic_research_focus') and getattr(self.config, 'use_basic_research_focus', False):
            basic_research_terms = [
                'molecular mechanisms', 'gene expression', 'protein function', 
                'cellular processes', 'developmental biology', 'evolutionary biology',
                'biochemistry', 'molecular biology', 'genetics'
            ]
            basic_research_clause = ' AND (' + ' OR '.join([f'"{term}"[Title/Abstract]' for term in basic_research_terms[:5]]) + ')'
            for query_dict in queries:
                query_dict['query'] += basic_research_clause
        
        # Add date filter if configured
        if self.config.date_range_years:
            current_year = time.gmtime().tm_year
            start_year = current_year - self.config.date_range_years
            date_filter = f' AND "{start_year}"[Date - Publication] : "3000"[Date - Publication]'
            for query_dict in queries:
                query_dict['query'] += date_filter
        
        # Add publication type filters
        pub_filters = self._build_publication_filters()
        if pub_filters:
            for query_dict in queries:
                query_dict['query'] += pub_filters
        
        self.logger.info(f"Built {len(queries)} PubMed queries")
        return queries
    
    def _build_publication_filters(self) -> str:
        """Build publication type and other filtering clauses"""
        filters = []
        
        # v003: Only apply publication filtering if explicitly enabled
        if not hasattr(self.config, 'exclude_publication_types') or not getattr(self.config, 'exclude_publication_types', False):
            return ''
        
        # Publication type exclusions
        if hasattr(self.config, 'exclude_clinical_trials') and self.config.exclude_clinical_trials:
            filters.extend([
                'NOT "Clinical Trial"[Publication Type]',
                'NOT "Clinical Trial, Phase I"[Publication Type]',
                'NOT "Clinical Trial, Phase II"[Publication Type]', 
                'NOT "Clinical Trial, Phase III"[Publication Type]',
                'NOT "Clinical Trial, Phase IV"[Publication Type]',
                'NOT "Randomized Controlled Trial"[Publication Type]',
                'NOT "Controlled Clinical Trial"[Publication Type]'
            ])
        
        if self.config.exclude_case_reports:
            filters.append('NOT "Case Reports"[Publication Type]')
            
        if self.config.exclude_editorials:
            filters.append('NOT "Editorial"[Publication Type]')
            
        if self.config.exclude_letters:
            filters.append('NOT "Letter"[Publication Type]')
            
        if self.config.exclude_news:
            filters.append('NOT "News"[Publication Type]')
        
        # Language filter (disabled for now to avoid over-filtering)
        # if hasattr(self.config, 'languages') and self.config.languages:
        #     lang_filters = ' OR '.join([f'"{lang}"[Language]' for lang in self.config.languages])
        #     filters.append(f'({lang_filters})')
        
        # MeSH term filtering
        if hasattr(self.config, 'use_mesh_terms') and self.config.use_mesh_terms:
            if hasattr(self.config, 'exclude_clinical_mesh') and self.config.exclude_clinical_mesh:
                clinical_mesh = [
                    'NOT "Clinical Medicine"[MeSH Terms]',
                    'NOT "Drug Therapy"[MeSH Terms]',
                    'NOT "Therapeutics"[MeSH Terms]',
                    'NOT "Clinical Protocols"[MeSH Terms]',
                    'NOT "Patient Care"[MeSH Terms]'
                ]
                filters.extend(clinical_mesh)
        
        # Journal exclusions (if enabled)
        if self.config.exclude_clinical_journals:
            clinical_journals = [
                'NOT "N Engl J Med"[Journal]',
                'NOT "JAMA"[Journal]',
                'NOT "Lancet"[Journal]',
                'NOT "BMJ"[Journal]',
                'NOT "Ann Intern Med"[Journal]'
            ]
            filters.extend(clinical_journals)
        
        # Combine all filters
        if filters:
            return ' AND ' + ' AND '.join(filters)
        return ''
    
    def search_pubmed(self, query: str, max_results: int) -> List[str]:
        """
        Search PubMed and return list of PMIDs

        Args:
            query: PubMed search query
            max_results: Maximum number of results to return

        Returns:
            List of PMIDs
        """
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        # PubMed effectively limits results to ~10,000 per query when using retstart
        # To get more, we need to use date ranges or other strategies
        effective_limit = 9999  # PubMed's practical limit for pagination
        batch_size = 5000  # Size for each batch request
        all_pmids = []

        # First, get the total count
        params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': 0,  # Just get count
            'tool': self.config.tool_name,
            'email': self.config.email
        }

        if self.config.api_key:
            params['api_key'] = self.config.api_key

        try:
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            total_count = int(data.get('esearchresult', {}).get('count', 0))

            if total_count == 0:
                return []

            # Determine how many results to actually fetch
            results_to_fetch = min(total_count, max_results)

            # If we need more than the effective limit, we'll need a different strategy
            if results_to_fetch > effective_limit:
                self.logger.info(f"Query found {total_count} results, need {results_to_fetch}, using year-based splitting")
                # Use year-based approach for large result sets
                all_pmids = self._fetch_large_result_set_by_year(query, results_to_fetch, total_count)
                return all_pmids[:max_results]  # Ensure we don't exceed the requested limit
            else:
                self.logger.info(f"Query found {total_count} results, fetching up to {results_to_fetch}")

            # Now fetch in batches with retry logic
            for start_index in range(0, results_to_fetch, batch_size):
                batch_num = start_index // batch_size + 1
                retries = 3  # Number of retries for each batch
                batch_fetched = False

                for attempt in range(retries):
                    try:
                        batch_params = {
                            'db': 'pubmed',
                            'term': query,
                            'retmode': 'json',
                            'retstart': start_index,
                            'retmax': min(batch_size, results_to_fetch - start_index),
                            'sort': 'relevance',
                            'tool': self.config.tool_name,
                            'email': self.config.email
                        }

                        if self.config.api_key:
                            batch_params['api_key'] = self.config.api_key

                        response = self.session.get(search_url, params=batch_params, timeout=30)
                        response.raise_for_status()

                        try:
                            batch_data = response.json()
                            batch_pmids = batch_data.get('esearchresult', {}).get('idlist', [])

                            if batch_pmids:  # Successfully got PMIDs
                                all_pmids.extend(batch_pmids)
                                self.logger.debug(f"Batch {batch_num}: Fetched {len(batch_pmids)} PMIDs (start: {start_index})")
                                batch_fetched = True
                                break
                            else:
                                self.logger.warning(f"Batch {batch_num}: Empty result, retrying...")

                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Batch {batch_num}, attempt {attempt + 1}: JSON decode error: {e}")

                            # Try to clean and parse the response
                            try:
                                text = response.text
                                # Remove control characters
                                text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                                batch_data = json.loads(text)
                                batch_pmids = batch_data.get('esearchresult', {}).get('idlist', [])

                                if batch_pmids:
                                    all_pmids.extend(batch_pmids)
                                    self.logger.info(f"Batch {batch_num}: Recovered {len(batch_pmids)} PMIDs after cleaning")
                                    batch_fetched = True
                                    break
                            except:
                                if attempt < retries - 1:
                                    self.logger.warning(f"Batch {batch_num}: Parse failed, retrying...")
                                    time.sleep(1)  # Brief pause before retry
                                continue

                    except requests.RequestException as req_error:
                        self.logger.warning(f"Batch {batch_num}, attempt {attempt + 1}: Request error: {req_error}")
                        if attempt < retries - 1:
                            time.sleep(2)  # Longer pause for network errors

                if not batch_fetched:
                    self.logger.error(f"Batch {batch_num}: Failed after {retries} attempts, continuing with partial results")

                # Rate limiting between batches
                if start_index + batch_size < results_to_fetch:
                    time.sleep(self.config.rate_limit_delay)

            self.logger.info(f"Found {len(all_pmids)} PMIDs for query")
            return all_pmids

        except requests.RequestException as e:
            self.logger.error(f"PubMed search failed: {e}")
            return []

    def _fetch_large_result_set_by_year(self, query: str, max_results: int, total_count: int) -> List[str]:
        """
        Fetch large result sets by splitting queries by year ranges
        This works around PubMed's 10,000 result pagination limit

        Args:
            query: The base PubMed query
            max_results: Maximum results to fetch
            total_count: Total available results

        Returns:
            List of PMIDs
        """
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        all_pmids = []
        current_year = time.gmtime().tm_year

        # Strategy: Split by year ranges to get around the 10k limit
        # Start with recent years and work backwards
        year_ranges = [
            (current_year - 1, current_year),      # Last 2 years
            (current_year - 3, current_year - 2),  # 2-4 years ago
            (current_year - 5, current_year - 4),  # 4-6 years ago
            (current_year - 10, current_year - 6), # 6-11 years ago
            (current_year - 20, current_year - 11),# 11-21 years ago
            (1900, current_year - 21)              # Older papers
        ]

        for start_year, end_year in year_ranges:
            if len(all_pmids) >= max_results:
                break

            # Add date filter to the query
            year_query = f'{query} AND "{start_year}"[Date - Publication] : "{end_year}"[Date - Publication]'

            # Get count for this year range
            params = {
                'db': 'pubmed',
                'term': year_query,
                'retmode': 'json',
                'retmax': 0,
                'tool': self.config.tool_name,
                'email': self.config.email
            }

            if self.config.api_key:
                params['api_key'] = self.config.api_key

            try:
                response = self.session.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()
                range_count = int(data.get('esearchresult', {}).get('count', 0))

                if range_count == 0:
                    continue

                self.logger.info(f"Year range {start_year}-{end_year}: {range_count} results")

                # Fetch results for this year range (up to 9999)
                fetch_count = min(range_count, 9999, max_results - len(all_pmids))

                params['retmax'] = fetch_count
                response = self.session.get(search_url, params=params)
                response.raise_for_status()

                data = response.json()
                pmids = data.get('esearchresult', {}).get('idlist', [])

                all_pmids.extend(pmids)
                self.logger.info(f"Fetched {len(pmids)} PMIDs from year range {start_year}-{end_year}")

                # Rate limiting
                time.sleep(self.config.rate_limit_delay)

            except Exception as e:
                self.logger.warning(f"Error fetching year range {start_year}-{end_year}: {e}")
                continue

        self.logger.info(f"Total PMIDs fetched using year-based splitting: {len(all_pmids)}")
        return all_pmids

    def fetch_paper_details(self, pmids: List[str]) -> List[PaperResult]:
        """
        Fetch detailed information for a list of PMIDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of PaperResult objects
        """
        if not pmids:
            return []
        
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        # Process in batches to avoid URL length limits
        batch_size = 200
        all_papers = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            params = {
                'db': 'pubmed',
                'id': ','.join(batch_pmids),
                'retmode': 'xml',
                'tool': self.config.tool_name,
                'email': self.config.email
            }
            
            if self.config.api_key:
                params['api_key'] = self.config.api_key
            
            try:
                response = self.session.get(fetch_url, params=params)
                response.raise_for_status()
                
                # Parse XML response (simplified - would need proper XML parsing)
                papers = self._parse_pubmed_xml(response.text, batch_pmids)
                all_papers.extend(papers)
                
                # Rate limiting
                time.sleep(self.config.rate_limit_delay)
                
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch batch {i//batch_size + 1}: {e}")
                continue
        
        self.logger.info(f"Fetched details for {len(all_papers)} papers")
        return all_papers
    
    def _parse_pubmed_xml(self, xml_content: str, pmids: List[str]) -> List[PaperResult]:
        """
        Parse PubMed XML response
        
        Args:
            xml_content: XML response content
            pmids: List of PMIDs for this batch
            
        Returns:
            List of PaperResult objects
        """
        papers = []
        
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract PMID
                    pmid_elem = article.find('.//PMID')
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text
                    
                    # Extract title
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "No title available"
                    
                    # Extract abstract
                    abstract_parts = []
                    for abstract_elem in article.findall('.//AbstractText'):
                        if abstract_elem.text:
                            # Handle labeled abstracts
                            label = abstract_elem.get('Label', '')
                            text = abstract_elem.text
                            if label:
                                abstract_parts.append(f"{label}: {text}")
                            else:
                                abstract_parts.append(text)
                    
                    abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        last_name = author.find('LastName')
                        fore_name = author.find('ForeName')
                        if last_name is not None and fore_name is not None:
                            authors.append(f"{fore_name.text} {last_name.text}")
                        elif last_name is not None:
                            authors.append(last_name.text)
                    
                    if not authors:
                        authors = ["Unknown Author"]
                    
                    # Extract journal
                    journal_elem = article.find('.//Journal/Title')
                    if journal_elem is None:
                        journal_elem = article.find('.//Journal/ISOAbbreviation')
                    journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
                    
                    # Extract year
                    year_elem = article.find('.//PubDate/Year')
                    try:
                        year = int(year_elem.text) if year_elem is not None else 2023
                    except:
                        year = 2023
                    
                    # Extract DOI
                    doi = None
                    for article_id in article.findall('.//ArticleId'):
                        if article_id.get('IdType') == 'doi':
                            doi = article_id.text
                            break
                    
                    # Extract PMC ID
                    pmc_id = None
                    for article_id in article.findall('.//ArticleId'):
                        if article_id.get('IdType') == 'pmc':
                            pmc_id = article_id.text
                            break
                    
                    paper = PaperResult(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal,
                        year=year,
                        doi=doi,
                        pmc_id=pmc_id,
                        relevance_score=0.5,  # Will be calculated later
                        query_source="pubmed"
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse article: {e}")
                    continue
            
            return papers
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            # Fallback to placeholder data
            return [
                PaperResult(
                    pmid=pmid,
                    title=f"[Parsing Error] PMID {pmid}",
                    abstract="Could not parse abstract from PubMed response",
                    authors=["Unknown"],
                    journal="Unknown Journal",
                    year=2023,
                    doi=None,
                    pmc_id=None,
                    relevance_score=0.1,
                    query_source="pubmed_error"
                ) for pmid in pmids
            ]
    
    def calculate_relevance_scores(self, 
                                 papers: List[PaperResult], 
                                 query_config: QueryResult) -> List[PaperResult]:
        """
        Calculate relevance scores for papers based on query terms (IMPROVED VERSION)
        
        Args:
            papers: List of papers to score
            query_config: Query configuration used
            
        Returns:
            Papers with updated relevance scores
        """
        search_config = query_config.search_config.get('search_config', {})
        
        # Use weighted term categories (core terms matter more)
        term_weights = {
            'core_terms': 3.0,      # Most important
            'entity_terms': 2.0,    # Very important  
            'method_terms': 1.5,    # Important
            'context_terms': 1.0,   # Somewhat important
            'synonym_terms': 1.0    # Somewhat important
        }
        
        exclude_terms = search_config.get('exclude_terms', [])
        
        for paper in papers:
            # Combine title and abstract for scoring
            text = f"{paper.title} {paper.abstract}".lower()
            
            # Calculate weighted positive score
            total_weighted_score = 0
            total_possible_weight = 0
            
            for term_type, weight in term_weights.items():
                terms = search_config.get(term_type, [])
                if terms:
                    # Count matches with partial matching for better recall
                    matches = 0
                    for term in terms:
                        term_lower = term.lower()
                        # Exact match gets full credit
                        if term_lower in text:
                            matches += 1
                        # Partial match for multi-word terms gets partial credit
                        elif len(term.split()) > 1:
                            words = term_lower.split()
                            if any(word in text for word in words if len(word) > 3):
                                matches += 0.5
                    
                    # Calculate weighted score for this term type
                    if len(terms) > 0:
                        type_score = matches / len(terms)
                        total_weighted_score += type_score * weight
                        total_possible_weight += weight
            
            # Normalize to 0-1 range
            positive_score = total_weighted_score / total_possible_weight if total_possible_weight > 0 else 0
            
            # Small penalty for exclusion terms (not too harsh)
            exclusion_matches = sum(1 for term in exclude_terms if term.lower() in text)
            exclusion_penalty = exclusion_matches * 0.05  # Reduced penalty
            
            # Final relevance score
            paper.relevance_score = max(0, positive_score - exclusion_penalty)
        
        # Sort by relevance score
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        if papers:
            self.logger.info(f"Relevance scoring completed. Score range: {papers[0].relevance_score:.3f} to {papers[-1].relevance_score:.3f}")
        else:
            self.logger.warning("No papers to score - empty paper list")
        
        return papers
    
    def fetch_literature(self, query_config: QueryResult) -> Dict[str, Any]:
        """
        Main method to fetch literature based on query configuration

        Args:
            query_config: Generated query configuration

        Returns:
            Dictionary containing fetched literature and metadata
        """
        self.logger.info(f"Starting literature fetch for topic: {query_config.topic}")
        start_time = time.time()

        # Build PubMed queries
        pubmed_queries = self.build_pubmed_queries(query_config)

        # Get per-category limits if configured
        category_limits = getattr(self.config, 'max_papers_per_category', {})

        # Track papers per category
        all_pmids = set()
        category_pmids = {}
        query_results = {}

        for query_info in pubmed_queries:
            query_type = query_info['type']
            self.logger.info(f"Executing query: {query_info['description']}")

            # Determine max results for this query based on category limit
            category_limit = category_limits.get(query_type) if category_limits else None
            max_for_query = self.config.max_per_query

            # If category limit exists and we've already collected some PMIDs for this category
            if category_limit and query_type in category_pmids:
                already_collected = len(category_pmids[query_type])
                if already_collected >= category_limit:
                    self.logger.info(f"Category '{query_type}' already reached limit of {category_limit} papers")
                    continue
                # Adjust max_for_query to not exceed category limit
                max_for_query = min(max_for_query, category_limit - already_collected)
            elif category_limit:
                # First query for this category, limit to category max
                max_for_query = min(max_for_query, category_limit)

            pmids = self.search_pubmed(query_info['query'], max_for_query)

            # Track PMIDs by category
            if query_type not in category_pmids:
                category_pmids[query_type] = set()

            # Apply category limit if specified
            if category_limit:
                remaining_slots = category_limit - len(category_pmids[query_type])
                pmids_to_add = pmids[:remaining_slots]
            else:
                pmids_to_add = pmids

            category_pmids[query_type].update(pmids_to_add)
            all_pmids.update(pmids_to_add)

            query_results[query_type] = {
                'query': query_info['query'],
                'pmids': pmids_to_add,
                'count': len(pmids_to_add),
                'category_limit': category_limit,
                'total_found': len(pmids)
            }

            # Rate limiting
            time.sleep(self.config.rate_limit_delay)

            # Check if we've hit the total maximum
            if len(all_pmids) >= self.config.max_total_papers:
                self.logger.info(f"Reached total paper limit of {self.config.max_total_papers}")
                break

        # Limit total papers
        all_pmids = list(all_pmids)[:self.config.max_total_papers]

        # Log category distribution
        category_summary = {cat: len(pmids) for cat, pmids in category_pmids.items()}
        self.logger.info(f"Found {len(all_pmids)} unique papers across categories: {category_summary}")

        # Fetch detailed paper information
        papers = self.fetch_paper_details(all_pmids)

        # Tag papers with their source category for tracking
        for paper in papers:
            for cat, pmids in category_pmids.items():
                if paper.pmid in pmids:
                    paper.query_source = cat
                    break

        # Calculate relevance scores
        papers = self.calculate_relevance_scores(papers, query_config)

        # Filter by minimum relevance score
        filtered_papers = [p for p in papers if p.relevance_score >= self.config.min_relevance_score]

        # Create category distribution for filtered papers
        filtered_category_counts = {}
        for paper in filtered_papers:
            cat = paper.query_source
            filtered_category_counts[cat] = filtered_category_counts.get(cat, 0) + 1

        # Create result structure
        result = {
            'metadata': {
                'topic': query_config.topic,
                'total_papers': len(filtered_papers),
                'total_queries': len(pubmed_queries),
                'fetch_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'fetch_duration': time.time() - start_time,
                'config': self.config.__dict__,
                'average_relevance': sum(p.relevance_score for p in filtered_papers) / len(filtered_papers) if filtered_papers else 0,
                'category_distribution': filtered_category_counts,
                'category_limits_applied': category_limits if category_limits else {}
            },
            'query_results': query_results,
            'papers': [paper.__dict__ for paper in filtered_papers]
        }

        self.logger.info(f"Literature fetch completed: {len(filtered_papers)} papers")
        return result
    
    def save_literature(self, literature_data: Dict[str, Any], output_file: str) -> Path:
        """
        Save literature data to JSON file
        
        Args:
            literature_data: Literature data to save
            output_file: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(literature_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Literature data saved to: {output_path}")
        return output_path
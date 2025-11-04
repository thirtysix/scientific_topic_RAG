#!/usr/bin/env python3
"""
RAG System - Part 2: Literature Fetching WITH MeSH TERMS
Fetches literature from PubMed including MeSH terms for paper type classification
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load environment (only for API keys)
from run_with_env import load_env_file
load_env_file()

from core.query_generator import QueryResult
from core.literature_fetcher import LiteratureFetcher, LiteratureFetchConfig

# ============================================================================
# 🎯 LITERATURE FETCHING CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Literature fetching settings
MAX_TOTAL_PAPERS = 20000                  # Maximum papers to fetch across all queries
MAX_PAPERS_PER_QUERY = 10000              # Maximum papers per individual search
MIN_RELEVANCE_THRESHOLD = 0.02           # Minimum relevance score to keep papers

# NEW: MeSH-based paper type filtering
ENABLE_MESH_FILTERING = True              # Enable MeSH term extraction and filtering
PAPER_TYPE_FOCUS = "methodology"          # Options: "methodology", "review", "clinical", "basic_research", None (all)

# Specific MeSH terms for methodology papers
METHODOLOGY_MESH_TERMS = [
    "methods",                            # General methods qualifier
    "Single-Cell Analysis/methods",       # Single-cell specific
    "RNA-Seq/methods",                    # RNA sequencing methods
    "Sequence Analysis, RNA/methods",     # RNA analysis methods
    "Computational Biology/methods",      # Computational methods
    "High-Throughput Nucleotide Sequencing/methods",
    "Gene Expression Profiling/methods",
    "Algorithms",                         # Algorithm development
    "Software",                           # Software tools
    "Workflow"                            # Workflow development
]

# MeSH terms for other paper types
REVIEW_MESH_TERMS = ["Review", "Review Literature", "Systematic Review", "Meta-Analysis"]
CLINICAL_MESH_TERMS = ["Clinical Trial", "Clinical Study", "Humans", "Patient"]
BASIC_RESEARCH_MESH_TERMS = ["Animals", "Cell Line", "In Vitro Techniques", "Molecular Biology"]

INCLUDE_RECENT_YEARS = 10                # Only papers from last N years (None for all years)

# Rate limiting (be respectful to PubMed)
PUBMED_DELAY_SECONDS = 0.20              # ~3 requests per second
PUBMED_REQUEST_TIMEOUT = 30              # Timeout per request

# Publication filtering strategy
USE_NOT_OPERATOR = False                 # Disable NOT operator due to PubMed processing issues
USE_BASIC_RESEARCH_FOCUS = False         # Too restrictive - disable for now
USE_JOURNAL_FILTERING = False            # Too restrictive - disable for now
EXCLUDE_PUBLICATION_TYPES = False        # Disable to avoid zero results

# ============================================================================
# 🔧 ENHANCED IMPLEMENTATION WITH MESH TERMS
# ============================================================================

def get_mesh_terms_for_focus(paper_type_focus: Optional[str]) -> List[str]:
    """Get relevant MeSH terms based on paper type focus"""

    if not paper_type_focus:
        return []

    mesh_map = {
        "methodology": METHODOLOGY_MESH_TERMS,
        "review": REVIEW_MESH_TERMS,
        "clinical": CLINICAL_MESH_TERMS,
        "basic_research": BASIC_RESEARCH_MESH_TERMS
    }

    return mesh_map.get(paper_type_focus.lower(), [])


def create_literature_config():
    """Create literature fetching configuration with MeSH support"""
    literature_config = LiteratureFetchConfig()
    literature_config.email = os.getenv('EMAIL', 'your.email@example.com')
    literature_config.max_total_papers = MAX_TOTAL_PAPERS
    literature_config.max_per_query = MAX_PAPERS_PER_QUERY
    literature_config.min_relevance_score = MIN_RELEVANCE_THRESHOLD
    literature_config.rate_limit_delay = PUBMED_DELAY_SECONDS
    literature_config.timeout = PUBMED_REQUEST_TIMEOUT
    literature_config.date_range_years = INCLUDE_RECENT_YEARS

    # Apply publication filtering settings
    literature_config.use_not_operator = USE_NOT_OPERATOR
    literature_config.use_basic_research_focus = USE_BASIC_RESEARCH_FOCUS
    literature_config.use_journal_filtering = USE_JOURNAL_FILTERING
    literature_config.exclude_publication_types = EXCLUDE_PUBLICATION_TYPES

    # NEW: MeSH filtering configuration
    literature_config.enable_mesh_filtering = ENABLE_MESH_FILTERING
    literature_config.paper_type_focus = PAPER_TYPE_FOCUS
    literature_config.target_mesh_terms = get_mesh_terms_for_focus(PAPER_TYPE_FOCUS)

    return literature_config


def enhance_query_with_mesh(query: str, paper_type_focus: Optional[str]) -> str:
    """Enhance PubMed query with MeSH terms for paper type"""

    if not ENABLE_MESH_FILTERING or not paper_type_focus:
        return query

    # Add MeSH terms to query
    mesh_terms = get_mesh_terms_for_focus(paper_type_focus)

    if mesh_terms:
        # Create MeSH query component
        mesh_components = []
        for term in mesh_terms[:5]:  # Limit to avoid query too long
            # Handle terms with qualifiers
            if "/" in term:
                mesh_components.append(f'"{term}"[MeSH Terms]')
            else:
                mesh_components.append(f'"{term}"[MeSH Terms]')

        if mesh_components:
            mesh_query = " OR ".join(mesh_components)
            # Add to original query
            enhanced_query = f"({query}) AND ({mesh_query})"
            return enhanced_query

    return query


def parse_mesh_terms_from_xml(article_element) -> List[Dict[str, Any]]:
    """Extract MeSH terms from PubMed XML article element"""

    mesh_terms = []

    # Find MeSH heading list
    mesh_list = article_element.find('.//MeshHeadingList')
    if mesh_list is not None:
        for mesh_heading in mesh_list.findall('.//MeshHeading'):
            descriptor = mesh_heading.find('.//DescriptorName')
            if descriptor is not None:
                mesh_term = {
                    'descriptor': descriptor.text,
                    'major_topic': descriptor.get('MajorTopicYN', 'N') == 'Y',
                    'ui': descriptor.get('UI', ''),
                    'qualifiers': []
                }

                # Get qualifiers (e.g., /methods, /instrumentation)
                for qualifier in mesh_heading.findall('.//QualifierName'):
                    mesh_term['qualifiers'].append({
                        'name': qualifier.text,
                        'major_topic': qualifier.get('MajorTopicYN', 'N') == 'Y',
                        'ui': qualifier.get('UI', '')
                    })

                mesh_terms.append(mesh_term)

    return mesh_terms


def classify_paper_by_mesh(mesh_terms: List[Dict[str, Any]]) -> Dict[str, float]:
    """Classify paper type based on MeSH terms"""

    scores = {
        'methodology': 0.0,
        'review': 0.0,
        'clinical': 0.0,
        'basic_research': 0.0
    }

    if not mesh_terms:
        return scores

    # Check each MeSH term
    for mesh_term in mesh_terms:
        descriptor = mesh_term['descriptor'].lower()
        qualifiers = [q['name'].lower() for q in mesh_term.get('qualifiers', [])]
        is_major = mesh_term['major_topic']

        # Weight for major topics
        weight = 2.0 if is_major else 1.0

        # Check for methodology indicators
        if 'methods' in qualifiers or 'instrumentation' in qualifiers:
            scores['methodology'] += weight * 2  # Strong signal
        elif any(term in descriptor for term in ['algorithm', 'software', 'computational', 'bioinformatics']):
            scores['methodology'] += weight

        # Check for review indicators
        if 'review' in descriptor or 'meta-analysis' in descriptor:
            scores['review'] += weight * 2

        # Check for clinical indicators
        if any(term in descriptor for term in ['clinical', 'patient', 'treatment', 'therapy', 'human']):
            scores['clinical'] += weight

        # Check for basic research indicators
        if any(term in descriptor for term in ['animal', 'cell line', 'in vitro', 'molecular']):
            scores['basic_research'] += weight

    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        for key in scores:
            scores[key] = scores[key] / total

    return scores


def fetch_literature_with_mesh(query_file_path: str):
    """Enhanced literature fetching with MeSH terms"""

    query_file = Path(query_file_path)
    if not query_file.exists():
        print(f"❌ Query file not found: {query_file_path}")
        return False

    print("🏗️  RAG System - Part 2: Literature Fetching WITH MeSH Terms")
    print("=" * 60)
    print(f"Query file: {query_file.name}")
    if ENABLE_MESH_FILTERING and PAPER_TYPE_FOCUS:
        print(f"📋 Paper type focus: {PAPER_TYPE_FOCUS}")
        print(f"🔍 Target MeSH terms: {len(get_mesh_terms_for_focus(PAPER_TYPE_FOCUS))} terms")

    # Load queries
    with open(query_file, 'r') as f:
        query_data = json.load(f)

    query_result = QueryResult.from_json(query_data)

    # Create project directory
    project_dir, project_name = create_timestamped_project_dir(query_file_path)
    print(f"📁 Project directory: {project_dir}")

    # Copy query file to project directory
    shutil.copy2(query_file, project_dir / query_file.name)

    # Create literature fetcher with enhanced config
    literature_config = create_literature_config()
    fetcher = EnhancedLiteratureFetcher(literature_config)

    # Process each search query
    all_papers = []
    paper_type_stats = {
        'methodology': 0,
        'review': 0,
        'clinical': 0,
        'basic_research': 0,
        'unclassified': 0
    }

    print(f"\n🔍 Processing {len(query_result.search_queries)} search queries...")

    for idx, search_query in enumerate(query_result.search_queries, 1):
        print(f"\n📝 Query {idx}/{len(query_result.search_queries)}: {search_query.query[:100]}...")

        # Enhance query with MeSH terms if requested
        enhanced_query = enhance_query_with_mesh(search_query.query, PAPER_TYPE_FOCUS)

        if enhanced_query != search_query.query:
            print(f"   Enhanced with MeSH terms")

        # Fetch papers
        papers = fetcher.fetch_papers_for_query(enhanced_query, search_query.weight)

        # Classify papers by type if MeSH filtering is enabled
        if ENABLE_MESH_FILTERING:
            for paper in papers:
                if 'mesh_terms' in paper and paper['mesh_terms']:
                    paper['paper_type_scores'] = classify_paper_by_mesh(paper['mesh_terms'])

                    # Determine primary type
                    max_type = max(paper['paper_type_scores'], key=paper['paper_type_scores'].get)
                    if paper['paper_type_scores'][max_type] > 0.3:  # Confidence threshold
                        paper['primary_paper_type'] = max_type
                        paper_type_stats[max_type] += 1
                    else:
                        paper['primary_paper_type'] = 'unclassified'
                        paper_type_stats['unclassified'] += 1
                else:
                    paper['primary_paper_type'] = 'unclassified'
                    paper_type_stats['unclassified'] += 1

        all_papers.extend(papers)
        print(f"   ✅ Fetched {len(papers)} papers")

    # Filter by paper type if specified
    if ENABLE_MESH_FILTERING and PAPER_TYPE_FOCUS:
        filtered_papers = []
        for paper in all_papers:
            if paper.get('primary_paper_type') == PAPER_TYPE_FOCUS:
                filtered_papers.append(paper)
            elif paper.get('paper_type_scores', {}).get(PAPER_TYPE_FOCUS, 0) > 0.2:
                # Include papers with reasonable score for target type
                filtered_papers.append(paper)

        print(f"\n📊 Paper type filtering:")
        print(f"   Original papers: {len(all_papers)}")
        print(f"   {PAPER_TYPE_FOCUS} papers: {len(filtered_papers)}")
        all_papers = filtered_papers

    # Remove duplicates
    unique_papers = {}
    for paper in all_papers:
        pmid = paper.get('pmid')
        if pmid and pmid not in unique_papers:
            unique_papers[pmid] = paper

    final_papers = list(unique_papers.values())

    # Sort by relevance
    final_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    # Create output structure with MeSH metadata
    literature_data = {
        'metadata': {
            'query_file': query_file.name,
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(query_result.search_queries),
            'total_papers': len(final_papers),
            'mesh_filtering_enabled': ENABLE_MESH_FILTERING,
            'paper_type_focus': PAPER_TYPE_FOCUS,
            'paper_type_distribution': paper_type_stats if ENABLE_MESH_FILTERING else None
        },
        'queries': [
            {
                'query': sq.query,
                'weight': sq.weight,
                'category': sq.category,
                'specificity': sq.specificity,
                'mesh_enhanced': ENABLE_MESH_FILTERING
            }
            for sq in query_result.search_queries
        ],
        'papers': final_papers
    }

    # Save literature file
    literature_file = project_dir / f"{project_name}_literature.json"
    with open(literature_file, 'w') as f:
        json.dump(literature_data, f, indent=2)

    print(f"\n✅ Literature fetching completed!")
    print(f"📚 Total papers fetched: {len(final_papers)}")

    if ENABLE_MESH_FILTERING:
        print(f"\n📊 Paper type distribution:")
        for ptype, count in paper_type_stats.items():
            if count > 0:
                print(f"   {ptype}: {count} ({count/len(all_papers)*100:.1f}%)")

    print(f"💾 Saved to: {literature_file}")

    return literature_file


class EnhancedLiteratureFetcher(LiteratureFetcher):
    """Enhanced fetcher that extracts MeSH terms"""

    def _parse_pubmed_xml(self, xml_content: str, pmids: List[str]) -> List[Dict[str, Any]]:
        """Parse PubMed XML response including MeSH terms"""

        papers = []

        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(xml_content)

            for article in root.findall('.//PubmedArticle'):
                try:
                    # Get basic paper info (same as before)
                    paper = super()._parse_pubmed_xml_single(article)

                    # Add MeSH terms
                    if self.config.enable_mesh_filtering:
                        paper['mesh_terms'] = parse_mesh_terms_from_xml(article)

                    papers.append(paper)

                except Exception as e:
                    continue

        except Exception as e:
            print(f"Error parsing XML: {e}")

        return papers


def create_timestamped_project_dir(query_file_path: Path):
    """Create a timestamped project directory"""
    query_file = Path(query_file_path)

    # Extract project name from query file
    if query_file.stem.endswith('_queries'):
        project_base = query_file.stem[:-8]  # Remove '_queries' suffix
    else:
        project_base = query_file.stem

    # Extract timestamp from query file or create new one
    parts = project_base.split('_')
    if len(parts) >= 2 and len(parts[-1]) == 6 and parts[-1].isdigit():
        # Use existing timestamp from query file
        timestamp = f"{parts[-2]}_{parts[-1]}"
        project_name = '_'.join(parts[:-2])
    else:
        # Create new timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = project_base

    project_dir_name = f"{project_name}_{timestamp}"
    project_dir = Path("results") / project_dir_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (project_dir / "raw_responses").mkdir(exist_ok=True)
    (project_dir / "scripts").mkdir(exist_ok=True)

    return project_dir, project_name


def main():
    """Main function"""

    if len(sys.argv) != 2:
        print("Usage: python rag_part2_fetch_literature_mesh.py <query_file.json>")
        print("Example: python rag_part2_fetch_literature_mesh.py gene_reg_evolution_queries.json")
        return 1

    query_file = sys.argv[1]

    # Fetch literature with MeSH term extraction
    success = fetch_literature_with_mesh(query_file)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
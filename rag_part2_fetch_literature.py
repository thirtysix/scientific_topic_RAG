#!/usr/bin/env python3
"""
RAG System - Part 2: Literature Fetching
Fetches literature from PubMed using a query.json file and saves to literature.json
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

# Load environment (only for API keys)
from run_with_env import load_env_file
load_env_file()

from core.query_generator import QueryResult
from core.literature_fetcher import LiteratureFetcher, LiteratureFetchConfig

# ============================================================================
# 🎯 LITERATURE FETCHING CONFIGURATION
# ============================================================================

# Literature fetching settings
MAX_TOTAL_PAPERS = 30000                  # Maximum papers to fetch across all queries
MAX_PAPERS_PER_QUERY = 15000              # Maximum papers per individual search
MIN_RELEVANCE_THRESHOLD = 0.02           # Minimum relevance score to keep papers
INCLUDE_RECENT_YEARS = 25                # Only papers from last N years (None for all years)

# Per-category limits (None = no limit for that category)
# These limits apply to each query type independently
MAX_PAPERS_PER_CATEGORY = {
    'core_concepts': None,         # Limit for core concept queries
    'entities': 5000,              # Limit for entity-based queries
    'methods': 2000,               # Limit for method-based queries
    'context': 2000,               # Limit for context-based queries
    'combined': None               # Limit for combined broad queries
}

# Rate limiting (be respectful to PubMed)
PUBMED_DELAY_SECONDS = 0.20              # ~3 requests per second
PUBMED_REQUEST_TIMEOUT = 30              # Timeout per request

# Publication filtering strategy (v003: Minimal filtering to avoid issues)
USE_NOT_OPERATOR = False                 # Disable NOT operator due to PubMed processing issues
USE_BASIC_RESEARCH_FOCUS = False         # Too restrictive - disable for now
USE_JOURNAL_FILTERING = False            # Too restrictive - disable for now  
EXCLUDE_PUBLICATION_TYPES = False        # Disable to avoid zero results

# ============================================================================
# 🔧 IMPLEMENTATION
# ============================================================================

def create_literature_config():
    """Create literature fetching configuration"""
    literature_config = LiteratureFetchConfig()
    literature_config.email = os.getenv('EMAIL', 'your.email@example.com')
    literature_config.max_total_papers = MAX_TOTAL_PAPERS
    literature_config.max_per_query = MAX_PAPERS_PER_QUERY
    literature_config.min_relevance_score = MIN_RELEVANCE_THRESHOLD
    literature_config.rate_limit_delay = PUBMED_DELAY_SECONDS
    literature_config.timeout = PUBMED_REQUEST_TIMEOUT
    literature_config.date_range_years = INCLUDE_RECENT_YEARS

    # Apply publication filtering settings (v003: minimal approach)
    literature_config.use_not_operator = USE_NOT_OPERATOR
    literature_config.use_basic_research_focus = USE_BASIC_RESEARCH_FOCUS
    literature_config.use_journal_filtering = USE_JOURNAL_FILTERING
    literature_config.exclude_publication_types = EXCLUDE_PUBLICATION_TYPES

    # Add per-category limits
    literature_config.max_papers_per_category = MAX_PAPERS_PER_CATEGORY

    return literature_config


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


def save_run_summary(project_dir: Path, query_result, literature_data, fetcher, query_file_path, start_time, literature_config):
    """Save comprehensive run summary to file"""
    import time
    import urllib.parse
    
    run_summary = []
    run_summary.append("=" * 80)
    run_summary.append("RAG SYSTEM PART 2 - LITERATURE FETCHING SUMMARY")
    run_summary.append("=" * 80)
    run_summary.append(f"Generated: {datetime.now().isoformat()}")
    run_summary.append(f"Query file used: {Path(query_file_path).name}")
    run_summary.append(f"Total runtime: {time.time() - start_time:.1f}s")
    run_summary.append("")
    
    # Configuration
    run_summary.append("📋 FETCHING CONFIGURATION")
    run_summary.append("-" * 40)
    run_summary.append(f"Max Total Papers: {MAX_TOTAL_PAPERS}")
    run_summary.append(f"Max Per Query: {MAX_PAPERS_PER_QUERY}")
    run_summary.append(f"Min Relevance: {MIN_RELEVANCE_THRESHOLD}")
    run_summary.append(f"Recent Years: {INCLUDE_RECENT_YEARS}")
    run_summary.append("")

    # Per-category limits
    run_summary.append("📊 PER-CATEGORY LIMITS")
    run_summary.append("-" * 40)
    for category, limit in MAX_PAPERS_PER_CATEGORY.items():
        limit_str = str(limit) if limit is not None else "No limit"
        run_summary.append(f"{category}: {limit_str}")
    run_summary.append("")
    
    # Filtering settings (v003)
    run_summary.append("🔍 FILTERING SETTINGS (v003 - Minimal Approach)")
    run_summary.append("-" * 40)
    run_summary.append(f"Use NOT Operator: {USE_NOT_OPERATOR}")
    run_summary.append(f"Use Basic Research Focus: {USE_BASIC_RESEARCH_FOCUS}")
    run_summary.append(f"Use Journal Filtering: {USE_JOURNAL_FILTERING}")
    run_summary.append(f"Exclude Publication Types: {EXCLUDE_PUBLICATION_TYPES}")
    run_summary.append("")
    
    # Query summary
    search_config = query_result.search_config.get('search_config', {})
    total_terms = sum(len(terms) for terms in search_config.values() if isinstance(terms, list))
    run_summary.append(f"📝 QUERY TERMS SUMMARY: {total_terms} total terms")
    run_summary.append("-" * 40)
    for term_type, terms in search_config.items():
        if isinstance(terms, list):
            run_summary.append(f"{term_type.upper()}: {len(terms)} terms")
    run_summary.append("")
    
    # PubMed queries sent
    run_summary.append("🔍 ACTUAL PUBMED QUERIES SENT")
    run_summary.append("-" * 40)
    
    queries = fetcher.build_pubmed_queries(query_result)
    for i, query_dict in enumerate(queries):
        run_summary.append(f"Query {i+1}: {query_dict['description']}")
        run_summary.append(f"Type: {query_dict['type']}")
        run_summary.append(f"Full Query:")
        query = query_dict['query']
        run_summary.append(f"  {query}")
        run_summary.append("")
        
        # Add the complete URL for manual testing
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': literature_config.max_per_query,
            'sort': 'relevance',
            'tool': literature_config.tool_name,
            'email': literature_config.email
        }
        
        url_params = urllib.parse.urlencode(params)
        full_url = f"{base_url}?{url_params}"
        
        run_summary.append(f"Complete URL for manual testing:")
        run_summary.append(f"  {full_url}")
        run_summary.append("")
        
        browser_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(query)}&sort=relevance"
        run_summary.append(f"PubMed web interface URL:")
        run_summary.append(f"  {browser_url}")
        run_summary.append("=" * 60)
        run_summary.append("")
    
    # Literature results
    metadata = literature_data['metadata']
    run_summary.append("📚 LITERATURE FETCHING RESULTS")
    run_summary.append("-" * 40)
    run_summary.append(f"Total Papers Found: {metadata['total_papers']}")
    run_summary.append(f"Queries Executed: {len(literature_data['query_results'])}")
    run_summary.append(f"Fetch Duration: {metadata['fetch_duration']:.1f}s")
    run_summary.append(f"Average Relevance Score: {metadata['average_relevance']:.3f}")
    run_summary.append("")

    # Category distribution
    if 'category_distribution' in metadata:
        run_summary.append("📂 PAPERS BY CATEGORY")
        run_summary.append("-" * 40)
        for category, count in metadata['category_distribution'].items():
            limit_info = ""
            if 'category_limits_applied' in metadata and category in metadata['category_limits_applied']:
                limit = metadata['category_limits_applied'][category]
                if limit:
                    limit_info = f" (limit: {limit})"
            run_summary.append(f"{category}: {count} papers{limit_info}")
        run_summary.append("")
    
    # Sample papers with relevance scores
    papers = literature_data.get('papers', [])
    if papers:
        run_summary.append("📄 SAMPLE PAPERS (Top 10 by Relevance)")
        run_summary.append("-" * 40)
        for i, paper in enumerate(papers[:10]):
            run_summary.append(f"{i+1:2d}. PMID: {paper['pmid']} | Year: {paper['year']} | Score: {paper['relevance_score']:.3f}")
            run_summary.append(f"    Title: {paper['title']}")
            run_summary.append(f"    Journal: {paper['journal']}")
            run_summary.append("")
    
    # Check for exclude term violations
    run_summary.append("🚨 EXCLUDE TERM VIOLATION CHECK")
    run_summary.append("-" * 40)
    exclude_terms = search_config.get('exclude_terms', [])
    
    violations = []
    if papers and exclude_terms:
        for paper in papers[:20]:
            title_abstract = (paper['title'] + ' ' + paper.get('abstract', '')).lower()
            for exclude_term in exclude_terms:
                if exclude_term.lower() in title_abstract:
                    violations.append({
                        'pmid': paper['pmid'],
                        'title': paper['title'][:80] + '...',
                        'term': exclude_term,
                        'score': paper['relevance_score']
                    })
    
    if violations:
        run_summary.append("⚠️  WARNING: Found papers containing excluded terms!")
        for i, violation in enumerate(violations[:10]):
            run_summary.append(f"   {i+1}. PMID {violation['pmid']} (Score: {violation['score']:.3f})")
            run_summary.append(f"      Contains '{violation['term']}': {violation['title']}")
        run_summary.append("")
        run_summary.append(f"Total violations found: {len(violations)} out of top 20 papers checked")
    else:
        run_summary.append("✅ No exclude term violations found in top 20 papers")
    run_summary.append("")
    
    # Save to file
    summary_file = project_dir / "literature_fetch_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(run_summary))
    
    return summary_file


def move_raw_response_to_project(project_dir: Path):
    """Move any existing raw LLM response to project directory"""
    import glob
    raw_files = glob.glob("data/llm_raw_response_*.txt")
    if raw_files:
        latest_raw = max(raw_files, key=lambda f: Path(f).stat().st_mtime)
        destination = project_dir / "raw_responses" / Path(latest_raw).name
        if not destination.exists():
            Path(latest_raw).rename(destination)
        return destination
    return None


def fetch_literature(query_file_path: str):
    """Fetch literature using query file"""
    import time
    start_time = time.time()
    
    query_file = Path(query_file_path)
    if not query_file.exists():
        print(f"❌ Query file not found: {query_file_path}")
        return False
    
    print("📚 RAG System - Part 2: Literature Fetching")
    print("=" * 60)
    print(f"Query file: {query_file.name}")
    
    try:
        # Load query configuration
        with open(query_file, 'r') as f:
            query_data = json.load(f)
        
        # Create QueryResult object with required parameters
        query_result = QueryResult(
            topic=query_data.get("topic", ""),
            search_config=query_data,
            metadata={"topic": query_data.get("topic", "")},
            raw_response="Loaded from file",
            timestamp=datetime.now().isoformat()
        )
        
        # Create project directory
        project_dir, project_name = create_timestamped_project_dir(query_file)
        print(f"📁 Project directory: {project_dir}")
        
        # Copy query file and scripts to project directory
        query_dest = project_dir / query_file.name
        shutil.copy2(query_file, query_dest)
        
        script_dest = project_dir / "scripts" / Path(__file__).name
        shutil.copy2(Path(__file__), script_dest)
        
        # Move raw response if available
        raw_response_file = move_raw_response_to_project(project_dir)
        
        # Create literature configuration
        literature_config = create_literature_config()
        
        # Show query summary
        search_config = query_result.search_config.get('search_config', {})
        total_terms = sum(len(terms) for terms in search_config.values() if isinstance(terms, list))
        print(f"\n📝 Query Summary:")
        print(f"   Total terms: {total_terms}")
        for term_type, terms in search_config.items():
            if isinstance(terms, list):
                print(f"   {term_type}: {len(terms)} terms")
        
        # Fetch Literature
        print(f"\n📚 Fetching Literature from PubMed...")
        print(f"   Max total papers: {MAX_TOTAL_PAPERS}")
        print(f"   Max per query: {MAX_PAPERS_PER_QUERY}")
        print(f"   Recent years only: {INCLUDE_RECENT_YEARS or 'All years'}")
        
        fetcher = LiteratureFetcher(literature_config)
        literature_data = fetcher.fetch_literature(query_result)
        
        # Save literature data
        literature_file = project_dir / f"{project_name}_literature.json"
        fetcher.save_literature(literature_data, str(literature_file))
        
        # Show literature summary
        metadata = literature_data['metadata']
        print(f"\n✅ Literature fetched:")
        print(f"   Papers found: {metadata['total_papers']}")
        print(f"   Queries executed: {len(literature_data['query_results'])}")
        print(f"   Fetch time: {metadata['fetch_duration']:.1f}s")
        print(f"   Average relevance: {metadata['average_relevance']:.2f}")

        # Show category distribution if available
        if 'category_distribution' in metadata:
            print(f"\n📂 Papers by category:")
            for category, count in metadata['category_distribution'].items():
                limit_info = ""
                if 'category_limits_applied' in metadata and category in metadata['category_limits_applied']:
                    limit = metadata['category_limits_applied'][category]
                    if limit:
                        limit_info = f" (limit: {limit})"
                print(f"   {category}: {count}{limit_info}")
        
        # Create detailed run summary
        print(f"\n📊 Creating Detailed Run Summary...")
        summary_file = save_run_summary(project_dir, query_result, literature_data, fetcher, query_file_path, start_time, literature_config)
        
        print(f"\n✅ Literature fetching completed!")
        print(f"\n📁 All files saved to: {project_dir}")
        print(f"   📄 Literature data: {literature_file.name}")
        print(f"   📋 Query file: {query_dest.name}")
        print(f"   📊 Detailed summary: {summary_file.name}")
        print(f"   🔧 Script used: {script_dest.name}")
        if raw_response_file:
            print(f"   📝 LLM raw response: {raw_response_file.name}")
        
        # Show sample papers
        if literature_data['papers']:
            print(f"\n📄 Sample papers fetched:")
            for i, paper in enumerate(literature_data['papers'][:3]):
                print(f"   {i+1}. {paper['title'][:70]}...")
                print(f"      PMID: {paper['pmid']}, {paper['year']}, Relevance: {paper['relevance_score']:.2f}")
        
        print(f"\n🔧 Next Steps:")
        print(f"1. Review literature in {literature_file.name}")
        print(f"2. Check detailed summary in {summary_file.name}")
        print(f"3. Run Part 3: python rag_part3_build_rag.py {literature_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Literature fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    
    if len(sys.argv) != 2:
        print("Usage: python rag_part2_fetch_literature.py <query_file.json>")
        print("Example: python rag_part2_fetch_literature.py gene_reg_evolution_queries_20250901_123456.json")
        return 1
    
    query_file = sys.argv[1]
    
    # Fetch literature
    success = fetch_literature(query_file)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

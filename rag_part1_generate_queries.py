#!/usr/bin/env python3
"""
RAG System - Part 1: Query Generation
Generates search queries using LLM and saves to query.json for manual editing.
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Load environment (only for API keys)
from run_with_env import load_env_file
load_env_file()

from core.query_generator import QueryGenerator, QueryGenerationConfig

# ============================================================================
# 🎯 PROJECT CONFIGURATION
# ============================================================================

# Project identification
PROJECT_NAME = "DYRK1B"        # Used for directory naming
RESEARCH_TOPIC = "DYRK1B"
RESEARCH_DOMAIN = "biology"               # biology, medicine, ai, physics, chemistry

# Search term customization
INCLUDE_TERMS = []

EXCLUDE_TERMS = [
    "commercial applications",
    "patent"]

# Custom instructions for LLM query generation
CUSTOM_INSTRUCTIONS = """
Focus on basic biology, mechanisms and biological processes, and cancer studies.
"""

# LLM settings
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLM_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
LLM_TEMPERATURE = 0.3 # Lower = more focused, higher = more creative
LLM_MAX_TOKENS = 10000 # Maximum response length

# ============================================================================
# 🔧 IMPLEMENTATION
# ============================================================================

def create_query_config():
    """Create LLM configuration"""
    query_config = QueryGenerationConfig()
    query_config.api_key = os.getenv('DEEPINFRA_API_KEY')
    query_config.model_name = LLM_MODEL
    query_config.temperature = LLM_TEMPERATURE
    query_config.max_tokens = LLM_MAX_TOKENS
    query_config.timeout = 120
    
    if not query_config.api_key:
        raise ValueError("DEEPINFRA_API_KEY not found in environment. Check your .env file.")
    
    return query_config


def create_timestamped_output():
    """Create timestamped output filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"queries/{PROJECT_NAME}_queries_{timestamp}.json"
    return output_file


def generate_queries():
    """Generate queries using LLM and save to JSON"""
    
    print("🔍 RAG System - Part 1: Query Generation")
    print("=" * 60)
    print(f"Project: {PROJECT_NAME}")
    print(f"Topic: {RESEARCH_TOPIC}")
    print(f"Domain: {RESEARCH_DOMAIN}")
    
    try:
        # Create configuration
        query_config = create_query_config()
        
        # Generate queries
        print(f"\n📝 Generating Search Queries...")
        print(f"   LLM Model: {LLM_MODEL}")
        print(f"   Include terms: {len(INCLUDE_TERMS)}")
        print(f"   Exclude terms: {len(EXCLUDE_TERMS)}")
        
        generator = QueryGenerator(query_config)
        query_result = generator.generate_queries(
            topic=RESEARCH_TOPIC,
            domain=RESEARCH_DOMAIN,
            include_terms=INCLUDE_TERMS,
            exclude_terms=EXCLUDE_TERMS,
            custom_instructions=CUSTOM_INSTRUCTIONS.strip() or None,
            save_raw_response=True
        )
        
        # Show generated terms summary
        search_config = query_result.search_config.get('search_config', {})
        print(f"\n✅ Queries generated:")
        total_terms = 0
        for term_type, terms in search_config.items():
            if isinstance(terms, list):
                print(f"   {term_type}: {len(terms)} terms")
                total_terms += len(terms)
        print(f"   Total terms: {total_terms}")
        
        # Save query configuration
        output_file = create_timestamped_output()
        generator.save_query_config(query_result, str(Path.cwd() / output_file))
        
        print(f"\n✅ Query generation completed!")
        print(f"📄 Saved to: {output_file}")
        print(f"\n🔧 Next Steps:")
        print(f"1. Review and edit the generated queries in: {output_file}")
        print(f"2. Adjust term categories, add/remove terms as needed")
        print(f"3. Run Part 2: python rag_part2_fetch_literature.py {output_file}")
        print(f"\n💡 The query file contains:")
        print(f"   • Core terms: Main research concepts")
        print(f"   • Entity terms: Specific genes, proteins, molecules")
        print(f"   • Method terms: Research techniques and methods")
        print(f"   • Context terms: Research domains and applications")
        print(f"   • Synonym terms: Alternative terminology")
        print(f"   • Exclude terms: Terms to avoid in literature")
        
        return True
        
    except Exception as e:
        print(f"❌ Query generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    
    # Check environment
    if not os.getenv('DEEPINFRA_API_KEY'):
        print("❌ No DEEPINFRA_API_KEY found!")
        print("   Make sure your .env file contains your DeepInfra API key")
        return 1
    
    if not RESEARCH_TOPIC.strip():
        print("❌ Please set RESEARCH_TOPIC in the configuration section")
        return 1
    
    # Generate queries
    success = generate_queries()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

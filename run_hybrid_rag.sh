#!/bin/bash
# Script to run the hybrid RAG pipeline

echo "Running Hybrid RAG Pipeline with Semantic Sections"
echo "===================================================="
echo ""

# Change to the semantic sections directory
cd /home/harl/Dropbox/manuscripts/0.datasets_visualizations/002.AI_projects/RAG/generalized_rag_pipeline_semantic_sections

# Check if we're in the right directory
echo "Current directory: $(pwd)"
echo ""

# Run the RAG pipeline with the literature file
if [ -f "results/gene_reg_evolution_queries_20250902_135629/gene_reg_evolution_queries_literature.json" ]; then
    echo "Starting hybrid RAG construction..."
    python3 rag_part3_build_rag.py results/gene_reg_evolution_queries_20250902_135629/gene_reg_evolution_queries_literature.json
else
    echo "Literature file not found. Please ensure it exists at:"
    echo "results/gene_reg_evolution_queries_20250902_135629/gene_reg_evolution_queries_literature.json"
fi
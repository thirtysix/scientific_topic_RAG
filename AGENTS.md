# AGENTS.md

Orientation for AI coding agents (and humans) arriving at this repo: what it is, how to run it, and the conventions worth knowing before making changes.

## What this is
A generalized, domain-agnostic pipeline for building Retrieval-Augmented Generation systems from scientific literature. It runs as three independent, editable stages: LLM-based query generation, PubMed literature fetching with relevance scoring, and RAG construction (chunking, embeddings, vector database). Configure it for any research field.

## Stack & layout
- **Language**: Python 3.8+ (GPU optional; CPU mode supported).
- **Stage scripts**: `rag_part1_generate_queries.py`, `rag_part2_fetch_literature.py` (plus a MeSH variant), `rag_part3_build_rag_optimized.py`.
- **Core modules**: `core/` (`query_generator.py`, `literature_fetcher.py`), `config/pipeline_config.py` (dataclass-based configs), `utils/logger.py`.
- **Key deps**: sentence-transformers, transformers, torch, chromadb (default) or faiss, rank-bm25, beautifulsoup4, python-dotenv.
- **Default branch** is `master`.

## Run, test, lint
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python rag_part1_generate_queries.py                        # -> queries/<project>_<ts>.json
python rag_part2_fetch_literature.py queries/<file>.json    # -> results/<project>_<ts>/..._literature.json
python rag_part3_build_rag_optimized.py results/<project>_<ts>/<project>_literature.json
```
Each stage takes the previous stage's output file as an argument, so you can inspect and hand-edit the intermediate JSON before continuing.

## Conventions
- Per-run configuration lives in the constants at the top of each stage script (project name, research topic, paper limits, chunking strategy, embedding model, device). `config/pipeline_config.py` holds the dataclass configs with biology / medicine / AI presets.
- Stage 3 writes 8+ output formats into `rag_system/` (ChromaDB/FAISS, OpenAI, LangChain, CSV, TXT, hybrid-reranking, plus a runnable `example_usage.py`).
- MIT licensed.

## Gotchas
- Copy `.env.sample` to `.env` and set `DEEPINFRA_API_KEY` (required for query generation) and `EMAIL` (required for the PubMed API); `NCBI_API_KEY` is optional for higher rate limits.
- On CUDA out-of-memory, lower the batch size, enable FP16, or switch to a smaller embedding model / CPU (all toggled in the stage-3 constants).
- `queries/`, `data/`, `results/`, `cache/`, `temp/`, and `venv/` are generated or gitignored; `old/` holds archived legacy scripts, not the current pipeline.

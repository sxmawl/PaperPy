# PaperPy

A chat with research paper application used to explain comprehensive evaluation system for RAG (Retrieval-Augmented Generation) systems using synthetic question-answer pairs generated from PDF documents.

## Features

- **Synthetic Data Generation**: Auto-generates QA pairs from PDFs using GPT-4o-mini
- **RAG System Evaluation**: Tests your RAG system against generated QA pairs
- **Comprehensive Metrics**: Retrieval accuracy, answer quality, and response time
- **Multiple Chunking Strategies**: Fixed character, recursive (LangChain), and semantic chunking

## Files Structure

```
evals/
├── data/                          # PDF documents for evaluation
│   ├── *.pdf                      # Your PDF files
├── output/                        # Generated datasets and results
│   ├── synthetic_qa_dataset.json  # Full synthetic dataset
│   ├── synthetic_qa_quick.json    # Quick evaluation dataset
│   └── evaluation_results.json    # Evaluation results
├── synthetic_data.py              # QA pair generation
├── evaluate_rag.py                # RAG system evaluation
├── run_evaluation.py              # Complete pipeline runner
└── README.md                      # This file
```

## Quick Start

### 1. Setup

Install dependencies from the existing lock file:
```bash
uv sync
```

Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=your_postgresql_connection_string
```

### 2. Generate Synthetic Data

**Quick generation (5 chunks per doc, 2 questions per chunk):**
```bash
uv run evals/synthetic_data.py --quick
```

**Full generation (20 chunks per doc, 3 questions per chunk):**
```bash
uv run evals/synthetic_data.py --full
```

**With specific chunker:**
```bash
# Recursive chunking (default, uses LangChain)
uv run evals/synthetic_data.py --quick --chunker recursive

# Fixed character chunking
uv run evals/synthetic_data.py --quick --chunker fixed_character

# Semantic chunking
uv run evals/synthetic_data.py --quick --chunker semantic
```

### 3. Run Evaluation

**Quick evaluation:**
```bash
uv run evals/run_evaluation.py --quick
```

**Full evaluation:**
```bash
uv run evals/run_evaluation.py --full
```

**With additional options:**
```bash
# Force regenerate QA pairs even if they exist
uv run evals/run_evaluation.py --quick --force-regenerate

# Use specific chunker type
uv run evals/run_evaluation.py --quick --chunker semantic

# Set number of top documents to retrieve
uv run evals/run_evaluation.py --quick --top-k 5

# Ingest documents into vector store during evaluation
uv run evals/run_evaluation.py --quick --ingest

# Combine multiple options
uv run evals/run_evaluation.py --full --chunker recursive --top-k 5 --force-regenerate --ingest
```

## Command Line Arguments

The `run_evaluation.py` script supports the following arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--quick` | flag | - | Run quick evaluation with limited data (1 chunk per doc, 1 question per chunk) |
| `--full` | flag | - | Run full evaluation with comprehensive data (20 chunks per doc, 3 questions per chunk) |
| `--chunker` | string | `fixed_character` | Type of chunker to use: `fixed_character`, `recursive`, or `semantic` |
| `--force-regenerate` | flag | - | Force regeneration of synthetic QA pairs even if they already exist |
| `--top-k` | integer | `3` | Number of top documents to retrieve during RAG evaluation |
| `--ingest` | flag | - | Ingest documents into vector store during evaluation |

**Note:** If neither `--quick` nor `--full` is specified, the script defaults to quick evaluation.

## How It Works

### 1. Synthetic Data Generation
- **Text Extraction**: Extracts text from PDF files
- **Text Chunking**: Splits text using configurable strategies (fixed, recursive, semantic)
- **QA Generation**: Uses GPT-4o-mini to generate diverse question-answer pairs
- **Dataset Creation**: Saves structured QA pairs with metadata

### 2. RAG Evaluation
- **Question Processing**: Sends questions to your RAG system
- **Answer Quality Assessment**: Uses GPT-4o-mini to evaluate answer quality
- **Retrieval Accuracy**: Measures context retrieval performance
- **Performance Metrics**: Calculates response times and overall scores

**Metrics:**
- Retrieval Score: How well the system finds relevant information
- Answer Quality Score: Accuracy and completeness of answers
- Response Time: System performance
- Question Type Breakdown: Performance across different question types

## Text Chunking Strategies

### 1. Fixed Character Chunker (`fixed_character`)
- Splits text into fixed-size chunks
- Tries to break at sentence boundaries
- Fastest and most predictable

### 2. Recursive Text Splitter (`recursive`) - **DEFAULT**
- Uses LangChain's `RecursiveCharacterTextSplitter`
- Applies separators in order: paragraphs → lines → sentences → words → characters
- Enhanced multilingual support
- **Recommended for most use cases**

### 3. Semantic Chunker (`semantic`)
- Uses embeddings to find semantic break points
- Maintains semantic coherence within chunks
- Most computationally expensive

## Output Files

- **`synthetic_qa_dataset.json`**: Complete dataset with all QA pairs
- **`synthetic_qa_quick.json`**: Smaller dataset for quick testing
- **`evaluation_results.json`**: Detailed evaluation results

## Performance Assessment

- **EXCELLENT** (≥0.8): Outstanding performance
- **GOOD** (≥0.6): Solid performance with room for improvement
- **FAIR** (≥0.4): Acceptable but needs optimization
- **NEEDS IMPROVEMENT** (<0.4): Significant improvements required

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check API key, credits, and rate limits
2. **PDF Processing**: Ensure PDFs are text-based (not scanned images)
3. **Database Connection**: Verify PostgreSQL connection and pgvector extension
4. **Semantic Chunker**: Requires OpenAI API key, more expensive

### Debug Mode
```python
logging.basicConfig(level=logging.DEBUG)
```

Or run with debug logging:
```bash
uv run evals/run_evaluation.py --quick --chunker semantic
```


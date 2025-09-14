"""
Synthetic Data Generation for RAG Evaluation

This module generates synthetic question-answer pairs from PDF documents for
evaluating RAG (Retrieval-Augmented Generation) systems.

The process:
1. Extract text from PDF documents
2. Chunk text into manageable segments
3. Generate QA pairs using GPT-4o-mini
4. Save results in structured format for evaluation
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import PyPDF2
import io
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class QAPair:
    """Represents a single question-answer pair with context."""
    question: str
    answer: str
    context: str
    source_document: str
    chunk_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationDataset:
    """Complete evaluation dataset containing multiple QA pairs."""
    qa_pairs: List[QAPair]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "qa_pairs": [asdict(qa) for qa in self.qa_pairs],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        """Create from dictionary."""
        qa_pairs = [QAPair(**qa) for qa in data["qa_pairs"]]
        return cls(qa_pairs=qa_pairs, metadata=data["metadata"])


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.chunker import ChunkerFactory, Chunk
from agents import Agent

class PDFProcessor:
    """Handles PDF text extraction and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, chunker_type: str = "fixed_character"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker_type = chunker_type
        
        # Use LangChain recursive splitter for better text processing
        if chunker_type == "recursive":
            self.chunker = ChunkerFactory.create_langchain_recursive_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            self.chunker = ChunkerFactory.create_chunker(
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                all_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    print(f"Page {page_num + 1}")
                    page_text = page.extract_text()
                    all_text += page_text + "\n"

                return all_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into chunks using the configured chunker.
        
        Returns:
            List of tuples (chunk_id, chunk_text)
        """
        chunks = self.chunker.chunk_text(text)
        return [(chunk.chunk_id, chunk.text) for chunk in chunks]


class QAGenerator:
    """Generates QA pairs using GPT-4o-mini."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
    
    async def generate_qa_pairs(
        self, 
        chunk_text: str, 
        chunk_id: str, 
        source_document: str,
        num_questions: int = 3
    ) -> List[QAPair]:
        """
        Generate QA pairs from a text chunk.
        
        Args:
            chunk_text: The text chunk to generate questions from
            chunk_id: Identifier for the chunk
            source_document: Name of the source document
            num_questions: Number of questions to generate per chunk
            
        Returns:
            List of QA pairs
        """

        # PROMPT FOR QA GENERATION
        prompt = f"""
            You are an expert at creating high-quality question-answer pairs for evaluating RAG (Retrieval-Augmented Generation) systems.

            Given the following text chunk, create {num_questions} diverse question-answer pairs that:
            1. Can be answered using ONLY the information in the provided text
            2. Cover different aspects of the content (factual, conceptual, analytical)
            3. Have clear, unambiguous answers
            4. Test the system's ability to retrieve and synthesize information

            Text chunk:
            {chunk_text}

            Generate {num_questions} QA pairs in the following JSON format:
            {{
                "qa_pairs": [
                    {{
                        "question": "A clear, specific question about the text",
                        "answer": "A comprehensive answer that can be derived from the text",
                        "question_type": "factual|conceptual|analytical"
                    }}
                ]
            }}

            Focus on questions that require understanding of the text content, not just keyword matching.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating evaluation datasets for RAG systems."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse structured output
            try:
                content = response.choices[0].message.content
                if not content:
                    return []
                
                data = json.loads(content)
                qa_pairs = []
                
                for qa_data in data.get("qa_pairs", []):
                    qa_pair = QAPair(
                        question=qa_data["question"],
                        answer=qa_data["answer"],
                        context=chunk_text,
                        source_document=source_document,
                        chunk_id=chunk_id,
                        metadata={
                            "question_type": qa_data.get("question_type", "factual"),
                            "generated_by": self.model
                        }
                    )
                    qa_pairs.append(qa_pair)
                
                return qa_pairs
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse structured output for chunk {chunk_id}: {e}")
                logger.debug(f"Response content: {content}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating QA pairs for chunk {chunk_id}: {e}")
            return []


class SyntheticDataGenerator:
    """Main class for generating synthetic evaluation data."""
    
    def __init__(
        self,
        data_dir: str = "evals/data",
        output_dir: str = "evals/output",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        questions_per_chunk: int = 3,
        max_chunks_per_doc: Optional[int] = None,
        chunker_type: str = "fixed_character",
        ingest_to_vector_store: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.questions_per_chunk = questions_per_chunk
        self.max_chunks_per_doc = max_chunks_per_doc
        self.chunker_type = chunker_type
        self.ingest_to_vector_store = ingest_to_vector_store
        
        # Initialize components
        self.pdf_processor = PDFProcessor(chunk_size, chunk_overlap, chunker_type)
        self.qa_generator = QAGenerator()
        
        # Initialize agent for vector store ingestion if needed
        if self.ingest_to_vector_store:
            self.agent = Agent()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the data directory."""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    async def process_document(self, pdf_path: Path) -> List[QAPair]:
        """Process a single PDF document and generate QA pairs."""
        logger.info(f"Processing document: {pdf_path.name}")
        
        # Extract text
        text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return []
        
        # Get ALL chunks from the document (for ingestion)
        all_chunks = self.pdf_processor.chunk_text(text)
        logger.info(f"Created {len(all_chunks)} total chunks from {pdf_path.name}")
        
        # Ingest ALL chunks into vector store if enabled (ignore max_chunks_per_doc)
        if self.ingest_to_vector_store:
            logger.info(f"Ingesting ALL {len(all_chunks)} chunks from {pdf_path.name} into vector store...")
            source_name = pdf_path.stem  # Use filename without extension as source
            
            for chunk_id, chunk_text in tqdm(all_chunks, desc=f"Ingesting all chunks from {pdf_path.name}"):
                try:
                    vector_store_id = self.agent.store_knowledge(
                        text=chunk_text,
                        source=source_name
                    )
                    logger.debug(f"Stored chunk {chunk_id} with vector store ID: {vector_store_id}")
                except Exception as e:
                    logger.error(f"Error storing chunk {chunk_id}: {e}")
                    continue
        
        # Limit chunks for QA generation only (max_chunks_per_doc only affects QA generation)
        qa_chunks = all_chunks
        if self.max_chunks_per_doc:
            qa_chunks = all_chunks[:self.max_chunks_per_doc]
            logger.info(f"Using {len(qa_chunks)} chunks for QA generation (limited by max_chunks_per_doc)")
        else:
            logger.info(f"Using all {len(qa_chunks)} chunks for QA generation")
        
        # Generate QA pairs for limited chunks only
        all_qa_pairs = []
        
        for chunk_id, chunk_text in tqdm(qa_chunks, desc=f"Generating QA pairs for {pdf_path.name}"):
            qa_pairs = await self.qa_generator.generate_qa_pairs(
                chunk_text=chunk_text,
                chunk_id=chunk_id,
                source_document=pdf_path.name,
                num_questions=self.questions_per_chunk
            )
            all_qa_pairs.extend(qa_pairs)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        logger.info(f"Generated {len(all_qa_pairs)} QA pairs from {pdf_path.name}")
        return all_qa_pairs
    
    async def generate_dataset(self) -> EvaluationDataset:
        """Generate the complete evaluation dataset."""
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.data_dir}")
        
        all_qa_pairs = []
        
        for pdf_file in pdf_files:
            try:
                qa_pairs = await self.process_document(pdf_file)
                all_qa_pairs.extend(qa_pairs)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        # Create metadata
        metadata = {
            "total_qa_pairs": len(all_qa_pairs),
            "source_documents": [f.name for f in pdf_files],
            "generation_config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunker_type": self.chunker_type,
                "questions_per_chunk": self.questions_per_chunk,
                "max_chunks_per_doc": self.max_chunks_per_doc,
                "model": self.qa_generator.model
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return EvaluationDataset(qa_pairs=all_qa_pairs, metadata=metadata)
    
    def save_dataset(self, dataset: EvaluationDataset, filename: str = "synthetic_qa_dataset.json"):
        """Save the dataset to a JSON file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Total QA pairs: {len(dataset.qa_pairs)}")
        
        # Print summary statistics
        question_types = {}
        for qa in dataset.qa_pairs:
            q_type = qa.metadata.get("question_type", "unknown")
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        logger.info("Question type distribution:")
        for q_type, count in question_types.items():
            logger.info(f"  {q_type}: {count}")
    

    
    def verify_vector_store_ingestion(self, sample_query: str = "test") -> Dict[str, Any]:
        """Verify that data was ingested correctly by performing a test search."""
        if not hasattr(self, 'agent'):
            return {"error": "Agent not initialized"}
        
        try:
            results = self.agent.search_knowledge(sample_query, top_k=5)
            
            # Check if sources are properly set
            sources = [result.get("source", "unknown") for result in results]
            unique_sources = set(sources)
            
            verification_result = {
                "total_results": len(results),
                "sources_found": list(unique_sources),
                "unknown_sources": sources.count("unknown"),
                "sample_results": results[:2]  # Show first 2 results for inspection
            }
            
            logger.info(f"Vector store verification results: {verification_result}")
            return verification_result
            
        except Exception as e:
            logger.error(f"Error verifying vector store ingestion: {e}")
            return {"error": str(e)}


async def main():
    """Main function to run the synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs from PDF documents")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents into vector store during generation"
    )
    parser.add_argument(
        "--chunker",
        type=str,
        default="recursive",
        choices=["fixed_character", "recursive", "semantic"],
        help="Type of chunker to use for text processing"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=2,
        help="Maximum chunks per document for QA generation (default: 2)"
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=1,
        help="Number of questions to generate per chunk (default: 1)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(
        data_dir="evals/data",
        output_dir="evals/output",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        questions_per_chunk=args.questions_per_chunk,
        max_chunks_per_doc=args.max_chunks,
        chunker_type=args.chunker,
        ingest_to_vector_store=args.ingest
    )
    
    try:
        # Generate dataset (this will also ingest chunks into vector store if enabled)
        if args.ingest:
            logger.info("Starting synthetic data generation with vector store ingestion...")
        else:
            logger.info("Starting synthetic data generation (no vector store ingestion)...")
            
        dataset = await generator.generate_dataset()
        generator.save_dataset(dataset)
        
        # Verify vector store ingestion if enabled
        if args.ingest:
            logger.info("Verifying vector store ingestion...")
            verification = generator.verify_vector_store_ingestion()
            
            if verification.get("unknown_sources", 0) > 0:
                logger.warning(f"Found {verification['unknown_sources']} results with 'unknown' sources")
            else:
                logger.info("All results have proper source information")
        
        # Also save a smaller sample for quick testing
        if len(dataset.qa_pairs) > 50:
            sample_dataset = EvaluationDataset(
                qa_pairs=dataset.qa_pairs[:50],
                metadata={**dataset.metadata, "is_sample": True, "sample_size": 50}
            )
            generator.save_dataset(sample_dataset, "synthetic_qa_sample.json")
            
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Complete RAG Evaluation Pipeline

This script runs the entire evaluation pipeline:
1. Generate synthetic QA pairs from PDF documents
2. Evaluate the RAG system using the generated pairs
3. Generate a comprehensive report

Usage:
    python evals/run_evaluation.py [--quick] [--full]
"""

import asyncio
import argparse
import logging
from pathlib import Path
import sys
import os
import json

# Add parent directory to path for imports
sys.path.append('..')

from synthetic_data import SyntheticDataGenerator, EvaluationDataset
from evaluate_rag import RAGEvaluator
from agents import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dataset_exists(output_dir: str, filename: str) -> bool:
    """
    Check if a dataset file already exists in the output directory.
    
    Args:
        output_dir: Output directory path
        filename: Dataset filename to check
        
    Returns:
        True if dataset exists and is valid, False otherwise
    """
    dataset_path = Path(output_dir) / filename
    
    if not dataset_path.exists():
        logger.info(f"Dataset {filename} not found at {dataset_path}")
        return False
    
    try:
        # Try to load and validate the dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it has the expected structure
        if "qa_pairs" not in data or "metadata" not in data:
            logger.warning(f"Dataset {filename} exists but has invalid structure")
            return False
        
        qa_pairs = data["qa_pairs"]
        if not qa_pairs:
            logger.warning(f"Dataset {filename} exists but contains no QA pairs")
            return False
        
        logger.info(f"Found existing dataset {filename} with {len(qa_pairs)} QA pairs")
        return True
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"Dataset {filename} exists but is corrupted: {e}")
        return False


def load_existing_dataset(output_dir: str, filename: str) -> EvaluationDataset:
    """
    Load an existing dataset from the output directory.
    
    Args:
        output_dir: Output directory path
        filename: Dataset filename to load
        
    Returns:
        EvaluationDataset object
    """
    dataset_path = Path(output_dir) / filename
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return EvaluationDataset.from_dict(data)


async def run_full_evaluation(chunker_type: str = "fixed_character", force_regenerate: bool = False, top_k: int = 3, ingest: bool = False):
    """Run the complete evaluation pipeline."""
    logger.info("Starting full RAG evaluation pipeline...")
    
    output_dir = "evals/output"
    dataset_filename = "synthetic_qa_dataset.json"
    
    # Step 1: Check if dataset already exists
    logger.info("Step 1: Checking for existing synthetic QA pairs...")
    if not force_regenerate and check_dataset_exists(output_dir, dataset_filename):
        logger.info("Using existing dataset - skipping generation")
        dataset = load_existing_dataset(output_dir, dataset_filename)
    else:
        if force_regenerate:
            logger.info("Force regenerate flag set - generating new dataset")
        logger.info("Generating new synthetic QA pairs...")
        generator = SyntheticDataGenerator(
            data_dir="evals/data",
            output_dir=output_dir,
            chunk_size=1000,
            chunk_overlap=200,
            questions_per_chunk=3,
            max_chunks_per_doc=20,  # More chunks for full evaluation
            chunker_type=chunker_type,
            ingest_to_vector_store=ingest  # Use command line argument
        )
        
        dataset = await generator.generate_dataset()
        generator.save_dataset(dataset, dataset_filename)
    
    # Step 2: Evaluate RAG system
    logger.info(f"Step 2: Evaluating RAG system with top_k={top_k}...")
    agent = Agent()
    evaluator = RAGEvaluator(agent, default_top_k=top_k)
    
    dataset_path = "evals/output/synthetic_qa_dataset.json"
    results = await evaluator.evaluate_dataset(dataset_path)
    
    # Step 3: Calculate and save results
    logger.info("Step 3: Calculating evaluation results...")
    summary = evaluator.calculate_summary(results)
    evaluator.save_results(results, summary, f"evals/output/full_evaluation_results_topk{top_k}.json")
    
    # Step 4: Print tabular results
    logger.info("Step 4: Printing tabular results...")
    evaluator.print_tabular_results(results)
    
    logger.info("Full evaluation completed successfully!")
    return summary


async def run_quick_evaluation(chunker_type: str = "fixed_character", force_regenerate: bool = False, top_k: int = 3, ingest: bool = False):
    """Run a quick evaluation with limited data."""
    logger.info("Starting quick RAG evaluation...")
    
    output_dir = "evals/output"
    dataset_filename = "synthetic_qa_dataset.json"
    
    # Step 1: Check if dataset already exists
    logger.info("Step 1: Checking for existing quick synthetic QA pairs...")
    if not force_regenerate and check_dataset_exists(output_dir, dataset_filename):
        logger.info("Using existing quick dataset - skipping generation")
        dataset = load_existing_dataset(output_dir, dataset_filename)
    else:
        if force_regenerate:
            logger.info("Force regenerate flag set - generating new quick dataset")
        logger.info("Generating new quick synthetic QA pairs...")
        generator = SyntheticDataGenerator(
            data_dir="evals/data",
            output_dir=output_dir,
            chunk_size=1000,
            chunk_overlap=200,
            questions_per_chunk=1,
            max_chunks_per_doc=1,  # Limited chunks for quick evaluation
            chunker_type=chunker_type,
            ingest_to_vector_store=ingest  # Use command line argument
        )
        
        dataset = await generator.generate_dataset()
        generator.save_dataset(dataset, dataset_filename)
    
    # Step 2: Evaluate RAG system
    logger.info(f"Step 2: Evaluating RAG system with top_k={top_k}...")
    agent = Agent()
    evaluator = RAGEvaluator(agent, default_top_k=top_k)
    
    dataset_path = "evals/output/synthetic_qa_dataset.json"
    results = await evaluator.evaluate_dataset(dataset_path)
    
    # Step 3: Calculate and save results
    logger.info("Step 3: Calculating evaluation results...")
    summary = evaluator.calculate_summary(results)
    evaluator.save_results(results, summary, f"evals/output/quick_evaluation_results_topk{top_k}.json")
    
    # Step 4: Print tabular results
    logger.info("Step 4: Printing tabular results...")
    evaluator.print_tabular_results(results)
    
    logger.info("Quick evaluation completed successfully!")
    return summary


def print_evaluation_summary(summary):
    """Print a formatted evaluation summary."""
    print("\n" + "="*60)
    print("RAG EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions Evaluated: {summary.total_questions}")
    print(f"Overall System Score: {summary.overall_score:.3f}/1.0")
    print(f"Average F1 Score: {summary.average_f1_score:.3f}/1.0")
    print(f"Average LLM as Judge Score: {summary.average_llm_as_judge_score:.3f}/1.0")
    print(f"Average Mean Cosine Score: {summary.average_mean_cosine_score:.3f}/1.0")
    print(f"Average Latency: {summary.average_latency:.3f} seconds")
    
    print("\nQuestion Type Performance:")
    print("-" * 40)
    for q_type, breakdown in summary.question_type_breakdown.items():
        print(f"{q_type.upper()}:")
        print(f"  Count: {breakdown['count']}")
        print(f"  F1 Score: {breakdown['avg_f1_score']:.3f}/1.0")
        print(f"  LLM as Judge Score: {breakdown['avg_llm_as_judge_score']:.3f}/1.0")
        print(f"  Latency: {breakdown['avg_latency']:.3f}s")
        print()
    
    # Performance assessment
    if summary.overall_score >= 0.8:
        performance = "EXCELLENT"
    elif summary.overall_score >= 0.6:
        performance = "GOOD"
    elif summary.overall_score >= 0.4:
        performance = "FAIR"
    else:
        performance = "NEEDS IMPROVEMENT"
    
    print(f"Overall Performance: {performance}")
    print("="*60)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick evaluation with limited data"
    )
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Run full evaluation with comprehensive data"
    )
    parser.add_argument(
        "--chunker",
        type=str,
        default="fixed_character",
        choices=["fixed_character", "recursive", "semantic"],
        help="Type of chunker to use for text processing"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of synthetic QA pairs even if they already exist"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top documents to retrieve (default: 3)"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents into vector store during evaluation"
    )
    
    args = parser.parse_args()
    
    # Validate top_k
    if args.top_k < 1:
        logger.error("top-k must be at least 1")
        return
    
    # Check if output directory exists
    output_dir = Path("evals/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.quick:
            summary = await run_quick_evaluation(args.chunker, args.force_regenerate, args.top_k, args.ingest)
        elif args.full:
            summary = await run_full_evaluation(args.chunker, args.force_regenerate, args.top_k, args.ingest)
        else:
            # Default to quick evaluation
            logger.info("No mode specified, running quick evaluation...")
            summary = await run_quick_evaluation(args.chunker, args.force_regenerate, args.top_k, args.ingest)
        
        print_evaluation_summary(summary)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 
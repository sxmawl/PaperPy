import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import time
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from tabulate import tabulate

# Import your RAG system
import sys
sys.path.append('..')
from agents import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class EvaluationResult:
    """Represents the evaluation result for a single QA pair."""
    question: str
    expected_answer: str
    actual_answer: str
    expected_correct_chunks: str
    sources: List[str]
    chunk_ids: List[str]
    f1_score: float
    precision_at_k: float
    recall_at_k: float
    mean_cosine_score: float
    llm_as_judge_score: float
    latency: float
    retrieved_chunks: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""
    total_questions: int
    average_f1_score: float
    average_llm_as_judge_score: float
    average_latency: float
    average_mean_cosine_score: float
    question_type_breakdown: Dict[str, Dict[str, float]]
    overall_score: float


class RAGEvaluator:
    """Evaluates RAG system performance using synthetic QA pairs."""
    
    def __init__(self, agent: Agent, openai_api_key: Optional[str] = None, default_top_k: int = 3):
        self.agent = agent
        self.openai_client = AsyncOpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.default_top_k = default_top_k
    
    async def evaluate_answer_quality(
        self, 
        question: str, 
        expected_answer: str, 
        actual_answer: str
    ) -> float:
        """
        Evaluate the quality of the generated answer compared to the expected answer.
        Returns a score between 0 and 1.
        """
        prompt = (
            "You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.\n\n"
            "Given a question and two answers (expected and actual), rate how well the actual answer addresses the question compared to the expected answer.\n\n"
            f"Question: {question}\n\n"
            f"Expected Answer: {expected_answer}\n\n"
            f"Actual Answer: {actual_answer}\n\n"
            "Rate the actual answer on a scale of 0 to 1, where:\n"
            "- 0: Completely incorrect or irrelevant\n"
            "- 0.5: Partially correct but missing key information\n"
            "- 1: Fully correct and comprehensive\n\n"
            "Consider:\n"
            "1. Accuracy of information\n"
            "2. Completeness of the answer\n"
            "3. Relevance to the question\n"
            "4. Clarity and coherence\n\n"
            "Return only a number between 0 and 1 (e.g., 0.85).\n\n"
            "Respond ONLY with a valid JSON object on a single line, with the following fields:\n"
            "- score: A number between 0 and 1\n"
            "- explanation: A short explanation of the score\n"
            "Example: {\"score\": 0.85, \"explanation\": \"The answer is mostly correct but misses some details.\"}"
        )

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for RAG systems."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content

            # Try to parse as JSON, but if it fails, try to extract the JSON substring
            try:
                data = json.loads(content)
                score = data.get("score", 0.5)
                return max(0.0, min(1.0, float(score)))  # Clamp between 0 and 1
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Could not parse structured output: {e}")
                logger.debug(f"Response content: {content}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error evaluating answer quality: {e}")
            return 0.5
    
    async def evaluate_retrieval_accuracy(
        self, 
        expected_context: str, 
        retrieved_context: str
    ) -> tuple[float, float, float]:
        """
        Evaluate how well the retrieved context matches the expected context.
        Returns a tuple of (f1_score, precision, recall) between 0 and 1.
        """
        # Simple overlap-based scoring for now
        # In a more sophisticated implementation, you might use semantic similarity
        
        expected_words = set(expected_context.lower().split())
        retrieved_words = set(retrieved_context.lower().split())
        
        if not expected_words:
            return 0.0, 0.0, 0.0
        overlap = len(expected_words & retrieved_words)
        precision = overlap / len(retrieved_words) if retrieved_words else 0.0
        recall = overlap / len(expected_words) if expected_words else 0.0

        logger.info(
            f"Retrieval Evaluation:\n"
            f"  Overlap: {overlap}\n"
            f"  Retrieved words ({len(retrieved_words)}): {sorted(retrieved_words)}\n"
            f"  Expected words ({len(expected_words)}): {sorted(expected_words)}\n"
            f"  Precision: {precision:.3f}, Recall: {recall:.3f}"
        )
        
        # F1 score
        if precision + recall == 0:
            return 0.0, precision, recall
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score, precision, recall
    
    async def evaluate_single_qa_pair(self, qa_pair: Dict[str, Any], top_k: Optional[int] = None) -> EvaluationResult:
        """Evaluate a single QA pair."""
        start_time = time.time()
        
        try:
            # Use provided top_k or default
            k = top_k if top_k is not None else self.default_top_k
            
            # Get response from RAG system with context
            response_data = self.agent.generate_response_with_context(qa_pair["question"], top_k=k)
            actual_answer = response_data["answer"]
            retrieved_chunks_text = response_data["context"]["text"]
            retrieved_chunks = response_data["context"]["chunks"]
            sources = response_data["context"]["sources"]
            mean_cosine_score = response_data["context"]["mean_cosine_score"]
            response_time = time.time() - start_time
            
            # Evaluate answer quality
            answer_quality_score = await self.evaluate_answer_quality(
                qa_pair["question"],
                qa_pair["answer"],
                actual_answer
            )
            
            # Evaluate retrieval accuracy (simplified)
            f1_score, precision_at_k, recall_at_k = await self.evaluate_retrieval_accuracy(
                qa_pair["context"],
                retrieved_chunks_text
            )
            
            # Ensure all scores are floats to prevent type errors
            f1_score = float(f1_score) if f1_score is not None else 0.0
            precision_at_k = float(precision_at_k) if precision_at_k is not None else 0.0
            recall_at_k = float(recall_at_k) if recall_at_k is not None else 0.0
            llm_as_judge_score = float(answer_quality_score) if answer_quality_score is not None else 0.0
            mean_cosine_score = float(mean_cosine_score) if mean_cosine_score is not None else 0.0
            
            # Extract source documents and chunk IDs from the retrieved chunks
            chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
            
            return EvaluationResult(
                question=qa_pair["question"],
                expected_answer=qa_pair["answer"],
                actual_answer=actual_answer,
                expected_correct_chunks=qa_pair["context"],
                sources=sources,
                chunk_ids=chunk_ids,
                f1_score=f1_score,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                mean_cosine_score=mean_cosine_score,
                llm_as_judge_score=llm_as_judge_score,
                latency=response_time,
                retrieved_chunks=retrieved_chunks_text,
                metadata={
                    **(qa_pair.get("metadata", {})),
                    "top_k": k,
                    "retrieved_chunks_array": retrieved_chunks
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating QA pair: {e}")
            return EvaluationResult(
                question=qa_pair["question"],
                expected_answer=qa_pair["answer"],
                actual_answer="Error occurred",
                expected_correct_chunks=qa_pair["context"],
                sources=[],
                chunk_ids=[],
                f1_score=0.0,
                precision_at_k=0.0,
                recall_at_k=0.0,
                mean_cosine_score=0.0,
                llm_as_judge_score=0.0,
                latency=time.time() - start_time,
                retrieved_chunks="",
                metadata=qa_pair.get("metadata", {})
            )
    
    async def evaluate_dataset(self, dataset_path: str) -> List[EvaluationResult]:
        """Evaluate the entire dataset."""
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = data["qa_pairs"]
        logger.info(f"Evaluating {len(qa_pairs)} QA pairs")
        
        results = []
        
        for qa_pair in tqdm(qa_pairs, desc="Evaluating QA pairs"):
            result = await self.evaluate_single_qa_pair(qa_pair)
            results.append(result)
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        return results
    
    def calculate_summary(self, results: List[EvaluationResult]) -> EvaluationSummary:
        """Calculate summary statistics from evaluation results, including precision and accuracy."""
        if not results:
            return EvaluationSummary(
                total_questions=0,
                average_f1_score=0.0,
                average_llm_as_judge_score=0.0,
                average_latency=0.0,
                average_mean_cosine_score=0.0,
                question_type_breakdown={},
                overall_score=0.0
            )

        total_questions = len(results)
        avg_f1_score = sum(float(r.f1_score) for r in results) / total_questions
        avg_llm_as_judge_score = sum(float(r.llm_as_judge_score) for r in results) / total_questions
        avg_latency = sum(float(r.latency) for r in results) / total_questions
        avg_precision_at_k = sum(float(r.precision_at_k) for r in results) / total_questions
        avg_recall_at_k = sum(float(r.recall_at_k) for r in results) / total_questions
        avg_mean_cosine = sum(float(r.mean_cosine_score) for r in results) / total_questions

        # Question type breakdown
        question_type_breakdown = {}
        for result in results:
            q_type = result.metadata.get("question_type", "unknown") if result.metadata else "unknown"

            if q_type not in question_type_breakdown:
                question_type_breakdown[q_type] = {
                    "count": 0,
                    "avg_f1_score": 0.0,
                    "avg_llm_as_judge_score": 0.0,
                    "avg_latency": 0.0,
                    "avg_precision_at_k": 0.0,
                    "avg_recall_at_k": 0.0,
                    "avg_mean_cosine": 0.0
                }

            breakdown = question_type_breakdown[q_type]
            breakdown["count"] += 1
            breakdown["avg_f1_score"] += float(result.f1_score)
            breakdown["avg_llm_as_judge_score"] += float(result.llm_as_judge_score)
            breakdown["avg_latency"] += float(result.latency)
            breakdown["avg_precision_at_k"] += float(result.precision_at_k)
            breakdown["avg_recall_at_k"] += float(result.recall_at_k)
            breakdown["avg_mean_cosine"] += float(result.mean_cosine_score)

        # Calculate averages for each question type
        for q_type, breakdown in question_type_breakdown.items():
            count = breakdown["count"]
            breakdown["avg_f1_score"] /= count
            breakdown["avg_llm_as_judge_score"] /= count
            breakdown["avg_latency"] /= count
            breakdown["avg_precision_at_k"] /= count
            breakdown["avg_recall_at_k"] /= count
            breakdown["avg_mean_cosine"] /= count

        # Overall score (weighted average)
        overall_score = (avg_f1_score * 0.25 + avg_llm_as_judge_score * 0.4 + avg_precision_at_k * 0.1 + avg_recall_at_k * 0.1 + avg_mean_cosine * 0.15)

        return EvaluationSummary(
            total_questions=total_questions,
            average_f1_score=avg_f1_score,
            average_llm_as_judge_score=avg_llm_as_judge_score,
            average_latency=avg_latency,
            average_mean_cosine_score=avg_mean_cosine,
            question_type_breakdown=question_type_breakdown,
            overall_score=overall_score
        )
    def print_tabular_results(self, results: List[EvaluationResult], save_to_file: bool = True, output_dir: str = "evals/output"):
        """Print evaluation results in tabular format using tabulate and optionally save to file."""
        if not results:
            print("No results to display.")
            return
        
        # Prepare table content
        table_content = []
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        table_content.append(f"RAG Evaluation Results - Generated on {timestamp}")
        table_content.append("=" * 100)
        table_content.append("")
        
        # Table 1: Search Evaluation Table
        search_table_header = "SEARCH EVALUATION RESULTS"
        print("\n" + "="*100)
        print(search_table_header)
        print("="*100)
        
        table_content.append(search_table_header)
        table_content.append("=" * 100)
        
        # Prepare data for search evaluation table
        search_data = []
        for result in results:
            # Get sources from the result
            sources_str = "N/A"
            if result.sources:
                sources_str = ", ".join(result.sources[:2])  # Limit to first 2 sources
                if len(result.sources) > 2:
                    sources_str += "..."
            
            # Truncate long text fields for better display (make them thinner)
            query = result.question[:10] + "..." if len(result.question) > 10 else result.question
            retrieved = result.retrieved_chunks[:10] + "..." if len(result.retrieved_chunks) > 10 else result.retrieved_chunks
            expected = result.expected_correct_chunks[:10] + "..." if len(result.expected_correct_chunks) > 10 else result.expected_correct_chunks
            
            search_data.append([
                query,
                retrieved,
                expected,
                sources_str,
                f"{result.precision_at_k:.3f}",
                f"{result.recall_at_k:.3f}",
                f"{result.f1_score:.3f}",
                f"{result.mean_cosine_score:.3f}",
                result.metadata.get("top_k", "N/A") if result.metadata else "N/A"
            ])
        
        # Define headers for search evaluation table
        search_headers = [
            "User Query", 
            "Retrieved Chunks", 
            "Expected Correct Chunks", 
            "Sources", 
            "Precision@K", 
            "Recall@K", 
            "F1 Score", 
            "Mean Cosine", 
            "Top-K"
        ]
        
        # Generate and print search evaluation table with thinner columns
        search_table = tabulate(search_data, headers=search_headers, tablefmt="grid", maxcolwidths=[25, 30, 30, 20, 12, 12, 10, 12, 8])
        print(search_table)
        table_content.append(search_table)
        table_content.append("")
        
        # Table 2: Answer System Table
        answer_table_header = "ANSWER SYSTEM EVALUATION RESULTS"
        print("\n" + "="*100)
        print(answer_table_header)
        print("="*100)
        
        table_content.append(answer_table_header)
        table_content.append("=" * 100)
        
        # Prepare data for answer system table
        answer_data = []
        for result in results:
            # Truncate long text fields for better display
            query = result.question[:10] + "..." if len(result.question) > 10 else result.question
            generated = result.actual_answer[:20] + "..." if len(result.actual_answer) > 20 else result.actual_answer
            expected = result.expected_answer[:20] + "..." if len(result.expected_answer) > 20 else result.expected_answer
            sources_str = ", ".join(result.sources[:2]) if result.sources else "N/A"
            if len(result.sources) > 2:
                sources_str += "..."
            
            answer_data.append([
                query,
                generated,
                expected,
                f"{result.llm_as_judge_score:.3f}",
                f"{result.latency:.3f}",
                sources_str
            ])
        
        # Define headers for answer system table
        answer_headers = [
            "User Query", 
            "Generated Answer", 
            "Expected Answer", 
            "LLM as Judge Score", 
            "Latency (s)", 
            "Sources"
        ]
        
        # Generate and print answer system table
        answer_table = tabulate(answer_data, headers=answer_headers, tablefmt="grid", maxcolwidths=[20, 20, 20, 15, 15, 25])
        print(answer_table)
        table_content.append(answer_table)
        table_content.append("")
        
        print("="*100)
        table_content.append("=" * 100)
        
        # Save to file if requested
        if save_to_file:
            try:
                # Create output directory if it doesn't exist
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evaluation_tables_{timestamp_str}.txt"
                filepath = Path(output_dir) / filename
                
                # Write table content to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(table_content))
                
                logger.info(f"Evaluation tables saved to: {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving tables to file: {e}")

    def save_results(
        self, 
        results: List[EvaluationResult], 
        summary: EvaluationSummary,
        output_path: str
    ):
        """Save evaluation results to JSON file."""
        output_data = {
            "summary": {
                "total_questions": summary.total_questions,
                "average_f1_score": summary.average_f1_score,
                "average_llm_as_judge_score": summary.average_llm_as_judge_score,
                "average_latency": summary.average_latency,
                "average_mean_cosine_score": summary.average_mean_cosine_score,
                "question_type_breakdown": summary.question_type_breakdown,
                "overall_score": summary.overall_score
            },
            "detailed_results": [
                {
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "actual_answer": r.actual_answer,
                    "expected_correct_chunks": r.expected_correct_chunks,
                    "sources": r.sources,
                    "chunk_ids": r.chunk_ids,
                    "f1_score": r.f1_score,
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "mean_cosine_score": r.mean_cosine_score,
                    "llm_as_judge_score": r.llm_as_judge_score,
                    "latency": r.latency,
                    "retrieved_chunks_text": r.retrieved_chunks,
                    "retrieved_chunks": r.metadata.get("retrieved_chunks_array", []),
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {output_path}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Questions: {summary.total_questions}")
        logger.info(f"Overall Score: {summary.overall_score:.3f}")
        logger.info(f"Average F1 Score: {summary.average_f1_score:.3f}")
        logger.info(f"Average LLM as Judge Score: {summary.average_llm_as_judge_score:.3f}")
        logger.info(f"Average Mean Cosine Score: {summary.average_mean_cosine_score:.3f}")
        logger.info(f"Average Latency: {summary.average_latency:.3f}s")
        
        logger.info("\nQuestion Type Breakdown:")
        for q_type, breakdown in summary.question_type_breakdown.items():
            logger.info(f"  {q_type}:")
            logger.info(f"    Count: {breakdown['count']}")
            logger.info(f"    Avg F1 Score: {breakdown['avg_f1_score']:.3f}")
            logger.info(f"    Avg LLM as Judge Score: {breakdown['avg_llm_as_judge_score']:.3f}")
            logger.info(f"    Avg Precision@K: {breakdown['avg_precision_at_k']:.3f}")
            logger.info(f"    Avg Recall@K: {breakdown['avg_recall_at_k']:.3f}")
            logger.info(f"    Avg Mean Cosine: {breakdown['avg_mean_cosine']:.3f}")
            logger.info(f"    Avg Latency: {breakdown['avg_latency']:.3f}s")


async def main():
    """Main function to run RAG evaluation with configurable top_k."""
    # This function is kept for backward compatibility but is not used
    # The main evaluation is run through evals/run_evaluation.py
    logger.info("This script is not meant to be run directly.")
    logger.info("Please use: python evals/run_evaluation.py --quick --top-k <value>")
    logger.info("or: python evals/run_evaluation.py --full --top-k <value>")


if __name__ == "__main__":
    asyncio.run(main()) 
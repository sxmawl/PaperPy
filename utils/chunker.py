"""
Text Chunking Utilities

This module provides three different text chunking strategies:
1. FixedCharacterChunker: Splits text into fixed-size chunks
2. RecursiveTextSplitter: Splits text recursively using multiple separators
3. SemanticChunker: Splits text based on semantic similarity using embeddings

All Chunkers implement a common interface for easy swapping and comparison.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import uuid
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
        
# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    start_char: int
    end_char: int
    metadata: Optional[Dict[str, Any]] = None


class BaseChunker(ABC):
    """Abstract base class for all chunkers."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[Chunk]:
        """Split text into chunks. Must be implemented by subclasses."""
        pass
    
    def _create_chunk_id(self, index: int) -> str:
        """Create a unique chunk ID."""
        return f"chunk_{index}_{uuid.uuid4().hex[:8]}"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


class FixedCharacterChunker(BaseChunker):
    """
    Fixed-size character-based chunker.
    
    Splits text into chunks of approximately the specified character size,
    trying to break at sentence boundaries when possible.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        break_at_sentences: bool = True
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.break_at_sentences = break_at_sentences
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """Split text into fixed-size chunks."""
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [Chunk(
                text=text,
                chunk_id=self._create_chunk_id(0),
                start_char=0,
                end_char=len(text)
            )]
        
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if self.break_at_sentences and end < len(text):
                # Try to break at sentence boundaries
                end = self._find_sentence_boundary(text, start, end)
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=self._create_chunk_id(chunk_id),
                    start_char=start,
                    end_char=end,
                    metadata={
                        "chunker_type": "fixed_character",
                        "chunk_size": len(chunk_text),
                        "break_at_sentences": self.break_at_sentences
                    }
                ))
                chunk_id += 1
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, target_end: int) -> int:
        """Find the best sentence boundary near the target end position."""
        # Look for sentence endings in the last 100 characters
        search_start = max(start, target_end - 100)
        search_end = min(len(text), target_end + 50)
        
        # Common sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        best_end = target_end
        for ending in sentence_endings:
            pos = text.rfind(ending, search_start, search_end)
            if pos != -1:
                # Found a sentence ending, use it
                best_end = pos + len(ending.rstrip())
                break
        
        return best_end


class RecursiveTextSplitter(BaseChunker):
    """
    Recursive text splitter using LangChain's RecursiveCharacterTextSplitter.
    
    This chunker uses LangChain's robust implementation that recursively splits text
    using a hierarchy of separators. It starts with the first separator and if the
    resulting chunks are still too large, it moves to the next separator in the list.
    
    The algorithm works as follows:
    1. Try to split by the first separator (e.g., "\n\n" for paragraphs)
    2. If any resulting chunk is still larger than chunk_size, recursively apply
       the next separator to that chunk
    3. Continue until all chunks are within the specified size limit
    4. If no separator works, split by characters as a last resort
    
    This approach ensures text is split at the most semantically meaningful boundaries
    first, preserving context and readability.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semi-colons
            ", ",    # Commas
            " ",     # Words
            ""       # Characters
        ]
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """Split text recursively using LangChain's implementation."""
        text = self._clean_text(text)
        
        # Use LangChain's splitter to get the text chunks
        # The recursive splitter handles chunk sizing automatically
        text_chunks = self.langchain_splitter.split_text(text)
        
        # Convert to our Chunk format
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(text_chunks):
            # Find the position of this chunk in the original text
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            current_pos = end_pos
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=self._create_chunk_id(i),
                start_char=start_pos,
                end_char=end_pos,
                metadata={
                    "chunker_type": "recursive_text_splitter",
                    "langchain_implementation": True,
                    "separators": self.separators,
                    "chunk_size": len(chunk_text)
                }
            ))
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that splits text based on semantic similarity.
    
    Uses embeddings to find natural break points in the text where
    semantic coherence is maintained within chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for SemanticChunker")
        
        self.client = OpenAI(api_key=api_key)
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """Split text based on semantic similarity."""
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [Chunk(
                text=text,
                chunk_id=self._create_chunk_id(0),
                start_char=0,
                end_char=len(text)
            )]
        
        # First, create potential chunks using a simpler method
        potential_chunks = self._create_potential_chunks(text)
        
        # Then merge chunks based on semantic similarity
        semantic_chunks = self._merge_semantic_chunks(potential_chunks)
        
        return semantic_chunks
    
    def _create_potential_chunks(self, text: str) -> List[Chunk]:
        """Create initial chunks using a simple character-based approach."""
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=self._create_chunk_id(chunk_id),
                    start_char=start,
                    end_char=end
                ))
                chunk_id += 1
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _merge_semantic_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks based on semantic similarity."""
        if len(chunks) <= 1:
            return chunks
        
        # Get embeddings for all chunks
        embeddings = self._get_embeddings([chunk.text for chunk in chunks])
        
        merged_chunks = []
        current_chunk = chunks[0]
        current_embedding = embeddings[0]
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            next_embedding = embeddings[i]
            
            # Calculate similarity
            similarity = self._cosine_similarity(current_embedding, next_embedding)
            
            # Check if chunks should be merged
            should_merge = (
                similarity >= self.similarity_threshold and
                len(current_chunk.text) + len(next_chunk.text) <= self.chunk_size * 1.5
            )
            
            if should_merge:
                # Merge chunks
                current_chunk.text += "\n\n" + next_chunk.text
                current_chunk.end_char = next_chunk.end_char
                # Update embedding (simple average for now)
                current_embedding = np.mean([current_embedding, next_embedding], axis=0)
            else:
                # Finalize current chunk and start new one
                current_chunk.metadata = {
                    "chunker_type": "semantic",
                    "similarity_threshold": self.similarity_threshold,
                    "embedding_model": self.embedding_model
                }
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
                current_embedding = next_embedding
        
        # Add the last chunk
        current_chunk.metadata = {
            "chunker_type": "semantic",
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embedding_model
        }
        merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Fall back to character-based chunking
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class ChunkerFactory:
    """Factory class for creating different types of chunkers."""
    
    @staticmethod
    def create_chunker(
        chunker_type: str = "fixed_character",
        **kwargs
    ) -> BaseChunker:
        """
        Create a chunker of the specified type.
        
        Args:
            chunker_type: Type of chunker ("fixed_character", "recursive", "semantic")
            **kwargs: Additional arguments for the chunker
        
        Returns:
            BaseChunker instance
        """
        chunker_type = chunker_type.lower()
        
        if chunker_type == "fixed_character":
            return FixedCharacterChunker(**kwargs)
        elif chunker_type == "recursive":
            return RecursiveTextSplitter(**kwargs)
        elif chunker_type == "semantic":
            return SemanticChunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
    
    @staticmethod
    def create_langchain_recursive_splitter(
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ) -> RecursiveTextSplitter:
        """
        Create a LangChain-based recursive text splitter with enhanced separators.
        
        This method provides a convenient way to create a recursive splitter
        with optimized separators for different languages and text types.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            separators: Custom separators (uses enhanced defaults if None)
            **kwargs: Additional arguments for RecursiveTextSplitter
        
        Returns:
            RecursiveTextSplitter instance
        """
        if separators is None:
            # Enhanced separators for better multilingual support
            separators = [
                "\n\n",    # Paragraphs
                "\n",      # Lines
                ". ",      # Sentences
                "! ",      # Exclamations
                "? ",      # Questions
                "; ",      # Semi-colons
                ", ",      # Commas
                " ",       # Words
                ".",       # Periods (for languages without spaces)
                "!",       # Exclamation marks
                "?",       # Question marks
                ";",       # Semi-colons
                ",",       # Commas
                "\u200b",  # Zero-width space (Thai, Myanmar, Khmer, Japanese)
                "\uff0c",  # Fullwidth comma (Chinese)
                "\u3001",  # Ideographic comma (Japanese, Chinese)
                "\uff0e",  # Fullwidth full stop (Chinese)
                "\u3002",  # Ideographic full stop (Japanese, Chinese)
                ""         # Characters (fallback)
            ]
        
        return RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )


# Convenience functions for easy usage
def chunk_text(
    text: str,
    chunker_type: str = "fixed_character",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> List[Chunk]:
    """
    Convenience function to chunk text using the specified chunker type.
    
    Args:
        text: Text to chunk
        chunker_type: Type of chunker to use
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        **kwargs: Additional arguments for the chunker
    
    Returns:
        List of Chunk objects
    """
    chunker = ChunkerFactory.create_chunker(
        chunker_type=chunker_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    return chunker.chunk_text(text)


def chunk_text_langchain_recursive(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    **kwargs
) -> List[Chunk]:
    """
    Convenience function to chunk text using LangChain's recursive text splitter.
    
    This is the recommended approach for most text chunking tasks as it uses
    LangChain's robust implementation with optimized separators.
    
    The recursive splitter automatically handles chunk sizing by trying separators
    in order of preference (paragraphs → lines → sentences → words → characters)
    until all chunks are within the specified size limit.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk (the splitter will ensure chunks don't exceed this)
        chunk_overlap: Overlap between chunks
        separators: Custom separators (uses enhanced defaults if None)
        **kwargs: Additional arguments for the chunker
    
    Returns:
        List of Chunk objects
    """
    chunker = ChunkerFactory.create_langchain_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        **kwargs
    )
    return chunker.chunk_text(text)


def get_chunk_texts(chunks: List[Chunk]) -> List[str]:
    """Extract just the text content from a list of chunks."""
    return [chunk.text for chunk in chunks]


def get_chunk_info(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    """Get detailed information about chunks."""
    return [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "length": len(chunk.text),
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ] 
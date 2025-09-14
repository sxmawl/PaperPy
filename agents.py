from openai import OpenAI
from typing import List, Dict, Any, Tuple
from utils.vector_store import VectorStore
import json
from dotenv import load_dotenv
import os
from utils.vector_store import ConnectionNotExistException

load_dotenv()

class Agent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.setup_vector_store()
        self.model = "gpt-4o"
        self.conversation_history = []
        
    def setup_vector_store(self):
        """Initialize or reinitialize the vector store."""
        self.uri = os.getenv("DATABASE_URL")
        self.vector_store = VectorStore(
            collection="knowledge",
            dimension=1536,
            uri=self.uri
        )

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def store_knowledge(self, text: str, source: str = "unknown") -> int:
        try:
            embedding = self.get_embedding(text)
            return self.vector_store.store_vectors([text], [embedding], [source])[0]
        except ConnectionNotExistException:
            # Reconnect and try again
            print("Reconnecting to PgVector...")
            self.setup_vector_store()
            embedding = self.get_embedding(text)
            return self.vector_store.store_vectors([text], [embedding], [source])[0]
        except Exception as e:
            print(f"Error storing knowledge: {e}")
            raise

    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.get_embedding(query)
            return self.vector_store.search_vectors(query_embedding, top_k)
        except ConnectionNotExistException:
            # Reconnect and try again
            print("Reconnecting to PgVector...")
            self.setup_vector_store()
            query_embedding = self.get_embedding(query)
            return self.vector_store.search_vectors(query_embedding, top_k)
        except Exception as e:
            print(f"Error searching the vector store: {e}")
            raise

    def calculate_mean_cosine_score(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate the mean cosine similarity score from search results."""
        if not search_results:
            return 0.0
        
        cosine_scores = [result.get("cosine_similarity", 0.0) for result in search_results]
        return sum(cosine_scores) / len(cosine_scores)

    def generate_response(self, query: str) -> str:
        """Generate response (backward compatibility method)."""
        result = self.generate_response_with_context(query)
        return result["answer"]

    def generate_response_with_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Generate response with both answer and retrieved context.
        Returns a dictionary with 'answer' and 'context' keys.
        
        Args:
            query: The user's question
            top_k: Number of top similar documents to retrieve (controls recall/precision)
        """
        try:
            # Search for relevant knowledge with configurable top_k
            relevant_knowledge = self.search_knowledge(query, top_k=top_k)

            # Concatenate all retrieved text
            retrieved_chunks_text = "\n".join([item["text"] for item in relevant_knowledge])
            
            # Create retrieved chunks array with proper structure
            retrieved_chunks = []
            sources = []
            for item in relevant_knowledge:
                chunk_obj = {
                    "chunk_id": item["id"],
                    "source": item["source"],
                    "chunk_text": item["text"]
                }
                retrieved_chunks.append(chunk_obj)
                sources.append(item["source"])
            
            # Calculate mean cosine score
            mean_cosine_score = self.calculate_mean_cosine_score(relevant_knowledge)
            
            # Store context details for evaluation
            context_details = {
                "text": retrieved_chunks_text,
                "sources": sources,
                "chunks": retrieved_chunks,
                "top_k": top_k,
                "mean_cosine_score": mean_cosine_score
            }

            # Prepare conversation history
            history = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in self.conversation_history[-5:]  # Keep last 5 exchanges
            ])

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer questions accurately."},
                    {"role": "system", "content": f"Previous conversation:\n{history}"},
                    {"role": "system", "content": f"Relevant context:\n{retrieved_chunks_text}"},
                    {"role": "user", "content": query}
                ],
                temperature=0.7
            )

            answer = response.choices[0].message.content

            # Store the exchange in history
            self.conversation_history.append({
                "user": query,
                "assistant": answer
            })

            return {
                "answer": answer,
                "context": context_details
            }
            
        except ConnectionNotExistException:
            # Reconnect and try again
            print("Reconnecting to PgVector...")
            self.setup_vector_store()
            return self.generate_response_with_context(query)
        except Exception as e:
            print(f"Error generating response: {e}")
            error_response = f"I'm sorry, I encountered an error: {str(e)}"
            return {
                "answer": error_response,
                "context": {
                    "text": "",
                    "sources": [],
                    "chunks": []
                }
            }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history 
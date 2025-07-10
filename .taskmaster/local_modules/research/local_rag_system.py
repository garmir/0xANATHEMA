#!/usr/bin/env python3
"""
Local RAG System for Task Master AI
Replaces Perplexity research with local vector search and knowledge synthesis
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import logging
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

# Vector search and embeddings
try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Vector search dependencies not available. Using fallback search.")

from ..core.api_abstraction import UnifiedModelAPI, TaskType

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document for RAG system"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

@dataclass
class SearchResult:
    """Search result with relevance scoring"""
    document: Document
    score: float
    relevance_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "relevance_reason": self.relevance_reason
        }

class EmbeddingEngine:
    """Local embedding engine using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self._lock = threading.Lock()
        
    def _ensure_model_loaded(self):
        """Ensure embedding model is loaded"""
        if self.model is None:
            with self._lock:
                if self.model is None:
                    if VECTOR_SEARCH_AVAILABLE:
                        try:
                            self.model = SentenceTransformer(self.model_name)
                            self.embedding_dim = self.model.get_sentence_embedding_dimension()
                            logger.info(f"Loaded embedding model: {self.model_name}")
                        except Exception as e:
                            logger.error(f"Failed to load embedding model: {e}")
                            self.model = None
                    else:
                        logger.warning("Sentence transformers not available, using fallback")
                        self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        self._ensure_model_loaded()
        
        if self.model is None:
            # Fallback: simple hash-based embedding
            return self._fallback_embedding(text)
        
        try:
            embedding = self.model.encode(text)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Fallback embedding using simple hash"""
        # Create a simple hash-based embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hex to numbers and normalize
        numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        # Pad or truncate to desired dimension
        while len(numbers) < self.embedding_dim:
            numbers.extend(numbers[:self.embedding_dim - len(numbers)])
        numbers = numbers[:self.embedding_dim]
        # Normalize
        embedding = np.array(numbers, dtype=np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        self._ensure_model_loaded()
        
        if self.model is None:
            return [self._fallback_embedding(text) for text in texts]
        
        try:
            embeddings = self.model.encode(texts)
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [self._fallback_embedding(text) for text in texts]

class VectorDatabase:
    """Local vector database for document storage and retrieval"""
    
    def __init__(self, db_path: str = ".taskmaster/local_modules/research/vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.use_qdrant = VECTOR_SEARCH_AVAILABLE
        self.embedding_engine = EmbeddingEngine()
        
        if self.use_qdrant:
            self._init_qdrant()
        else:
            self._init_sqlite()
    
    def _init_qdrant(self):
        """Initialize Qdrant vector database"""
        try:
            self.client = QdrantClient(path=str(self.db_path / "qdrant"))
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections()
            if "documents" not in [col.name for col in collections.collections]:
                self.client.create_collection(
                    collection_name="documents",
                    vectors_config=VectorParams(
                        size=self.embedding_engine.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            logger.info("Initialized Qdrant vector database")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self.use_qdrant = False
            self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite fallback database"""
        self.db_file = self.db_path / "documents.db"
        
        with sqlite3.connect(str(self.db_file)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp REAL,
                    embedding BLOB
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON documents(timestamp)
            """)
        
        logger.info("Initialized SQLite fallback database")
    
    def add_document(self, document: Document):
        """Add document to vector database"""
        # Generate embedding
        document.embedding = self.embedding_engine.embed_text(document.content)
        
        if self.use_qdrant:
            self._add_to_qdrant(document)
        else:
            self._add_to_sqlite(document)
    
    def _add_to_qdrant(self, document: Document):
        """Add document to Qdrant"""
        try:
            point = PointStruct(
                id=document.id,
                vector=document.embedding.tolist(),
                payload={
                    "title": document.title,
                    "content": document.content,
                    "metadata": document.metadata,
                    "timestamp": document.timestamp
                }
            )
            
            self.client.upsert(
                collection_name="documents",
                points=[point]
            )
        except Exception as e:
            logger.error(f"Failed to add document to Qdrant: {e}")
    
    def _add_to_sqlite(self, document: Document):
        """Add document to SQLite"""
        try:
            with sqlite3.connect(str(self.db_file)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, title, content, metadata, timestamp, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    document.id,
                    document.title,
                    document.content,
                    json.dumps(document.metadata),
                    document.timestamp,
                    document.embedding.tobytes()
                ))
        except Exception as e:
            logger.error(f"Failed to add document to SQLite: {e}")
    
    def search_similar(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for similar documents"""
        query_embedding = self.embedding_engine.embed_text(query)
        
        if self.use_qdrant:
            return self._search_qdrant(query_embedding, limit)
        else:
            return self._search_sqlite(query_embedding, limit)
    
    def _search_qdrant(self, query_embedding: np.ndarray, limit: int) -> List[SearchResult]:
        """Search using Qdrant"""
        try:
            results = self.client.search(
                collection_name="documents",
                query_vector=query_embedding.tolist(),
                limit=limit
            )
            
            search_results = []
            for result in results:
                document = Document(
                    id=result.id,
                    title=result.payload["title"],
                    content=result.payload["content"],
                    metadata=result.payload["metadata"],
                    timestamp=result.payload["timestamp"]
                )
                
                search_results.append(SearchResult(
                    document=document,
                    score=result.score
                ))
            
            return search_results
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def _search_sqlite(self, query_embedding: np.ndarray, limit: int) -> List[SearchResult]:
        """Search using SQLite with cosine similarity"""
        try:
            with sqlite3.connect(str(self.db_file)) as conn:
                cursor = conn.execute("""
                    SELECT id, title, content, metadata, timestamp, embedding
                    FROM documents
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """)
                
                results = []
                for row in cursor.fetchall():
                    doc_id, title, content, metadata_str, timestamp, embedding_bytes = row
                    
                    # Reconstruct embedding
                    doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    
                    document = Document(
                        id=doc_id,
                        title=title,
                        content=content,
                        metadata=json.loads(metadata_str),
                        timestamp=timestamp
                    )
                    
                    results.append(SearchResult(
                        document=document,
                        score=float(similarity)
                    ))
                
                # Sort by similarity and return top results
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:limit]
                
        except Exception as e:
            logger.error(f"SQLite search failed: {e}")
            return []

class KnowledgeBase:
    """Local knowledge base with domain-specific information"""
    
    def __init__(self, kb_path: str = ".taskmaster/local_modules/research/knowledge_base"):
        self.kb_path = Path(kb_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.vector_db = VectorDatabase()
        
        # Load initial knowledge
        self._load_initial_knowledge()
    
    def _load_initial_knowledge(self):
        """Load initial knowledge base"""
        initial_knowledge = {
            "autonomous_systems": {
                "title": "Autonomous Systems Design Principles",
                "content": """
                Autonomous systems require self-healing capabilities, adaptive workflows, and continuous monitoring.
                Key patterns include agentic design with clear separation of concerns, fault tolerance mechanisms,
                and evolutionary optimization approaches. Best practices include modularity, comprehensive logging,
                and performance monitoring with automatic recovery mechanisms.
                """,
                "metadata": {"domain": "autonomous_systems", "type": "principles"}
            },
            "memory_optimization": {
                "title": "Memory Optimization Techniques",
                "content": """
                Advanced memory optimization includes Williams 2025 sqrt-space algorithms, Cook-Mertz tree evaluation,
                and pebbling strategies. Complexity bounds achieve O(√n) and O(log n · log log n) improvements.
                Techniques include catalytic computing, memory reuse patterns, and adaptive allocation strategies.
                """,
                "metadata": {"domain": "algorithms", "type": "optimization"}
            },
            "task_management": {
                "title": "AI-Powered Task Management",
                "content": """
                Effective task management combines recursive decomposition with atomic task breakdown and hierarchical planning.
                AI approaches include ML-based prioritization, intelligent scheduling, and evolutionary optimization.
                Workflows should incorporate research-driven development, autonomous execution, and meta-improvement cycles.
                """,
                "metadata": {"domain": "task_management", "type": "methodologies"}
            },
            "local_llm_deployment": {
                "title": "Local LLM Deployment Best Practices",
                "content": """
                Local LLM deployment requires careful consideration of model selection, resource allocation, and performance optimization.
                Key providers include Ollama for simplicity, LocalAI for OpenAI compatibility, and LM Studio for GUI management.
                Performance optimization includes model quantization, GPU acceleration, and intelligent caching strategies.
                """,
                "metadata": {"domain": "ai_deployment", "type": "best_practices"}
            }
        }
        
        # Add initial documents to vector database
        for doc_id, doc_data in initial_knowledge.items():
            document = Document(
                id=doc_id,
                title=doc_data["title"],
                content=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            self.vector_db.add_document(document)
        
        logger.info("Loaded initial knowledge base")
    
    def add_knowledge(self, title: str, content: str, metadata: Dict[str, Any] = None):
        """Add new knowledge to the knowledge base"""
        if metadata is None:
            metadata = {}
        
        doc_id = hashlib.md5(f"{title}_{content}".encode()).hexdigest()
        document = Document(
            id=doc_id,
            title=title,
            content=content,
            metadata=metadata
        )
        
        self.vector_db.add_document(document)
        logger.info(f"Added knowledge: {title}")
    
    def search_knowledge(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search knowledge base"""
        return self.vector_db.search_similar(query, limit)

class LocalRAGSystem:
    """
    Local RAG (Retrieval-Augmented Generation) system
    Replaces Perplexity research with local vector search and knowledge synthesis
    """
    
    def __init__(self, 
                 api: UnifiedModelAPI,
                 data_dir: str = ".taskmaster/local_modules/research"):
        self.api = api
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.knowledge_base = KnowledgeBase()
        self.research_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Research session tracking
        self.research_sessions = {}
        
    async def research_query(self, 
                           query: str, 
                           context: str = "",
                           research_type: str = "comprehensive",
                           max_results: int = 10) -> Dict[str, Any]:
        """
        Perform research query using local RAG
        
        Args:
            query: Research question
            context: Additional context
            research_type: Type of research (comprehensive, quick, technical)
            max_results: Maximum number of results to consider
            
        Returns:
            Research results with synthesized information
        """
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.md5(f"{query}_{context}_{research_type}".encode()).hexdigest()
        if cache_key in self.research_cache:
            cached_result = self.research_cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.cache_ttl:
                cached_result["cached"] = True
                return cached_result
        
        # Retrieve relevant documents
        search_results = self.knowledge_base.search_knowledge(query, max_results)
        
        # Synthesize information using local LLM
        synthesis_result = await self._synthesize_research(query, context, search_results, research_type)
        
        # Structure final result
        research_result = {
            "query": query,
            "context": context,
            "research_type": research_type,
            "synthesis": synthesis_result,
            "sources": [result.to_dict() for result in search_results],
            "timestamp": time.time(),
            "execution_time": time.time() - start_time,
            "cached": False,
            "method": "local_rag"
        }
        
        # Cache result
        self.research_cache[cache_key] = research_result
        
        return research_result
    
    async def _synthesize_research(self, 
                                 query: str, 
                                 context: str, 
                                 search_results: List[SearchResult],
                                 research_type: str) -> Dict[str, Any]:
        """Synthesize research using local LLM"""
        
        # Prepare context from search results
        knowledge_context = ""
        for i, result in enumerate(search_results[:5]):  # Top 5 results
            knowledge_context += f"\n--- Source {i+1} (Score: {result.score:.3f}) ---\n"
            knowledge_context += f"Title: {result.document.title}\n"
            knowledge_context += f"Content: {result.document.content}\n"
        
        # Create synthesis prompt based on research type
        if research_type == "comprehensive":
            synthesis_prompt = f"""
            Conduct comprehensive research analysis based on the following query and available knowledge:
            
            QUERY: {query}
            CONTEXT: {context}
            
            AVAILABLE KNOWLEDGE:
            {knowledge_context}
            
            Please provide a comprehensive analysis including:
            1. Key findings and insights
            2. Best practices and methodologies
            3. Potential challenges and considerations
            4. Actionable recommendations
            5. Implementation guidance
            6. Future considerations
            
            Synthesize the information from all sources and provide original insights.
            Focus on practical, actionable information.
            """
        elif research_type == "quick":
            synthesis_prompt = f"""
            Provide a quick research summary based on available knowledge:
            
            QUERY: {query}
            CONTEXT: {context}
            
            AVAILABLE KNOWLEDGE:
            {knowledge_context}
            
            Provide a concise summary including:
            - Key points (3-5 bullet points)
            - Main recommendation
            - Next steps
            
            Keep response focused and actionable.
            """
        else:  # technical
            synthesis_prompt = f"""
            Provide technical analysis based on available knowledge:
            
            QUERY: {query}
            CONTEXT: {context}
            
            AVAILABLE KNOWLEDGE:
            {knowledge_context}
            
            Provide technical analysis including:
            1. Technical approach and methodology
            2. Implementation considerations
            3. Performance implications
            4. Best practices and patterns
            5. Potential challenges and solutions
            
            Focus on technical depth and implementation details.
            """
        
        try:
            response = await self.api.generate(
                synthesis_prompt,
                task_type=TaskType.RESEARCH,
                temperature=0.3
            )
            
            # Generate follow-up questions
            followup_questions = await self._generate_followup_questions(query, response.content)
            
            return {
                "synthesis": response.content,
                "model_used": response.model_used,
                "followup_questions": followup_questions,
                "confidence_score": self._calculate_confidence_score(search_results),
                "sources_used": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Research synthesis failed: {e}")
            return {
                "synthesis": "Research synthesis temporarily unavailable. Please refer to individual sources.",
                "model_used": "none",
                "followup_questions": [],
                "confidence_score": 0.0,
                "sources_used": len(search_results)
            }
    
    async def _generate_followup_questions(self, original_query: str, synthesis: str) -> List[str]:
        """Generate follow-up questions for deeper research"""
        followup_prompt = f"""
        Based on this research query and synthesis, generate 3-5 follow-up questions that would deepen understanding:
        
        ORIGINAL QUERY: {original_query}
        SYNTHESIS: {synthesis}
        
        Generate specific, actionable follow-up questions that:
        1. Address gaps in the current analysis
        2. Explore implementation details
        3. Consider edge cases or challenges
        4. Investigate related topics
        
        Return as a simple list, one question per line.
        """
        
        try:
            response = await self.api.generate(
                followup_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.4
            )
            
            # Parse questions
            questions = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Clean up question formatting
                    if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                        line = line[2:].strip()
                    if line:
                        questions.append(line)
            
            return questions[:5]  # Maximum 5 questions
            
        except Exception as e:
            logger.error(f"Follow-up question generation failed: {e}")
            return []
    
    def _calculate_confidence_score(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence score based on search results"""
        if not search_results:
            return 0.0
        
        # Average of top 3 scores
        top_scores = [result.score for result in search_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Adjust based on number of results
        result_factor = min(len(search_results) / 5.0, 1.0)  # Normalize to 5 results
        
        return avg_score * result_factor
    
    async def autonomous_research_loop(self, 
                                     problem: str, 
                                     context: str = "",
                                     max_iterations: int = 3) -> Dict[str, Any]:
        """
        Autonomous research loop that deepens understanding iteratively
        Replaces the original Perplexity-based research workflow
        """
        session_id = hashlib.md5(f"{problem}_{time.time()}".encode()).hexdigest()
        
        research_session = {
            "session_id": session_id,
            "problem": problem,
            "context": context,
            "iterations": [],
            "findings": [],
            "recommendations": []
        }
        
        current_query = problem
        
        for iteration in range(max_iterations):
            logger.info(f"Research iteration {iteration + 1}/{max_iterations}")
            
            # Perform research
            research_result = await self.research_query(
                current_query,
                context=context,
                research_type="comprehensive"
            )
            
            # Store iteration
            iteration_data = {
                "iteration": iteration + 1,
                "query": current_query,
                "result": research_result,
                "timestamp": time.time()
            }
            research_session["iterations"].append(iteration_data)
            
            # Extract key findings
            findings = self._extract_findings(research_result)
            research_session["findings"].extend(findings)
            
            # Generate next query based on follow-up questions
            if research_result["synthesis"]["followup_questions"]:
                current_query = research_result["synthesis"]["followup_questions"][0]
            else:
                break
        
        # Generate final recommendations
        recommendations = await self._generate_final_recommendations(research_session)
        research_session["recommendations"] = recommendations
        
        # Store session
        self.research_sessions[session_id] = research_session
        
        return research_session
    
    def _extract_findings(self, research_result: Dict[str, Any]) -> List[str]:
        """Extract key findings from research result"""
        synthesis = research_result["synthesis"]["synthesis"]
        findings = []
        
        # Simple extraction based on common patterns
        lines = synthesis.split('\n')
        for line in lines:
            line = line.strip()
            if any(line.lower().startswith(prefix) for prefix in [
                'key finding:', 'important:', 'note:', 'insight:', 'finding:'
            ]):
                findings.append(line)
            elif line.startswith(('•', '-', '*')) and len(line) > 20:
                findings.append(line)
        
        return findings[:10]  # Maximum 10 findings
    
    async def _generate_final_recommendations(self, research_session: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on research session"""
        # Compile all findings
        all_findings = "\n".join(research_session["findings"])
        
        recommendation_prompt = f"""
        Based on this research session, provide 5-8 actionable recommendations:
        
        PROBLEM: {research_session["problem"]}
        CONTEXT: {research_session["context"]}
        
        RESEARCH FINDINGS:
        {all_findings}
        
        Generate specific, actionable recommendations that:
        1. Address the core problem
        2. Are implementable with available resources
        3. Consider potential challenges
        4. Are prioritized by importance
        
        Format as numbered list with clear action items.
        """
        
        try:
            response = await self.api.generate(
                recommendation_prompt,
                task_type=TaskType.PLANNING,
                temperature=0.3
            )
            
            # Parse recommendations
            recommendations = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line and any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.']):
                    recommendations.append(line)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Final recommendations generation failed: {e}")
            return ["Continue research based on available findings"]
    
    def get_research_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get research session by ID"""
        return self.research_sessions.get(session_id)
    
    def add_external_knowledge(self, title: str, content: str, source: str = "external"):
        """Add external knowledge to the knowledge base"""
        self.knowledge_base.add_knowledge(
            title=title,
            content=content,
            metadata={"source": source, "added_at": time.time()}
        )

# Example usage
if __name__ == "__main__":
    async def test_local_rag():
        from ..core.api_abstraction import UnifiedModelAPI, ModelConfigFactory
        
        # Initialize API
        api = UnifiedModelAPI()
        api.add_model("ollama-llama2", ModelConfigFactory.create_ollama_config(
            "llama2", capabilities=[TaskType.RESEARCH, TaskType.ANALYSIS]
        ))
        
        # Initialize RAG system
        rag = LocalRAGSystem(api)
        
        # Test research query
        result = await rag.research_query(
            "How to implement recursive task decomposition in autonomous systems?",
            context="Working on Task Master AI local LLM migration"
        )
        
        print(f"Research result: {json.dumps(result, indent=2)}")
        
        # Test autonomous research loop
        research_session = await rag.autonomous_research_loop(
            "Optimize memory usage in local LLM deployment",
            context="Resource-constrained environment"
        )
        
        print(f"Research session: {json.dumps(research_session, indent=2)}")
    
    # Run test
    asyncio.run(test_local_rag())
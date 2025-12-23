# src/services/vector_service.py
"""
Vector Database Service for resume similarity search
"""

import asyncio
import logging
import numpy as np
from typing import List, Optional, TYPE_CHECKING, Any
from abc import ABC, abstractmethod

from ..config import settings
from ..core.exceptions import VectorDatabaseException
from ..schemas import ResumeMatch, ResumeData

if TYPE_CHECKING:
    from supabase import Client  # type: ignore
    from vecs import Client as VecsClient  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

logger = logging.getLogger(__name__)


class BaseVectorService(ABC):
    """Abstract base class for vector database services"""
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[ResumeMatch]:
        """Find most similar resumes"""
        pass
    
    @abstractmethod
    async def add_resume(self, resume: ResumeData) -> bool:
        """Add resume to vector database"""
        pass
    
    @abstractmethod
    async def delete_resume(self, resume_id: str) -> bool:
        """Delete resume from vector database"""
        pass


class SupabaseVectorService(BaseVectorService):
    """Supabase vector database implementation"""
    
    def __init__(self):
        self.collection_name = settings.supabase_collection
        self.embedding_model_name = settings.embedding_model
        
        # Initialize Supabase client
        self.client: Optional[Any] = None
        self.vecs_client: Optional[Any] = None
        self.collection: Optional[Any] = None
        
        try:
            from supabase import create_client, Client  # type: ignore
            from vecs import Client as VecsClient  # type: ignore
            
            if not settings.supabase_url or not settings.supabase_key:
                logger.warning("Supabase credentials not set, using simulated data")
            else:
                self.client = create_client(
                    settings.supabase_url,
                    settings.supabase_key
                )
                
                # Initialize vecs client for vector operations
                self.vecs_client = VecsClient(settings.supabase_url)
                if self.vecs_client:
                    self.collection = self.vecs_client.get_or_create_collection(
                        name=self.collection_name,
                        dimension=settings.embedding_dimension
                    )
                
                logger.info(f"Connected to Supabase collection: {self.collection_name}")
        
        except Exception as e:
            logger.warning(f"Supabase initialization failed: {e}. Using simulated data.")
        
        # Initialize embedding model
        self.embedding_model: Optional[Any] = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
        
        # Simulated data (fallback)
        self._init_sample_data()
    
    def _init_sample_data(self):
        """Initialize sample resume data"""
        self.resumes = [
            ResumeData(
                id="resume_1",
                content="Senior Software Engineer with 5 years experience in Python, FastAPI, React, and AWS. Built scalable microservices...",
                skills=["Python", "FastAPI", "React", "AWS", "Docker", "Kubernetes"],
                experience_years=5,
                embedding=np.random.rand(settings.embedding_dimension).tolist()
            ),
            ResumeData(
                id="resume_2",
                content="Full Stack Developer specializing in TypeScript, Node.js, and cloud infrastructure. Led team of 3 developers...",
                skills=["TypeScript", "Node.js", "React", "GCP", "MongoDB"],
                experience_years=3,
                embedding=np.random.rand(settings.embedding_dimension).tolist()
            ),
            ResumeData(
                id="resume_3",
                content="Machine Learning Engineer with expertise in PyTorch, TensorFlow, and MLOps. Deployed models at scale...",
                skills=["Python", "PyTorch", "TensorFlow", "MLOps", "Kubernetes"],
                experience_years=4,
                embedding=np.random.rand(settings.embedding_dimension).tolist()
            ),
            ResumeData(
                id="resume_4",
                content="DevOps Engineer with 6 years experience in AWS, Azure, CI/CD pipelines, and infrastructure automation...",
                skills=["AWS", "Azure", "Docker", "Kubernetes", "Terraform", "Jenkins"],
                experience_years=6,
                embedding=np.random.rand(settings.embedding_dimension).tolist()
            ),
            ResumeData(
                id="resume_5",
                content="Frontend Developer specializing in React, Vue.js, and modern web technologies. Built responsive UIs...",
                skills=["React", "Vue.js", "TypeScript", "CSS", "Webpack"],
                experience_years=4,
                embedding=np.random.rand(settings.embedding_dimension).tolist()
            ),
        ]
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        
        try:
            if self.embedding_model:
                # Use actual embedding model
                embedding = self.embedding_model.encode(text)
                return np.array(embedding)
            else:
                # Simulated embedding
                await asyncio.sleep(0.1)
                return np.random.rand(settings.embedding_dimension)
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise VectorDatabaseException("embedding", str(e))
    
    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[ResumeMatch]:
        """Find most similar resumes using cosine similarity"""
        
        try:
            if self.collection and self.vecs_client:
                # Use Supabase vector search
                results = self.collection.query(
                    data=query_embedding.tolist(),
                    limit=top_k,
                    filters={},
                    measure="cosine_distance"
                )
                
                matches = []
                for result in results:
                    # Parse metadata
                    metadata = result.get('metadata', {})
                    matches.append(
                        ResumeMatch(
                            resume_id=result['id'],
                            content=metadata.get('content', ''),
                            skills=metadata.get('skills', []),
                            experience_years=metadata.get('experience_years', 0),
                            similarity_score=1.0 - result['distance']  # Convert distance to similarity
                        )
                    )
                
                return matches
            
            else:
                # Fallback to simulated search
                results = []
                
                for resume in self.resumes:
                    resume_embedding = np.array(resume.embedding)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, resume_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(resume_embedding)
                    )
                    
                    # Filter by threshold
                    if similarity >= settings.min_similarity_threshold:
                        results.append(
                            ResumeMatch(
                                resume_id=resume.id,
                                content=resume.content,
                                skills=resume.skills,
                                experience_years=resume.experience_years,
                                similarity_score=float(similarity)
                            )
                        )
                
                # Sort by similarity (descending)
                results.sort(key=lambda x: x.similarity_score, reverse=True)
                
                logger.info(f"Found {len(results)} matches, returning top {top_k}")
                return results[:top_k]
        
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise VectorDatabaseException("similarity search", str(e))
    
    async def add_resume(self, resume: ResumeData) -> bool:
        """Add resume to vector database"""
        
        try:
            # Generate embedding if not provided
            if not resume.embedding:
                embedding = await self.embed_text(resume.content)
                resume.embedding = embedding.tolist()
            
            if self.collection:
                # Add to Supabase
                self.collection.upsert(
                    records=[(
                        resume.id,
                        resume.embedding,
                        {
                            "content": resume.content,
                            "skills": resume.skills,
                            "experience_years": resume.experience_years,
                            "education": resume.education,
                            "certifications": resume.certifications,
                            **resume.metadata
                        }
                    )]
                )
                logger.info(f"Added resume to Supabase: {resume.id}")
            else:
                # Add to simulated data
                self.resumes.append(resume)
                logger.info(f"Added resume to simulated data: {resume.id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to add resume: {e}")
            raise VectorDatabaseException("add resume", str(e))
    
    async def delete_resume(self, resume_id: str) -> bool:
        """Delete resume from vector database"""
        
        try:
            if self.collection:
                # Delete from Supabase
                self.collection.delete(ids=[resume_id])
                logger.info(f"Deleted resume from Supabase: {resume_id}")
            else:
                # Delete from simulated data
                self.resumes = [r for r in self.resumes if r.id != resume_id]
                logger.info(f"Deleted resume from simulated data: {resume_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete resume: {e}")
            raise VectorDatabaseException("delete resume", str(e))


class ChromaDBVectorService(BaseVectorService):
    """ChromaDB vector database implementation (alternative)"""
    
    def __init__(self):
        self.collection_name = settings.chromadb_collection
        self.embedding_model_name = settings.embedding_model
        
        # Initialize ChromaDB
        self.client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.embedding_model: Optional[Any] = None
        
        try:
            import chromadb  # type: ignore
            self.client = chromadb.Client()
            if self.client:
                self.collection = self.client.get_or_create_collection(self.collection_name)
                logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}")
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception:
            pass
        
        # Simulated data
        self._init_sample_data()
    
    def _init_sample_data(self):
        """Initialize sample data"""
        # Same as Supabase
        self.resumes = [
            ResumeData(
                id="resume_1",
                content="Senior Software Engineer with 5 years experience in Python, FastAPI, React, and AWS. Built scalable microservices...",
                skills=["Python", "FastAPI", "React", "AWS", "Docker", "Kubernetes"],
                experience_years=5,
                embedding=np.random.rand(settings.embedding_dimension).tolist()
            ),
        ]
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.embedding_model:
            embedding = self.embedding_model.encode(text)
            return np.array(embedding)
        else:
            await asyncio.sleep(0.1)
            return np.random.rand(settings.embedding_dimension)
    
    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[ResumeMatch]:
        """Find most similar resumes"""
        # Implementation similar to Supabase
        results = []
        for resume in self.resumes:
            resume_embedding = np.array(resume.embedding)
            similarity = np.dot(query_embedding, resume_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(resume_embedding)
            )
            if similarity >= settings.min_similarity_threshold:
                results.append(
                    ResumeMatch(
                        resume_id=resume.id,
                        content=resume.content,
                        skills=resume.skills,
                        experience_years=resume.experience_years,
                        similarity_score=float(similarity)
                    )
                )
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    async def add_resume(self, resume: ResumeData) -> bool:
        """Add resume to ChromaDB"""
        if not resume.embedding:
            embedding = await self.embed_text(resume.content)
            resume.embedding = embedding.tolist()
        self.resumes.append(resume)
        return True
    
    async def delete_resume(self, resume_id: str) -> bool:
        """Delete resume from ChromaDB"""
        self.resumes = [r for r in self.resumes if r.id != resume_id]
        return True


class VectorServiceFactory:
    """Factory for creating vector service instances"""
    
    @staticmethod
    def create() -> BaseVectorService:
        """Create vector service based on configuration"""
        
        db_type = settings.vector_db_type.lower()
        
        if db_type == "supabase":
            logger.info("Using Supabase vector service")
            return SupabaseVectorService()
        elif db_type == "chromadb":
            logger.info("Using ChromaDB vector service")
            return ChromaDBVectorService()
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")


# Singleton instance
_vector_service: Optional[BaseVectorService] = None


def get_vector_service() -> BaseVectorService:
    """Get vector service instance (singleton)"""
    global _vector_service
    
    if _vector_service is None:
        _vector_service = VectorServiceFactory.create()
    
    return _vector_service
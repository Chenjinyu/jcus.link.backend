from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from supabase import create_client, Client
import openai
from datetime import datetime
import httpx

@dataclass
class EmbeddingModel:
    name: str
    provider: str
    model_identifier: str
    dimensions: int
    is_local: bool

class EmbeddingFactory:
    """Factory for creating embeddings with different models"""
    
    def __init__(self, supabase: Client, openai_key: Optional[str] = None):
        self.supabase = supabase
        self.openai_client = openai.OpenAI(api_key=openai_key) if openai_key else None
        self._load_models()
    
    def _load_models(self):
        """Load available models from database"""
        result = self.supabase.table('embedding_models').select('*').eq('is_active', True).execute()
        self.models = {m['name']: EmbeddingModel(**m) for m in result.data}
    
    async def create_embedding(
        self, 
        text: str, 
        model_name: str = 'openai-small'
    ) -> List[float]:
        """Create embedding using specified model"""
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        if model.provider == 'openai':
            return await self._create_openai_embedding(text, model)
        elif model.provider == 'ollama':
            return await self._create_ollama_embedding(text, model)
        else:
            raise ValueError(f"Provider {model.provider} not supported")
    
    async def _create_openai_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Create OpenAI embedding"""
        response = self.openai_client.embeddings.create(
            model=model.model_identifier,
            input=text
        )
        return response.data[0].embedding
    
    async def _create_ollama_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Create Ollama embedding (local)"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': model.model_identifier,
                    'prompt': text
                }
            )
            return response.json()['embedding']

class VectorDatabase:
    """Main interface for vector database operations"""
    
    def __init__(self, supabase_url: str, supabase_key: str, openai_key: Optional[str] = None):
        self.supabase = create_client(supabase_url, supabase_key)
        self.embedding_factory = EmbeddingFactory(self.supabase, openai_key)
    
    async def add_document(
        self,
        user_id: str,
        content_type: str,
        title: str,
        content: str,
        metadata: Dict = None,
        tags: List[str] = None,
        model_names: List[str] = None  # Multi-model support!
    ) -> str:
        """Add document with embeddings from multiple models"""
        
        # Default to all active models if not specified
        if model_names is None:
            model_names = list(self.embedding_factory.models.keys())
        
        # Get content type ID
        content_type_result = self.supabase.table('content_types').select('id').eq('name', content_type).single().execute()
        content_type_id = content_type_result.data['id']
        
        # Insert document
        doc_result = self.supabase.table('documents').insert({
            'user_id': user_id,
            'content_type_id': content_type_id,
            'title': title,
            'content': content,
            'metadata': metadata or {},
            'tags': tags or []
        }).execute()
        
        document_id = doc_result.data[0]['id']
        
        # Create embeddings with multiple models
        for model_name in model_names:
            embedding = await self.embedding_factory.create_embedding(content, model_name)
            model_id = self.supabase.table('embedding_models').select('id').eq('name', model_name).single().execute().data['id']
            
            self.supabase.table('embeddings').insert({
                'document_id': document_id,
                'embedding_model_id': model_id,
                'embedding': embedding,
                'chunk_text': content
            }).execute()
        
        return document_id
    
    async def update_document(
        self,
        document_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        recreate_embeddings: bool = True
    ):
        """Update document and optionally recreate embeddings"""
        
        update_data = {}
        if title: update_data['title'] = title
        if content: update_data['content'] = content
        if metadata: update_data['metadata'] = metadata
        if tags: update_data['tags'] = tags
        
        # Update document
        self.supabase.table('documents').update(update_data).eq('id', document_id).execute()
        
        # Recreate embeddings if content changed
        if recreate_embeddings and content:
            # Get existing embedding models for this document
            existing = self.supabase.table('embeddings').select('embedding_model_id').eq('document_id', document_id).execute()
            model_ids = [e['embedding_model_id'] for e in existing.data]
            
            # Delete old embeddings
            self.supabase.table('embeddings').delete().eq('document_id', document_id).execute()
            
            # Create new embeddings
            for model_id in model_ids:
                model = self.supabase.table('embedding_models').select('name').eq('id', model_id).single().execute().data
                embedding = await self.embedding_factory.create_embedding(content, model['name'])
                
                self.supabase.table('embeddings').insert({
                    'document_id': document_id,
                    'embedding_model_id': model_id,
                    'embedding': embedding,
                    'chunk_text': content
                }).execute()
    
    async def search(
        self,
        query: str,
        user_id: str,
        model_name: str = 'openai-small',
        content_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Search documents using vector similarity"""
        
        # Create query embedding
        query_embedding = await self.embedding_factory.create_embedding(query, model_name)
        
        # Get model ID
        model_id = self.supabase.table('embedding_models').select('id').eq('name', model_name).single().execute().data['id']
        
        # Search using database function
        results = self.supabase.rpc('search_documents', {
            'query_embedding': query_embedding,
            'model_id': model_id,
            'match_threshold': threshold,
            'match_count': limit,
            'filter_user_id': user_id,
            'filter_content_types': content_types,
            'filter_tags': tags
        }).execute()
        
        return results.data
    
    async def add_profile_data(
        self,
        user_id: str,
        category: Literal['work_experience', 'education', 'certification', 'skill', 'value', 'goal'],
        data: Dict,
        create_embedding: bool = True,
        model_names: List[str] = None
    ) -> str:
        """Add structured profile data"""
        
        # Create searchable text from data
        searchable_text = self._flatten_dict_to_text(data)
        
        # Insert profile data
        result = self.supabase.table('profile_data').insert({
            'user_id': user_id,
            'category': category,
            'data': data,
            'searchable_text': searchable_text
        }).execute()
        
        profile_id = result.data[0]['id']
        
        # Create corresponding document with embeddings
        if create_embedding:
            await self.add_document(
                user_id=user_id,
                content_type=category,
                title=data.get('title', category),
                content=searchable_text,
                metadata={'profile_data_id': profile_id},
                model_names=model_names
            )
        
        return profile_id
    
    def _flatten_dict_to_text(self, data: Dict) -> str:
        """Convert dictionary to searchable text"""
        parts = []
        for key, value in data.items():
            if isinstance(value, (str, int, float)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                parts.append(f"{key}: {', '.join(map(str, value))}")
        return '. '.join(parts)
    
    async def add_article(
        self,
        user_id: str,
        title: str,
        content: str,
        tags: List[str] = None,
        category: str = None,
        status: str = 'draft',
        model_names: List[str] = None
    ) -> str:
        """Add article with embeddings"""
        
        # Create document with embeddings
        document_id = await self.add_document(
            user_id=user_id,
            content_type='article',
            title=title,
            content=content,
            tags=tags,
            model_names=model_names
        )
        
        # Create article entry
        article_result = self.supabase.table('articles').insert({
            'user_id': user_id,
            'document_id': document_id,
            'title': title,
            'content': content,
            'tags': tags or [],
            'category': category,
            'status': status,
            'slug': self._create_slug(title)
        }).execute()
        
        return article_result.data[0]['id']
    
    def _create_slug(self, title: str) -> str:
        """Create URL-friendly slug"""
        return title.lower().replace(' ', '-').replace('_', '-')
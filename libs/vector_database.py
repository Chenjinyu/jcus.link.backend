# vector_database.py
from typing import List, Dict, Optional, Union, Literal
from dataclasses import dataclass
from supabase import create_client, Client
import openai
from datetime import datetime
import httpx
import json
try:
    import google as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None

@dataclass
class EmbeddingModel:
    id: str
    name: str
    provider: str
    model_identifier: str
    dimensions: int
    is_local: bool
    cost_per_token: Optional[float] = None

class VectorDatabase:
    """
    Comprehensive vector database interface for managing documents,
    embeddings, articles, profile data, and personal attributes.
    
    Supports multiple AI providers: OpenAI, Ollama, Google Gemini
    """
    
    def __init__(
        self, 
        supabase_url: str, 
        supabase_key: str, 
        openai_key: Optional[str] = None,
        google_key: Optional[str] = None,
        ollama_url: str = 'http://localhost:11434'
    ):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Initialize AI providers
        self.openai_client = openai.OpenAI(api_key=openai_key) if openai_key else None
        self.ollama_url = ollama_url
        
        if google_key and GOOGLE_AVAILABLE:
            genai.configure(api_key=google_key)
            self.google_client = genai
        else:
            self.google_client = None
        
        self._models_cache: Dict[str, EmbeddingModel] = {}
        self._load_models()
    
    def _load_models(self):
        """Load embedding models from database"""
        result = self.supabase.table('embedding_models').select('*').eq('is_active', True).execute()
        for model_data in result.data:
            self._models_cache[model_data['name']] = EmbeddingModel(
                id=model_data['id'],
                name=model_data['name'],
                provider=model_data['provider'],
                model_identifier=model_data['model_identifier'],
                dimensions=model_data['dimensions'],
                is_local=model_data['is_local'],
                cost_per_token=model_data.get('cost_per_token')
            )
    
    # ========================================================================
    # EMBEDDING CREATION - MULTI-PROVIDER
    # ========================================================================
    
    async def create_embedding(
        self, 
        text: str, 
        model_name: str = 'nomic-embed-text-768'
    ) -> List[float]:
        """
        Create embedding using specified model.
        Supports: OpenAI, Ollama, Google Gemini
        """
        model = self._models_cache.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found or not active")
        
        if model.provider == 'openai':
            return await self._create_openai_embedding(text, model)
        elif model.provider == 'ollama':
            return await self._create_ollama_embedding(text, model)
        elif model.provider == 'google':
            return await self._create_google_embedding(text, model)
        else:
            raise ValueError(f"Provider {model.provider} not supported")
    
    async def _create_openai_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Create OpenAI embedding with dimension reduction if needed"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Provide openai_key in constructor.")
        
        response = self.openai_client.embeddings.create(
            model=model.model_identifier,
            input=text,
            dimensions=model.dimensions if model.dimensions <= 2000 else None
        )
        return response.data[0].embedding
    
    async def _create_ollama_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Create Ollama embedding (local)"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.ollama_url}/api/embeddings',
                json={
                    'model': model.model_identifier,
                    'prompt': text
                },
                timeout=30.0
            )
            return response.json()['embedding']
    
    async def _create_google_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Create Google Gemini embedding"""
        if not self.google_client:
            raise ValueError("Google client not initialized. Provide google_key in constructor.")
        
        result = self.google_client.embed_content(
            model=model.model_identifier,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    async def add_document(
        self,
        user_id: str,
        title: str,
        content: str,
        metadata: Dict = None,
        tags: List[str] = None,
        model_names: List[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> str:
        """
        Add document with embeddings from multiple models.
        Automatically chunks long content.
        """
        if model_names is None:
            model_names = list(self._models_cache.keys())
        
        doc_result = self.supabase.table('documents').insert({
            'user_id': user_id,
            'title': title,
            'content': content,
            'metadata': metadata or {},
            'tags': tags or []
        }).execute()
        
        document_id = doc_result.data[0]['id']
        
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        total_chunks = len(chunks)
        
        for model_name in model_names:
            model = self._models_cache[model_name]
            
            for chunk_index, chunk_text in enumerate(chunks):
                embedding = await self.create_embedding(chunk_text, model_name)
                
                self.supabase.table('embeddings').insert({
                    'document_id': document_id,
                    'embedding_model_id': model.id,
                    'embedding': embedding,
                    'chunk_text': chunk_text,
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks
                }).execute()
        
        return document_id
    
    async def update_document(
        self,
        document_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        recreate_embeddings: bool = True,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> bool:
        """Update document and optionally recreate embeddings"""
        update_data = {}
        if title is not None:
            update_data['title'] = title
        if content is not None:
            update_data['content'] = content
        if metadata is not None:
            update_data['metadata'] = metadata
        if tags is not None:
            update_data['tags'] = tags
        
        self.supabase.table('documents').update(update_data).eq('id', document_id).execute()
        
        if recreate_embeddings and content is not None:
            existing = self.supabase.table('embeddings').select(
                'embedding_model_id'
            ).eq('document_id', document_id).execute()
            
            model_ids = list(set([e['embedding_model_id'] for e in existing.data]))
            
            self.supabase.table('embeddings').delete().eq('document_id', document_id).execute()
            
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            
            for model_id in model_ids:
                model_name = next(
                    (name for name, m in self._models_cache.items() if m.id == model_id),
                    None
                )
                
                if model_name:
                    for chunk_index, chunk_text in enumerate(chunks):
                        embedding = await self.create_embedding(chunk_text, model_name)
                        
                        self.supabase.table('embeddings').insert({
                            'document_id': document_id,
                            'embedding_model_id': model_id,
                            'embedding': embedding,
                            'chunk_text': chunk_text,
                            'chunk_index': chunk_index,
                            'total_chunks': total_chunks
                        }).execute()
        
        return True
    
    def delete_document(self, document_id: str, soft_delete: bool = True) -> bool:
        """Delete document (soft or hard delete)"""
        if soft_delete:
            self.supabase.table('documents').update({
                'deleted_at': datetime.now().isoformat(),
                'is_current': False
            }).eq('id', document_id).execute()
        else:
            self.supabase.table('documents').delete().eq('id', document_id).execute()
        
        return True
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Get document by ID"""
        result = self.supabase.table('documents').select('*').eq('id', document_id).execute()
        return result.data[0] if result.data else None
    
    def get_documents(
        self,
        user_id: str,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get documents with optional filtering"""
        query = self.supabase.table('documents').select(
            '*'
        ).eq('user_id', user_id).eq('is_current', True).is_('deleted_at', 'null')
        
        if tags:
            query = query.overlaps('tags', tags)
        
        result = query.order('created_at', desc=True).limit(limit).range(offset, offset + limit - 1).execute()
        return result.data
    
    # ========================================================================
    # SQL FUNCTION: upsert_document_with_embedding
    # ========================================================================
    
    async def upsert_document_with_embedding_rpc_function(
        self,
        user_id: str,
        title: str,
        content: str,
        metadata: Dict = None,
        tags: List[str] = None,
        model_name: str = 'nomic-embed-text-768',
        chunks: List[Dict] = None
    ) -> str:
        """
        Call SQL function to upsert document with embeddings.
        
        Args:
            chunks: List of dicts with 'text', 'embedding', 'chunk_index'
        """
        chunks_json = chunks or []
        
        result = self.supabase.rpc('upsert_document_with_embedding', {
            'p_user_id': user_id,
            'p_title': title,
            'p_content': content,
            'p_metadata': metadata or {},
            'p_tags': tags or [],
            'p_embedding_model_name': model_name,
            'p_chunks': json.dumps(chunks_json)
        }).execute()
        
        return result.data
    
    # ========================================================================
    # ARTICLE OPERATIONS
    # ========================================================================
    
    async def add_article(
        self,
        user_id: str,
        title: str,
        content: str,
        subtitle: Optional[str] = None,
        excerpt: Optional[str] = None,
        tags: List[str] = None,
        category: Optional[str] = None,
        status: str = 'draft',
        seo_title: Optional[str] = None,
        seo_description: Optional[str] = None,
        og_image: Optional[str] = None,
        model_names: List[str] = None,
        chunk_size: int = 500
    ) -> Dict[str, str]:
        """Add article with document and embeddings"""
        document_id = await self.add_document(
            user_id=user_id,
            title=title,
            content=content,
            tags=tags,
            metadata={'source': 'article', 'category': category},
            model_names=model_names,
            chunk_size=chunk_size
        )
        
        slug = self._create_slug(title)
        
        article_result = self.supabase.table('articles').insert({
            'user_id': user_id,
            'document_id': document_id,
            'title': title,
            'subtitle': subtitle,
            'content': content,
            'excerpt': excerpt or content[:200],
            'tags': tags or [],
            'category': category,
            'status': status,
            'slug': slug,
            'seo_title': seo_title,
            'seo_description': seo_description,
            'og_image': og_image,
            'published_at': datetime.now().isoformat() if status == 'published' else None
        }).execute()
        
        return {
            'article_id': article_result.data[0]['id'],
            'document_id': document_id
        }
    
    async def update_article(
        self,
        article_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        subtitle: Optional[str] = None,
        excerpt: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        seo_title: Optional[str] = None,
        seo_description: Optional[str] = None,
        og_image: Optional[str] = None,
        recreate_embeddings: bool = True
    ) -> bool:
        """Update article and optionally recreate embeddings"""
        article = self.supabase.table('articles').select('document_id').eq('id', article_id).single().execute()
        
        if not article.data:
            raise ValueError(f"Article {article_id} not found")
        
        document_id = article.data['document_id']
        
        article_update = {}
        if title is not None:
            article_update['title'] = title
        if content is not None:
            article_update['content'] = content
        if subtitle is not None:
            article_update['subtitle'] = subtitle
        if excerpt is not None:
            article_update['excerpt'] = excerpt
        if tags is not None:
            article_update['tags'] = tags
        if category is not None:
            article_update['category'] = category
        if status is not None:
            article_update['status'] = status
            if status == 'published':
                article_update['published_at'] = datetime.now().isoformat()
        if seo_title is not None:
            article_update['seo_title'] = seo_title
        if seo_description is not None:
            article_update['seo_description'] = seo_description
        if og_image is not None:
            article_update['og_image'] = og_image
        
        self.supabase.table('articles').update(article_update).eq('id', article_id).execute()
        
        await self.update_document(
            document_id=document_id,
            title=title,
            content=content,
            tags=tags,
            recreate_embeddings=recreate_embeddings
        )
        
        return True
    
    def delete_article(self, article_id: str, soft_delete: bool = True) -> bool:
        """Delete article and associated document"""
        article = self.supabase.table('articles').select('document_id').eq('id', article_id).single().execute()
        
        if article.data:
            self.delete_document(article.data['document_id'], soft_delete=soft_delete)
        
        return True
    
    def get_article(self, article_id: str) -> Optional[Dict]:
        """Get article by ID"""
        result = self.supabase.table('articles').select('*').eq('id', article_id).execute()
        return result.data[0] if result.data else None
    
    def get_article_by_slug(self, slug: str) -> Optional[Dict]:
        """Get article by slug"""
        result = self.supabase.table('articles').select('*').eq('slug', slug).execute()
        return result.data[0] if result.data else None
    
    def get_articles(
        self,
        user_id: str,
        status: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get articles with optional filtering"""
        query = self.supabase.table('articles').select('*').eq('user_id', user_id)
        
        if status:
            query = query.eq('status', status)
        if category:
            query = query.eq('category', category)
        if tags:
            query = query.overlaps('tags', tags)
        
        result = query.order('created_at', desc=True).limit(limit).range(offset, offset + limit - 1).execute()
        return result.data
    
    def increment_article_views(self, article_id: str) -> bool:
        """Increment article view count"""
        article = self.get_article(article_id)
        if article:
            self.supabase.table('articles').update({
                'view_count': article.get('view_count', 0) + 1
            }).eq('id', article_id).execute()
            return True
        return False
    
    def increment_article_likes(self, article_id: str) -> bool:
        """Increment article like count"""
        article = self.get_article(article_id)
        if article:
            self.supabase.table('articles').update({
                'like_count': article.get('like_count', 0) + 1
            }).eq('id', article_id).execute()
            return True
        return False
    
    # ========================================================================
    # PROFILE DATA OPERATIONS
    # ========================================================================
    
    async def add_profile_data(
        self,
        user_id: str,
        category: Literal['work_experience', 'education', 'certification', 'skill', 'project', 'volunteer'],
        data: Dict,
        searchable: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_current: bool = False,
        is_featured: bool = False,
        display_order: Optional[int] = None,
        model_names: List[str] = None
    ) -> Dict[str, str]:
        """Add profile data with optional document/embedding"""
        result = {'profile_id': None, 'document_id': None}
        
        profile_result = self.supabase.table('profile_data').insert({
            'user_id': user_id,
            'category': category,
            'data': data,
            'start_date': start_date or data.get('start_date'),
            'end_date': end_date or data.get('end_date'),
            'is_current': is_current or data.get('current', False),
            'is_featured': is_featured,
            'display_order': display_order
        }).execute()
        
        result['profile_id'] = profile_result.data[0]['id']
        
        if searchable:
            profile = self.supabase.table('profile_data').select('searchable_text').eq(
                'id', result['profile_id']
            ).single().execute()
            
            searchable_text = profile.data['searchable_text']
            
            document_id = await self.add_document(
                user_id=user_id,
                title=data.get('title', category),
                content=searchable_text,
                metadata={'source': 'profile_data', 'category': category, 'profile_data': data},
                tags=[category],
                model_names=model_names
            )
            
            self.supabase.table('profile_data').update({
                'document_id': document_id
            }).eq('id', result['profile_id']).execute()
            
            result['document_id'] = document_id
        
        return result
    
    async def update_profile_data(
        self,
        profile_id: str,
        data: Optional[Dict] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_current: Optional[bool] = None,
        is_featured: Optional[bool] = None,
        display_order: Optional[int] = None,
        recreate_embeddings: bool = True
    ) -> bool:
        """Update profile data and optionally recreate embeddings"""
        profile = self.supabase.table('profile_data').select('*').eq('id', profile_id).single().execute()
        
        if not profile.data:
            raise ValueError(f"Profile data {profile_id} not found")
        
        current_data = profile.data
        
        update_data = {}
        if data:
            updated_data = {**current_data['data'], **data}
            update_data['data'] = updated_data
        if start_date is not None:
            update_data['start_date'] = start_date
        if end_date is not None:
            update_data['end_date'] = end_date
        if is_current is not None:
            update_data['is_current'] = is_current
        if is_featured is not None:
            update_data['is_featured'] = is_featured
        if display_order is not None:
            update_data['display_order'] = display_order
        
        self.supabase.table('profile_data').update(update_data).eq('id', profile_id).execute()
        
        if current_data['document_id'] and recreate_embeddings and data:
            updated_profile = self.supabase.table('profile_data').select(
                'searchable_text'
            ).eq('id', profile_id).single().execute()
            
            searchable_text = updated_profile.data['searchable_text']
            
            await self.update_document(
                document_id=current_data['document_id'],
                content=searchable_text,
                recreate_embeddings=True
            )
        
        return True
    
    def delete_profile_data(self, profile_id: str) -> bool:
        """Delete profile data"""
        profile = self.supabase.table('profile_data').select('document_id').eq('id', profile_id).single().execute()
        
        if profile.data and profile.data['document_id']:
            self.delete_document(profile.data['document_id'], soft_delete=False)
        else:
            self.supabase.table('profile_data').delete().eq('id', profile_id).execute()
        
        return True
    
    def get_profile_data(self, profile_id: str) -> Optional[Dict]:
        """Get profile data by ID"""
        result = self.supabase.table('profile_data').select('*').eq('id', profile_id).execute()
        return result.data[0] if result.data else None
    
    def get_profile_data_list(
        self,
        user_id: str,
        category: Optional[str] = None,
        is_current: Optional[bool] = None,
        is_featured: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get profile data with optional filtering"""
        query = self.supabase.table('profile_data').select('*').eq('user_id', user_id)
        
        if category:
            query = query.eq('category', category)
        if is_current is not None:
            query = query.eq('is_current', is_current)
        if is_featured is not None:
            query = query.eq('is_featured', is_featured)
        
        result = query.order('display_order', desc=False).order('created_at', desc=True).limit(limit).range(offset, offset + limit - 1).execute()
        return result.data
    
    # ========================================================================
    # PERSONAL ATTRIBUTES OPERATIONS
    # ========================================================================
    
    async def add_personal_attribute(
        self,
        user_id: str,
        attribute_type: Literal['soft_skill', 'value', 'worldview', 'aspiration', 'principle'],
        title: str,
        description: str,
        examples: List[str] = None,
        importance_score: Optional[int] = None,
        confidence_level: Optional[int] = None,
        related_articles: List[str] = None,
        related_experiences: List[str] = None,
        searchable: bool = True,
        model_names: List[str] = None
    ) -> Dict[str, str]:
        """Add personal attribute with optional document/embedding"""
        result = {'attribute_id': None, 'document_id': None}
        
        attr_result = self.supabase.table('personal_attributes').insert({
            'user_id': user_id,
            'attribute_type': attribute_type,
            'title': title,
            'description': description,
            'examples': examples or [],
            'importance_score': importance_score,
            'confidence_level': confidence_level,
            'related_articles': related_articles or [],
            'related_experiences': related_experiences or []
        }).execute()
        
        result['attribute_id'] = attr_result.data[0]['id']
        
        if searchable:
            attr = self.supabase.table('personal_attributes').select('searchable_text').eq(
                'id', result['attribute_id']
            ).single().execute()
            
            searchable_text = attr.data['searchable_text']
            
            document_id = await self.add_document(
                user_id=user_id,
                title=title,
                content=searchable_text,
                metadata={
                    'source': 'personal_attribute',
                    'attribute_type': attribute_type,
                    'importance_score': importance_score,
                    'confidence_level': confidence_level
                },
                tags=[attribute_type],
                model_names=model_names
            )
            
            self.supabase.table('personal_attributes').update({
                'document_id': document_id
            }).eq('id', result['attribute_id']).execute()
            
            result['document_id'] = document_id
        
        return result
    
    # ========================================================================
    # SQL FUNCTION: add_personal_attribute
    # ========================================================================
    
    def add_personal_attribute_rpc_function(
        self,
        user_id: str,
        attribute_type: str,
        title: str,
        description: str,
        examples: List[str] = None,
        importance_score: Optional[int] = None,
        confidence_level: Optional[int] = None,
        related_articles: List[str] = None,
        related_experiences: List[str] = None,
        model_name: str = 'nomic-embed-text-768',
        create_searchable: bool = True
    ) -> str:
        """
        Call SQL function to add personal attribute.
        Returns attribute_id.
        """
        result = self.supabase.rpc('add_personal_attribute', {
            'p_user_id': user_id,
            'p_attribute_type': attribute_type,
            'p_title': title,
            'p_description': description,
            'p_examples': examples or [],
            'p_importance_score': importance_score,
            'p_confidence_level': confidence_level,
            'p_related_articles': related_articles or [],
            'p_related_experiences': related_experiences or [],
            'p_embedding_model_name': model_name,
            'p_create_searchable': create_searchable
        }).execute()
        
        return result.data
    
    # ========================================================================
    # SQL FUNCTION: update_personal_attribute
    # ========================================================================
    
    def update_personal_attribute_rpc_function(
        self,
        attribute_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        examples: Optional[List[str]] = None,
        importance_score: Optional[int] = None,
        confidence_level: Optional[int] = None,
        related_articles: Optional[List[str]] = None,
        related_experiences: Optional[List[str]] = None,
        recreate_embedding: bool = True
    ) -> bool:
        """
        Call SQL function to update personal attribute.
        Returns success boolean.
        """
        result = self.supabase.rpc('update_personal_attribute', {
            'p_attribute_id': attribute_id,
            'p_title': title,
            'p_description': description,
            'p_examples': examples,
            'p_importance_score': importance_score,
            'p_confidence_level': confidence_level,
            'p_related_articles': related_articles,
            'p_related_experiences': related_experiences,
            'p_recreate_embedding': recreate_embedding
        }).execute()
        
        return result.data
    
    async def update_personal_attribute(
        self,
        attribute_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        examples: Optional[List[str]] = None,
        importance_score: Optional[int] = None,
        confidence_level: Optional[int] = None,
        related_articles: Optional[List[str]] = None,
        related_experiences: Optional[List[str]] = None,
        recreate_embeddings: bool = True
    ) -> bool:
        """Update personal attribute"""
        attr = self.supabase.table('personal_attributes').select('*').eq('id', attribute_id).single().execute()
        
        if not attr.data:
            raise ValueError(f"Personal attribute {attribute_id} not found")
        
        current = attr.data
        
        update_data = {}
        if title is not None:
            update_data['title'] = title
        if description is not None:
            update_data['description'] = description
        if examples is not None:
            update_data['examples'] = examples
        if importance_score is not None:
            update_data['importance_score'] = importance_score
        if confidence_level is not None:
            update_data['confidence_level'] = confidence_level
        if related_articles is not None:
            update_data['related_articles'] = related_articles
        if related_experiences is not None:
            update_data['related_experiences'] = related_experiences
        
        self.supabase.table('personal_attributes').update(update_data).eq('id', attribute_id).execute()
        
        if current['document_id'] and recreate_embeddings and (title or description or examples):
            updated_attr = self.supabase.table('personal_attributes').select(
                'searchable_text', 'title'
            ).eq('id', attribute_id).single().execute()
            
            searchable_text = updated_attr.data['searchable_text']
            new_title = updated_attr.data['title']
            
            await self.update_document(
                document_id=current['document_id'],
                title=new_title,
                content=searchable_text,
                recreate_embeddings=True
            )
        
        return True
    
    def delete_personal_attribute(self, attribute_id: str) -> bool:
        """Delete personal attribute"""
        attr = self.supabase.table('personal_attributes').select('document_id').eq('id', attribute_id).single().execute()
        
        if attr.data and attr.data['document_id']:
            self.delete_document(attr.data['document_id'], soft_delete=False)
        else:
            self.supabase.table('personal_attributes').delete().eq('id', attribute_id).execute()
        
        return True
    
    def get_personal_attribute(self, attribute_id: str) -> Optional[Dict]:
        """Get personal attribute by ID"""
        result = self.supabase.table('personal_attributes').select('*').eq('id', attribute_id).execute()
        return result.data[0] if result.data else None
    
    def get_personal_attributes(
        self,
        user_id: str,
        attribute_type: Optional[str] = None,
        min_importance: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get personal attributes with optional filtering"""
        query = self.supabase.table('personal_attributes').select('*').eq('user_id', user_id)
        
        if attribute_type:
            query = query.eq('attribute_type', attribute_type)
        if min_importance is not None:
            query = query.gte('importance_score', min_importance)
        
        result = query.order('importance_score', desc=True).order('created_at', desc=True).limit(limit).range(offset, offset + limit - 1).execute()
        return result.data
    
    # ========================================================================
    # SEARCH OPERATIONS (Using SQL Functions)
    # ========================================================================
    
    async def search_rpc_function(
        self,
        query: str,
        user_id: str,
        content_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        threshold: float = 0.7,
        limit: int = 10,
        model_name: str = 'nomic-embed-text-768'
    ) -> List[Dict]:
        """
        Search across all content using vector similarity.
        Uses the search_documents SQL function.
        """
        query_embedding = await self.create_embedding(query, model_name)
        
        model = self._models_cache.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        results = self.supabase.rpc('search_documents', {
            'query_embedding': query_embedding,
            'model_id': model.id,
            'match_threshold': threshold,
            'match_count': limit,
            'filter_user_id': user_id,
            'filter_content_types': content_types,
            'filter_tags': tags
        }).execute()
        
        return results.data
    
    async def search_all_rpc_function(
        self,
        query: str,
        user_id: str,
        threshold: float = 0.7,
        limit: int = 10,
        model_name: str = 'nomic-embed-text-768'
    ) -> List[Dict]:
        """
        Simplified search across all content.
        Uses the search_all_content SQL function.
        """
        query_embedding = await self.create_embedding(query, model_name)
        
        results = self.supabase.rpc('search_all_content', {
            'query_embedding': query_embedding,
            'user_id_filter': user_id,
            'match_threshold': threshold,
            'match_count': limit
        }).execute()
        
        return results.data
    
    # ========================================================================
    # SMART UPDATE OPERATIONS
    # ========================================================================
    
    async def smart_update(
        self,
        user_id: str,
        update_description: str,
        new_content: str,
        content_type: Optional[str] = None,
        similarity_threshold: float = 0.85,
        model_name: str = 'nomic-embed-text-768'
    ) -> Dict[str, any]:
        """Intelligently find and update documents based on semantic similarity"""
        matches = await self.search(
            query=update_description,
            user_id=user_id,
            content_types=[content_type] if content_type else None,
            threshold=similarity_threshold,
            limit=3,
            model_name=model_name
        )
        
        if not matches:
            return {
                'success': False,
                'message': 'No matching documents found',
                'matches': []
            }
        
        best_match = matches[0]
        document_id = best_match['document_id']
        
        document = self.get_document(document_id)
        
        update_result = await self._apply_smart_update(
            document=document,
            best_match=best_match,
            new_content=new_content
        )
        
        return {
            'success': True,
            'matched_document_id': document_id,
            'similarity': best_match['similarity'],
            'content_type': best_match['content_type'],
            'article_id': best_match.get('article_id'),
            'profile_data_id': best_match.get('profile_data_id'),
            'personal_attribute_id': best_match.get('personal_attribute_id'),
            'update_result': update_result,
            'all_matches': matches
        }
    
    async def _apply_smart_update(
        self,
        document: Dict,
        best_match: Dict,
        new_content: str
    ) -> Dict:
        """Apply update to the appropriate table based on match type"""
        
        if best_match.get('article_id'):
            await self.update_article(
                article_id=best_match['article_id'],
                content=new_content,
                recreate_embeddings=True
            )
            return {'type': 'article', 'id': best_match['article_id']}
        
        elif best_match.get('profile_data_id'):
            profile = self.get_profile_data(best_match['profile_data_id'])
            updated_data = self._merge_profile_data(profile['data'], new_content)
            
            await self.update_profile_data(
                profile_id=best_match['profile_data_id'],
                data=updated_data,
                recreate_embeddings=True
            )
            return {'type': 'profile_data', 'id': best_match['profile_data_id']}
        
        elif best_match.get('personal_attribute_id'):
            await self.update_personal_attribute(
                attribute_id=best_match['personal_attribute_id'],
                description=new_content,
                recreate_embeddings=True
            )
            return {'type': 'personal_attribute', 'id': best_match['personal_attribute_id']}
        
        else:
            await self.update_document(
                document_id=document['id'],
                content=new_content,
                recreate_embeddings=True
            )
            return {'type': 'document', 'id': document['id']}
    
    async def propose_updates(
        self,
        user_id: str,
        update_request: str,
        model_name: str = 'nomic-embed-text-768'
    ) -> List[Dict]:
        """Find potential updates but don't apply them yet"""
        matches = await self.search(
            query=update_request,
            user_id=user_id,
            threshold=0.75,
            limit=5,
            model_name=model_name
        )
        
        proposals = []
        for match in matches:
            proposal = {
                'document_id': match['document_id'],
                'current_content': match['chunk_text'],
                'similarity': match['similarity'],
                'content_type': match['content_type'],
                'title': match['title'],
                'article_id': match.get('article_id'),
                'profile_data_id': match.get('profile_data_id'),
                'personal_attribute_id': match.get('personal_attribute_id'),
                'proposed_action': 'update'
            }
            proposals.append(proposal)
        
        return proposals
    
    async def apply_confirmed_update(
        self,
        document_id: str,
        new_content: str,
        article_id: Optional[str] = None,
        profile_data_id: Optional[str] = None,
        personal_attribute_id: Optional[str] = None
    ) -> bool:
        """Apply a confirmed update after user approval"""
        
        if article_id:
            await self.update_article(
                article_id=article_id,
                content=new_content,
                recreate_embeddings=True
            )
        elif profile_data_id:
            profile = self.get_profile_data(profile_data_id)
            updated_data = self._merge_profile_data(profile['data'], new_content)
            await self.update_profile_data(
                profile_id=profile_data_id,
                data=updated_data,
                recreate_embeddings=True
            )
        elif personal_attribute_id:
            await self.update_personal_attribute(
                attribute_id=personal_attribute_id,
                description=new_content,
                recreate_embeddings=True
            )
        else:
            await self.update_document(
                document_id=document_id,
                content=new_content,
                recreate_embeddings=True
            )
        
        return True
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _chunk_text(
        self, 
        text: str, 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _flatten_dict_to_text(self, data: Dict) -> str:
        """Convert dictionary to searchable text"""
        parts = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                parts.append(f"{key}: {', '.join(map(str, value))}")
            elif isinstance(value, dict):
                parts.append(f"{key}: {self._flatten_dict_to_text(value)}")
        return '. '.join(parts)
    
    def _merge_profile_data(self, existing_data: Dict, new_content: str) -> Dict:
        """Merge new content with existing profile data"""
        updated_data = existing_data.copy()
        
        if 'description' in updated_data:
            updated_data['description'] = new_content
        elif 'details' in updated_data:
            updated_data['details'] = new_content
        else:
            updated_data['updated_info'] = new_content
        
        return updated_data
    
    def _create_slug(self, title: str) -> str:
        """Create URL-friendly slug from title"""
        import re
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug[:100]
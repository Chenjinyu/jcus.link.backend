# vector_database.py
from typing import List, Dict, Optional, Union, Literal
from dataclasses import dataclass
from supabase import create_client, Client
import openai
from datetime import datetime
import httpx
import json

@dataclass
class EmbeddingModel:
    id: str
    name: str
    provider: str
    model_identifier: str
    dimensions: int
    is_local: bool

class VectorDatabase:
    """
    Comprehensive vector database interface for managing documents,
    embeddings, articles, profile data, and personal attributes.
    """
    
    def __init__(
        self, 
        supabase_url: str, 
        supabase_key: str, 
        openai_key: Optional[str] = None,
        ollama_url: str = 'http://localhost:11434'
    ):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_client = openai.OpenAI(api_key=openai_key) if openai_key else None
        self.ollama_url = ollama_url
        self._models_cache: Dict[str, EmbeddingModel] = {}
        self._content_types_cache: Dict[str, str] = {}
        self._load_models()
        self._load_content_types()
    
    def _load_models(self):
        """Load embedding models from database"""
        result = self.supabase.table('embedding_models').select('*').eq('is_active', True).execute()
        for model_data in result.data:
            self._models_cache[model_data['name']] = EmbeddingModel(**model_data)
    
    def _load_content_types(self):
        """Load content types from database"""
        result = self.supabase.table('content_types').select('*').execute()
        for ct in result.data:
            self._content_types_cache[ct['name']] = ct['id']
    
    # ========================================================================
    # EMBEDDING CREATION
    # ========================================================================
    
    async def create_embedding(
        self, 
        text: str, 
        model_name: str = 'openai-small'
    ) -> List[float]:
        """Create embedding using specified model"""
        model = self._models_cache.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found or not active")
        
        if model.provider == 'openai':
            return await self._create_openai_embedding(text, model)
        elif model.provider == 'ollama':
            return await self._create_ollama_embedding(text, model)
        else:
            raise ValueError(f"Provider {model.provider} not supported")
    
    async def _create_openai_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Create OpenAI embedding"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Provide openai_key in constructor.")
        
        response = self.openai_client.embeddings.create(
            model=model.model_identifier,
            input=text
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
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    async def add_document(
        self,
        user_id: str,
        content_type: str,
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
        
        Args:
            user_id: User identifier
            content_type: Type of content (must exist in content_types table)
            title: Document title
            content: Document content
            metadata: Optional metadata dictionary
            tags: Optional list of tags
            model_names: List of embedding models to use (default: all active)
            chunk_size: Size of chunks for long documents
            chunk_overlap: Overlap between chunks
            
        Returns:
            document_id: UUID of created document
        """
        # Get content type ID
        content_type_id = self._content_types_cache.get(content_type)
        if not content_type_id:
            raise ValueError(f"Content type '{content_type}' not found")
        
        # Default to all active models
        if model_names is None:
            model_names = list(self._models_cache.keys())
        
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
        
        # Chunk content if necessary
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        total_chunks = len(chunks)
        
        # Create embeddings with multiple models
        for model_name in model_names:
            model = self._models_cache[model_name]
            
            for chunk_index, chunk_text in enumerate(chunks):
                # Create embedding
                embedding = await self.create_embedding(chunk_text, model_name)
                
                # Insert embedding
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
        """
        Update document and optionally recreate embeddings.
        
        Args:
            document_id: Document UUID
            title: New title (optional)
            content: New content (optional)
            metadata: New metadata (optional)
            tags: New tags (optional)
            recreate_embeddings: Whether to regenerate embeddings
            chunk_size: Size of chunks if recreating embeddings
            chunk_overlap: Overlap between chunks
            
        Returns:
            True if successful
        """
        # Build update data
        update_data = {}
        if title is not None:
            update_data['title'] = title
        if content is not None:
            update_data['content'] = content
        if metadata is not None:
            update_data['metadata'] = metadata
        if tags is not None:
            update_data['tags'] = tags
        
        # Update document
        self.supabase.table('documents').update(update_data).eq('id', document_id).execute()
        
        # Recreate embeddings if content changed
        if recreate_embeddings and content is not None:
            # Get existing embedding models for this document
            existing = self.supabase.table('embeddings').select(
                'embedding_model_id'
            ).eq('document_id', document_id).execute()
            
            model_ids = list(set([e['embedding_model_id'] for e in existing.data]))
            
            # Delete old embeddings
            self.supabase.table('embeddings').delete().eq('document_id', document_id).execute()
            
            # Chunk new content
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            
            # Create new embeddings
            for model_id in model_ids:
                # Find model name by ID
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
        """
        Delete document (soft or hard delete).
        
        Args:
            document_id: Document UUID
            soft_delete: If True, marks as deleted; if False, permanently deletes
            
        Returns:
            True if successful
        """
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
        content_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get documents with optional filtering.
        
        Args:
            user_id: User identifier
            content_type: Filter by content type
            tags: Filter by tags (documents with ANY of these tags)
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of documents
        """
        query = self.supabase.table('documents').select(
            '*, content_types(name)'
        ).eq('user_id', user_id).eq('is_current', True).is_('deleted_at', 'null')
        
        if content_type:
            content_type_id = self._content_types_cache.get(content_type)
            if content_type_id:
                query = query.eq('content_type_id', content_type_id)
        
        if tags:
            query = query.overlaps('tags', tags)
        
        result = query.order('created_at', desc=True).limit(limit).range(offset, offset + limit - 1).execute()
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
        model_names: List[str] = None,
        chunk_size: int = 500
    ) -> Dict[str, str]:
        """
        Add article with document and embeddings.
        
        Returns:
            Dict with 'article_id' and 'document_id'
        """
        # Create document with embeddings
        document_id = await self.add_document(
            user_id=user_id,
            content_type='article',
            title=title,
            content=content,
            tags=tags,
            model_names=model_names,
            chunk_size=chunk_size
        )
        
        # Create slug
        slug = self._create_slug(title)
        
        # Create article entry
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
        recreate_embeddings: bool = True
    ) -> bool:
        """Update article and optionally recreate embeddings"""
        # Get article to find document_id
        article = self.supabase.table('articles').select('document_id').eq('id', article_id).single().execute()
        
        if not article.data:
            raise ValueError(f"Article {article_id} not found")
        
        document_id = article.data['document_id']
        
        # Update article
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
            if status == 'published' and 'published_at' not in article_update:
                article_update['published_at'] = datetime.now().isoformat()
        
        self.supabase.table('articles').update(article_update).eq('id', article_id).execute()
        
        # Update document
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
        # Get article to find document_id
        article = self.supabase.table('articles').select('document_id').eq('id', article_id).single().execute()
        
        if article.data:
            # Delete document (CASCADE will delete article)
            self.delete_document(article.data['document_id'], soft_delete=soft_delete)
        
        return True
    
    def get_article(self, article_id: str) -> Optional[Dict]:
        """Get article by ID"""
        result = self.supabase.table('articles').select('*').eq('id', article_id).execute()
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
    
    # ========================================================================
    # PROFILE DATA OPERATIONS
    # ========================================================================
    
    async def add_profile_data(
        self,
        user_id: str,
        category: Literal['work_experience', 'education', 'certification', 'skill', 'value', 'goal'],
        data: Dict,
        searchable: bool = True,
        model_names: List[str] = None
    ) -> Dict[str, str]:
        """
        Add profile data with optional document/embedding.
        
        Returns:
            Dict with 'profile_id' and optionally 'document_id'
        """
        result = {'profile_id': None, 'document_id': None}
        
        # Create searchable text
        searchable_text = self._flatten_dict_to_text(data)
        
        # Create document only if searchable
        document_id = None
        if searchable:
            document_id = await self.add_document(
                user_id=user_id,
                content_type=category,
                title=data.get('title', category),
                content=searchable_text,
                metadata={'profile_data': data},
                model_names=model_names
            )
            result['document_id'] = document_id
        
        # Insert profile data
        profile_result = self.supabase.table('profile_data').insert({
            'user_id': user_id,
            'category': category,
            'data': data,
            'document_id': document_id,
            'searchable_text': searchable_text,
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'is_current': data.get('current', True)
        }).execute()
        
        result['profile_id'] = profile_result.data[0]['id']
        return result
    
    async def update_profile_data(
        self,
        profile_id: str,
        data: Optional[Dict] = None,
        searchable: Optional[bool] = None,
        recreate_embeddings: bool = True
    ) -> bool:
        """Update profile data and optionally recreate embeddings"""
        # Get current profile
        profile = self.supabase.table('profile_data').select('*').eq('id', profile_id).single().execute()
        
        if not profile.data:
            raise ValueError(f"Profile data {profile_id} not found")
        
        current_data = profile.data
        
        # Update data
        if data:
            updated_data = {**current_data['data'], **data}
            searchable_text = self._flatten_dict_to_text(updated_data)
            
            self.supabase.table('profile_data').update({
                'data': updated_data,
                'searchable_text': searchable_text
            }).eq('id', profile_id).execute()
            
            # Update document if exists
            if current_data['document_id'] and recreate_embeddings:
                await self.update_document(
                    document_id=current_data['document_id'],
                    content=searchable_text,
                    recreate_embeddings=True
                )
        
        return True
    
    def delete_profile_data(self, profile_id: str) -> bool:
        """Delete profile data (and associated document if exists)"""
        # Get profile to find document_id
        profile = self.supabase.table('profile_data').select('document_id').eq('id', profile_id).single().execute()
        
        if profile.data and profile.data['document_id']:
            # Delete document (CASCADE will delete profile_data)
            self.delete_document(profile.data['document_id'], soft_delete=False)
        else:
            # Delete profile directly
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
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get profile data with optional filtering"""
        query = self.supabase.table('profile_data').select('*').eq('user_id', user_id)
        
        if category:
            query = query.eq('category', category)
        if is_current is not None:
            query = query.eq('is_current', is_current)
        
        result = query.order('created_at', desc=True).limit(limit).range(offset, offset + limit - 1).execute()
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
        """
        Add personal attribute with optional document/embedding.
        
        Returns:
            Dict with 'attribute_id' and optionally 'document_id'
        """
        result = {'attribute_id': None, 'document_id': None}
        
        # Build searchable text
        searchable_text = f"{title}. {description}"
        if examples:
            searchable_text += ". Examples: " + ". ".join(examples)
        
        # Create document only if searchable
        document_id = None
        if searchable:
            document_id = await self.add_document(
                user_id=user_id,
                content_type=attribute_type,
                title=title,
                content=searchable_text,
                metadata={
                    'attribute_type': attribute_type,
                    'importance_score': importance_score,
                    'confidence_level': confidence_level
                },
                model_names=model_names
            )
            result['document_id'] = document_id
        
        # Insert personal attribute
        attr_result = self.supabase.table('personal_attributes').insert({
            'user_id': user_id,
            'document_id': document_id,
            'attribute_type': attribute_type,
            'title': title,
            'description': description,
            'examples': examples or [],
            'searchable_text': searchable_text,
            'importance_score': importance_score,
            'confidence_level': confidence_level,
            'related_articles': related_articles or [],
            'related_experiences': related_experiences or []
        }).execute()
        
        result['attribute_id'] = attr_result.data[0]['id']
        return result
    
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
        """Update personal attribute and optionally recreate embeddings"""
        # Get current attribute
        attr = self.supabase.table('personal_attributes').select('*').eq('id', attribute_id).single().execute()
        
        if not attr.data:
            raise ValueError(f"Personal attribute {attribute_id} not found")
        
        current = attr.data
        
        # Build update
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
        
        # Update personal_attributes
        self.supabase.table('personal_attributes').update(update_data).eq('id', attribute_id).execute()
        
        # Update document if exists and content changed
        if current['document_id'] and (title is not None or description is not None or examples is not None):
            # Get updated attribute
            updated_attr = self.supabase.table('personal_attributes').select('*').eq('id', attribute_id).single().execute()
            attr_data = updated_attr.data
            
            # Build new searchable text
            new_searchable_text = f"{attr_data['title']}. {attr_data['description']}"
            if attr_data['examples']:
                new_searchable_text += ". Examples: " + ". ".join(attr_data['examples'])
            
            # Update document
            await self.update_document(
                document_id=current['document_id'],
                title=attr_data['title'],
                content=new_searchable_text,
                recreate_embeddings=recreate_embeddings
            )
            
            # Update searchable_text
            self.supabase.table('personal_attributes').update({
                'searchable_text': new_searchable_text
            }).eq('id', attribute_id).execute()
        
        return True
    
    def delete_personal_attribute(self, attribute_id: str) -> bool:
        """Delete personal attribute (and associated document if exists)"""
        # Get attribute to find document_id
        attr = self.supabase.table('personal_attributes').select('document_id').eq('id', attribute_id).single().execute()
        
        if attr.data and attr.data['document_id']:
            # Delete document (CASCADE will delete personal_attributes)
            self.delete_document(attr.data['document_id'], soft_delete=False)
        else:
            # Delete attribute directly
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
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get personal attributes with optional filtering"""
        query = self.supabase.table('personal_attributes').select('*').eq('user_id', user_id)
        
        if attribute_type:
            query = query.eq('attribute_type', attribute_type)
        
        result = query.order('created_at', desc=True).limit(limit).range(offset, offset + limit - 1).execute()
        return result.data
    
    # ------------------ Semantic Search to Find Matching Documents ------------------
    async def smart_update(
        self,
        user_id: str,
        update_description: str,
        new_content: str,
        content_type: Optional[str] = None,
        similarity_threshold: float = 0.85,
        model_name: str = 'openai-small'
    ) -> Dict[str, any]:
        """
        Intelligently find and update documents based on semantic similarity.
        
        Args:
            user_id: User identifier
            update_description: Natural language description of what to update
                                e.g., "Update my current job information"
            new_content: The new/updated content
            content_type: Optional filter by content type
            similarity_threshold: How similar content must be to match (0-1)
            model_name: Embedding model to use
            
        Returns:
            Dict with update results and matched documents
        """
        # Step 1: Find matching documents using semantic search
        matches = await self.search(
            query=update_description,
            user_id=user_id,
            content_types=[content_type] if content_type else None,
            threshold=similarity_threshold,
            limit=3,  # Get top 3 matches
            model_name=model_name
        )
        
        if not matches:
            return {
                'success': False,
                'message': 'No matching documents found',
                'matches': []
            }
        
        # Step 2: Determine best match
        best_match = matches[0]
        document_id = best_match['document_id']
        
        # Step 3: Get full document details
        document = self.get_document(document_id)
        
        # Step 4: Determine update strategy based on content type
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
            'all_matches': matches  # Return all matches for user confirmation
        }
    
    async def _apply_smart_update(
        self,
        document: Dict,
        best_match: Dict,
        new_content: str
    ) -> Dict:
        """Apply update to the appropriate table based on match type"""
        
        # Update article
        if best_match.get('article_id'):
            await self.update_article(
                article_id=best_match['article_id'],
                content=new_content,
                recreate_embeddings=True
            )
            return {'type': 'article', 'id': best_match['article_id']}
        
        # Update profile data
        elif best_match.get('profile_data_id'):
            # Parse new content and merge with existing data
            profile = self.get_profile_data(best_match['profile_data_id'])
            updated_data = self._merge_profile_data(profile['data'], new_content)
            
            await self.update_profile_data(
                profile_id=best_match['profile_data_id'],
                data=updated_data,
                recreate_embeddings=True
            )
            return {'type': 'profile_data', 'id': best_match['profile_data_id']}
        
        # Update personal attribute
        elif best_match.get('personal_attribute_id'):
            await self.update_personal_attribute(
                attribute_id=best_match['personal_attribute_id'],
                description=new_content,
                recreate_embeddings=True
            )
            return {'type': 'personal_attribute', 'id': best_match['personal_attribute_id']}
        
        # Update document directly
        else:
            await self.update_document(
                document_id=document['id'],
                content=new_content,
                recreate_embeddings=True
            )
            return {'type': 'document', 'id': document['id']}
    
    def _merge_profile_data(self, existing_data: Dict, new_content: str) -> Dict:
        """
        Merge new content with existing profile data.
        Uses AI to extract structured updates from natural language.
        """
        # Simple merge - update description field
        # In production, you'd use AI to parse new_content into structured data
        updated_data = existing_data.copy()
        
        # Try to intelligently update based on keywords
        if 'description' in updated_data:
            updated_data['description'] = new_content
        elif 'details' in updated_data:
            updated_data['details'] = new_content
        else:
            # Add as new field
            updated_data['updated_info'] = new_content
        
        return updated_data
    
    # ========================================================================
    # AI-ASSISTED UPDATE (USE LLM TO PARSE INTENT)
    # ========================================================================
    
    async def ai_assisted_update(
        self,
        user_id: str,
        conversation_context: str,
        model_name: str = 'openai-small'
    ) -> Dict:
        """
        Use AI to understand what the user wants to update and apply changes.
        
        Args:
            user_id: User identifier
            conversation_context: Full conversation context from chat
            model_name: Embedding model to use
            
        Returns:
            Dict with update results
        """
        # Step 1: Use AI to extract update intent
        update_intent = await self._extract_update_intent(conversation_context)
        
        if not update_intent:
            return {
                'success': False,
                'message': 'Could not determine update intent'
            }
        
        # Step 2: Find matching documents
        matches = await self.search(
            query=update_intent['search_query'],
            user_id=user_id,
            content_types=update_intent.get('content_types'),
            threshold=0.8,
            limit=3,
            model_name=model_name
        )
        
        if not matches:
            return {
                'success': False,
                'message': 'No matching content found',
                'intent': update_intent
            }
        
        # Step 3: Apply updates
        results = []
        for match in matches[:update_intent.get('max_updates', 1)]:
            document = self.get_document(match['document_id'])
            result = await self._apply_smart_update(
                document=document,
                best_match=match,
                new_content=update_intent['new_content']
            )
            results.append(result)
        
        return {
            'success': True,
            'intent': update_intent,
            'updates_applied': results,
            'matches': matches
        }
    
    async def _extract_update_intent(self, conversation_context: str) -> Optional[Dict]:
        """
        Use OpenAI to extract structured update intent from conversation.
        
        Returns:
            Dict with:
            - search_query: What to search for
            - new_content: The updated content
            - content_types: Which types to update
            - max_updates: How many items to update
        """
        if not self.openai_client:
            return None
        
        system_prompt = """
        You are a data update assistant. Extract update intent from conversations.
        
        Return a JSON object with:
        {
            "search_query": "what to search for (be specific)",
            "new_content": "the new/updated information",
            "content_types": ["relevant content types"],
            "max_updates": 1,
            "confidence": 0.0-1.0
        }
        
        Content types: article, work_experience, education, skill, value, worldview, aspiration, principle
        
        If no clear update intent, return null.
        """
        
        response = self.openai_client.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': conversation_context}
            ],
            response_format={'type': 'json_object'}
        )
        
        try:
            intent = json.loads(response.choices[0].message.content)
            if intent and intent.get('confidence', 0) > 0.7:
                return intent
        except:
            pass
        
        return None
    # ------------------ BATCH UPDATE WITH CONFIRMATION ------------------
    
    async def detect_changes(
        self,
        user_id: str,
        new_information: str,
        model_name: str = 'openai-small'
    ) -> Dict:
        """
        Detect what information changed by comparing with existing content.
        
        Args:
            user_id: User identifier
            new_information: New information from AI conversation
            
        Returns:
            Dict with detected changes and affected documents
        """
        # Step 1: Search for related existing content
        related = await self.search(
            query=new_information,
            user_id=user_id,
            threshold=0.7,
            limit=10,
            model_name=model_name
        )
        
        # Step 2: Use AI to detect what changed
        changes = await self._detect_semantic_changes(
            new_information=new_information,
            existing_content=related
        )
        
        return {
            'new_information': new_information,
            'detected_changes': changes,
            'affected_documents': related
        }
    
    async def _detect_semantic_changes(
        self,
        new_information: str,
        existing_content: List[Dict]
    ) -> List[Dict]:
        """
        Use AI to detect what specifically changed.
        """
        if not self.openai_client or not existing_content:
            return []
        
        # Build comparison prompt
        existing_text = "\n\n".join([
            f"[{c['content_type']}] {c['title']}: {c['chunk_text'][:200]}"
            for c in existing_content[:5]
        ])
        
        system_prompt = """
        Compare new information with existing content and identify changes.
        
        Return JSON array of changes:
        [
            {
                "change_type": "update|add|contradiction",
                "field": "what changed (e.g., job_title, skill_level)",
                "old_value": "previous value",
                "new_value": "new value",
                "affected_content_type": "work_experience|skill|etc",
                "confidence": 0.0-1.0
            }
        ]
        """
        
        user_prompt = f"""
        New Information:
        {new_information}
        
        Existing Content:
        {existing_text}
        
        What changed?
        """
        
        response = self.openai_client.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            response_format={'type': 'json_object'}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get('changes', [])
        except:
            return []
    
    async def apply_detected_changes(
        self,
        user_id: str,
        changes: List[Dict],
        auto_apply: bool = False
    ) -> List[Dict]:
        """
        Apply detected changes to database.
        
        Args:
            user_id: User identifier
            changes: List of detected changes
            auto_apply: If True, applies without confirmation
            
        Returns:
            List of applied updates
        """
        applied = []
        
        for change in changes:
            if change.get('confidence', 0) < 0.8 and not auto_apply:
                continue  # Skip low-confidence changes unless auto_apply
            
            # Find affected document
            content_type = change.get('affected_content_type')
            search_query = f"{change.get('field')} {change.get('old_value', '')}"
            
            matches = await self.search(
                query=search_query,
                user_id=user_id,
                content_types=[content_type] if content_type else None,
                threshold=0.85,
                limit=1
            )
            
            if matches:
                # Apply update
                result = await self.smart_update(
                    user_id=user_id,
                    update_description=search_query,
                    new_content=change.get('new_value', ''),
                    content_type=content_type
                )
                
                applied.append({
                    'change': change,
                    'update_result': result
                })
        
        return applied
    
    # ------------------ Semantic Search to Find Matching Documents END------------------
    # ------------------ Content Fingerprinting & Change Detection ------------------
    async def detect_changes_2(
        self,
        user_id: str,
        new_information: str,
        model_name: str = 'openai-small'
    ) -> Dict:
        """
        Detect what information changed by comparing with existing content.
        
        Args:
            user_id: User identifier
            new_information: New information from AI conversation
            
        Returns:
            Dict with detected changes and affected documents
        """
        # Step 1: Search for related existing content
        related = await self.search(
            query=new_information,
            user_id=user_id,
            threshold=0.7,
            limit=10,
            model_name=model_name
        )
        
        # Step 2: Use AI to detect what changed
        changes = await self._detect_semantic_changes(
            new_information=new_information,
            existing_content=related
        )
        
        return {
            'new_information': new_information,
            'detected_changes': changes,
            'affected_documents': related
        }
    
    async def _detect_semantic_changes_2(
        self,
        new_information: str,
        existing_content: List[Dict]
    ) -> List[Dict]:
        """
        Use AI to detect what specifically changed.
        """
        if not self.openai_client or not existing_content:
            return []
        
        # Build comparison prompt
        existing_text = "\n\n".join([
            f"[{c['content_type']}] {c['title']}: {c['chunk_text'][:200]}"
            for c in existing_content[:5]
        ])
        
        system_prompt = """
        Compare new information with existing content and identify changes.
        
        Return JSON array of changes:
        [
            {
                "change_type": "update|add|contradiction",
                "field": "what changed (e.g., job_title, skill_level)",
                "old_value": "previous value",
                "new_value": "new value",
                "affected_content_type": "work_experience|skill|etc",
                "confidence": 0.0-1.0
            }
        ]
        """
        
        user_prompt = f"""
        New Information:
        {new_information}
        
        Existing Content:
        {existing_text}
        
        What changed?
        """
        
        response = self.openai_client.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            response_format={'type': 'json_object'}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get('changes', [])
        except:
            return []
    
    async def apply_detected_changes_2(
        self,
        user_id: str,
        changes: List[Dict],
        auto_apply: bool = False
    ) -> List[Dict]:
        """
        Apply detected changes to database.
        
        Args:
            user_id: User identifier
            changes: List of detected changes
            auto_apply: If True, applies without confirmation
            
        Returns:
            List of applied updates
        """
        applied = []
        
        for change in changes:
            if change.get('confidence', 0) < 0.8 and not auto_apply:
                continue  # Skip low-confidence changes unless auto_apply
            
            # Find affected document
            content_type = change.get('affected_content_type')
            search_query = f"{change.get('field')} {change.get('old_value', '')}"
            
            matches = await self.search(
                query=search_query,
                user_id=user_id,
                content_types=[content_type] if content_type else None,
                threshold=0.85,
                limit=1
            )
            
            if matches:
                # Apply update
                result = await self.smart_update(
                    user_id=user_id,
                    update_description=search_query,
                    new_content=change.get('new_value', ''),
                    content_type=content_type
                )
                
                applied.append({
                    'change': change,
                    'update_result': result
                })
        
        return applied
    # ------------------ Content Fingerprinting & Change Detection END ------------------
    
    # ========================================================================
    # BATCH UPDATE WITH CONFIRMATION
    # ========================================================================
    
    async def propose_updates(
        self,
        user_id: str,
        update_request: str,
        model_name: str = 'openai-small'
    ) -> List[Dict]:
        """
        Find potential updates but don't apply them yet.
        Returns proposals for user confirmation.
        
        Args:
            user_id: User identifier
            update_request: Natural language update request
            
        Returns:
            List of proposed updates with match details
        """
        # Find matching documents
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
        """
        Apply a confirmed update after user approval.
        """
        # Update based on type
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
    # INTELLIGENT UPDATE METHODS (NO ID REQUIRED)
    # ========================================================================
    
    async def smart_update(
        self,
        user_id: str,
        update_description: str,
        new_content: str,
        content_type: Optional[str] = None,
        similarity_threshold: float = 0.85,
        model_name: str = 'openai-small'
    ) -> Dict[str, any]:
        """
        Intelligently find and update documents based on semantic similarity.
        
        Args:
            user_id: User identifier
            update_description: Natural language description of what to update
                                e.g., "Update my current job information"
            new_content: The new/updated content
            content_type: Optional filter by content type
            similarity_threshold: How similar content must be to match (0-1)
            model_name: Embedding model to use
            
        Returns:
            Dict with update results and matched documents
        """
        # Step 1: Find matching documents using semantic search
        matches = await self.search(
            query=update_description,
            user_id=user_id,
            content_types=[content_type] if content_type else None,
            threshold=similarity_threshold,
            limit=3,  # Get top 3 matches
            model_name=model_name
        )
        
        if not matches:
            return {
                'success': False,
                'message': 'No matching documents found',
                'matches': []
            }
        
        # Step 2: Determine best match
        best_match = matches[0]
        document_id = best_match['document_id']
        
        # Step 3: Get full document details
        document = self.get_document(document_id)
        
        # Step 4: Determine update strategy based on content type
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
            'all_matches': matches  # Return all matches for user confirmation
        }
    
    async def _apply_smart_update(
        self,
        document: Dict,
        best_match: Dict,
        new_content: str
    ) -> Dict:
        """Apply update to the appropriate table based on match type"""
        
        # Update article
        if best_match.get('article_id'):
            await self.update_article(
                article_id=best_match['article_id'],
                content=new_content,
                recreate_embeddings=True
            )
            return {'type': 'article', 'id': best_match['article_id']}
        
        # Update profile data
        elif best_match.get('profile_data_id'):
            # Parse new content and merge with existing data
            profile = self.get_profile_data(best_match['profile_data_id'])
            updated_data = self._merge_profile_data(profile['data'], new_content)
            
            await self.update_profile_data(
                profile_id=best_match['profile_data_id'],
                data=updated_data,
                recreate_embeddings=True
            )
            return {'type': 'profile_data', 'id': best_match['profile_data_id']}
        
        # Update personal attribute
        elif best_match.get('personal_attribute_id'):
            await self.update_personal_attribute(
                attribute_id=best_match['personal_attribute_id'],
                description=new_content,
                recreate_embeddings=True
            )
            return {'type': 'personal_attribute', 'id': best_match['personal_attribute_id']}
        
        # Update document directly
        else:
            await self.update_document(
                document_id=document['id'],
                content=new_content,
                recreate_embeddings=True
            )
            return {'type': 'document', 'id': document['id']}
    
    def _merge_profile_data(self, existing_data: Dict, new_content: str) -> Dict:
        """
        Merge new content with existing profile data.
        Uses AI to extract structured updates from natural language.
        """
        # Simple merge - update description field
        # In production, you'd use AI to parse new_content into structured data
        updated_data = existing_data.copy()
        
        # Try to intelligently update based on keywords
        if 'description' in updated_data:
            updated_data['description'] = new_content
        elif 'details' in updated_data:
            updated_data['details'] = new_content
        else:
            # Add as new field
            updated_data['updated_info'] = new_content
        
        return updated_data
    
    # ========================================================================
    # AI-ASSISTED UPDATE (USE LLM TO PARSE INTENT)
    # ========================================================================
    
    async def ai_assisted_update(
        self,
        user_id: str,
        conversation_context: str,
        model_name: str = 'openai-small'
    ) -> Dict:
        """
        Use AI to understand what the user wants to update and apply changes.
        
        Args:
            user_id: User identifier
            conversation_context: Full conversation context from chat
            model_name: Embedding model to use
            
        Returns:
            Dict with update results
        """
        # Step 1: Use AI to extract update intent
        update_intent = await self._extract_update_intent(conversation_context)
        
        if not update_intent:
            return {
                'success': False,
                'message': 'Could not determine update intent'
            }
        
        # Step 2: Find matching documents
        matches = await self.search(
            query=update_intent['search_query'],
            user_id=user_id,
            content_types=update_intent.get('content_types'),
            threshold=0.8,
            limit=3,
            model_name=model_name
        )
        
        if not matches:
            return {
                'success': False,
                'message': 'No matching content found',
                'intent': update_intent
            }
        
        # Step 3: Apply updates
        results = []
        for match in matches[:update_intent.get('max_updates', 1)]:
            document = self.get_document(match['document_id'])
            result = await self._apply_smart_update(
                document=document,
                best_match=match,
                new_content=update_intent['new_content']
            )
            results.append(result)
        
        return {
            'success': True,
            'intent': update_intent,
            'updates_applied': results,
            'matches': matches
        }
    
    async def _extract_update_intent(self, conversation_context: str) -> Optional[Dict]:
        """
        Use OpenAI to extract structured update intent from conversation.
        
        Returns:
            Dict with:
            - search_query: What to search for
            - new_content: The updated content
            - content_types: Which types to update
            - max_updates: How many items to update
        """
        if not self.openai_client:
            return None
        
        system_prompt = """
        You are a data update assistant. Extract update intent from conversations.
        
        Return a JSON object with:
        {
            "search_query": "what to search for (be specific)",
            "new_content": "the new/updated information",
            "content_types": ["relevant content types"],
            "max_updates": 1,
            "confidence": 0.0-1.0
        }
        
        Content types: article, work_experience, education, skill, value, worldview, aspiration, principle
        
        If no clear update intent, return null.
        """
        
        response = self.openai_client.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': conversation_context}
            ],
            response_format={'type': 'json_object'}
        )
        
        try:
            intent = json.loads(response.choices[0].message.content)
            if intent and intent.get('confidence', 0) > 0.7:
                return intent
        except:
            pass
        
        return None
    
    # ========================================================================
    # BATCH UPDATE WITH CONFIRMATION
    # ========================================================================
    
    async def propose_updates(
        self,
        user_id: str,
        update_request: str,
        model_name: str = 'openai-small'
    ) -> List[Dict]:
        """
        Find potential updates but don't apply them yet.
        Returns proposals for user confirmation.
        
        Args:
            user_id: User identifier
            update_request: Natural language update request
            
        Returns:
            List of proposed updates with match details
        """
        # Find matching documents
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
        """
        Apply a confirmed update after user approval.
        """
        # Update based on type
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
    # SEARCH OPERATIONS
    # ========================================================================
    
    async def search(
        self,
        query: str,
        user_id: str,
        content_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        threshold: float = 0.7,
        limit: int = 10,
        model_name: str = 'openai-small'
    ) -> List[Dict]:
        """
        Search across all content using vector similarity.
        
        Args:
            query: Search query text
            user_id: User identifier
            content_types: Filter by content types
            tags: Filter by tags
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results
            model_name: Embedding model to use
            
        Returns:
            List of search results with similarity scores
        """
        # Create query embedding
        query_embedding = await self.create_embedding(query, model_name)
        
        # Get model ID
        model = self._models_cache.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        # Search using database function
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
        
        # If text is short enough, return as single chunk
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
    
    def _create_slug(self, title: str) -> str:
        """Create URL-friendly slug from title"""
        import re
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug[:100]  # Limit length
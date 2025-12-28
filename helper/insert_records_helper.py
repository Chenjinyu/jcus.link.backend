"""
Easy-to-use script for inserting records into the vector database.

This script provides simple functions to insert:
- Documents
- Articles
- Profile Data (work experience, education, skills, etc.)
- Personal Attributes (values, principles, aspirations, etc.)

Usage:
    python examples/insert_records.py
"""

import asyncio
import os
from typing import Dict, List, Optional
import sys

# Add parent directory to path to import vector_database
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.vector_database import VectorDatabase


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_db(
    supabase_url,
    supabase_key,
    ollama_url
    ) -> VectorDatabase:
    """Create and initialize VectorDatabase instance"""
    return VectorDatabase(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        ollama_url=ollama_url
    )

async def insert_document_example(db: VectorDatabase, user_id: str):
    """Example: Insert a simple document"""
    print("\n📄 Inserting Document...")
    
    document_id = await db.add_document(
        user_id=user_id,
        title="Introduction to Python",
        content="""
        Python is a high-level programming language known for its simplicity and readability.
        It was created by Guido van Rossum and first released in 1991.
        Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
        It has a large standard library and an active community.
        """,
        metadata={
            "author": "John Doe",
            "category": "programming",
            "difficulty": "beginner"
        },
        tags=["python", "programming", "tutorial"],
        model_names=None,  # Uses all active models
        chunk_size=500,
        chunk_overlap=50
    )
    
    print(f"✅ Document inserted with ID: {document_id}")
    return document_id


async def insert_article_example(db: VectorDatabase, user_id: str):
    """Example: Insert an article"""
    print("\n📰 Inserting Article...")
    
    result = await db.add_article(
        user_id=user_id,
        title="Getting Started with Vector Databases",
        content="""
        Vector databases are specialized databases designed to store and query high-dimensional vectors.
        They are essential for AI applications that use embeddings for semantic search.
        Popular vector databases include Pinecone, Weaviate, and Supabase with pgvector.
        
        Key features:
        - Fast similarity search
        - Support for high-dimensional vectors
        - Integration with ML/AI pipelines
        """,
        subtitle="A comprehensive guide to vector databases",
        excerpt="Learn about vector databases and their applications in AI",
        tags=["vector-db", "ai", "embeddings"],
        category="technology",
        status="published",  # or "draft"
        model_names=None,
        chunk_size=500
    )
    
    print(f"✅ Article inserted:")
    print(f"   Article ID: {result['article_id']}")
    print(f"   Document ID: {result['document_id']}")
    return result


async def insert_work_experience_example(db: VectorDatabase, user_id: str):
    """Example: Insert work experience"""
    print("\n💼 Inserting Work Experience...")
    
    result = await db.add_profile_data(
        user_id=user_id,
        category="work_experience",
        data={
            "title": "Senior Software Engineer",
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "description": "Led development of microservices architecture. Built REST APIs using FastAPI and Python.",
            "start_date": "2022-01-01",
            "end_date": None,  # Current position
            "current": True,
            "technologies": ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
            "achievements": [
                "Reduced API response time by 40%",
                "Led team of 5 engineers",
                "Implemented CI/CD pipeline"
            ]
        },
        searchable=True,  # Creates embeddings for search
        model_names=None
    )
    
    print(f"✅ Work experience inserted:")
    print(f"   Profile ID: {result['profile_id']}")
    print(f"   Document ID: {result['document_id']}")
    return result


async def insert_education_example(db: VectorDatabase, user_id: str):
    """Example: Insert education"""
    print("\n🎓 Inserting Education...")
    
    result = await db.add_profile_data(
        user_id=user_id,
        category="education",
        data={
            "institution": "University of Technology",
            "degree": "Bachelor of Science in Computer Science",
            "field_of_study": "Computer Science",
            "start_date": "2018-09-01",
            "end_date": "2022-05-15",
            "gpa": 3.8,
            "honors": ["Summa Cum Laude", "Dean's List"],
            "description": "Focused on software engineering and machine learning"
        },
        searchable=True,
        model_names=None
    )
    
    print(f"✅ Education inserted:")
    print(f"   Profile ID: {result['profile_id']}")
    return result


async def insert_skill_example(db: VectorDatabase, user_id: str):
    """Example: Insert skill"""
    print("\n🛠️ Inserting Skill...")
    
    result = await db.add_profile_data(
        user_id=user_id,
        category="skill",
        data={
            "name": "Python Programming",
            "level": "expert",
            "years_of_experience": 5,
            "description": "Proficient in Python with experience in web development, data science, and automation",
            "certifications": ["Python Professional Certification"],
            "projects": ["Built REST APIs", "Data analysis pipelines"]
        },
        searchable=True,
        model_names=None
    )
    
    print(f"✅ Skill inserted:")
    print(f"   Profile ID: {result['profile_id']}")
    return result


async def insert_certification_example(db: VectorDatabase, user_id: str):
    """Example: Insert certification"""
    print("\n🏆 Inserting Certification...")
    
    result = await db.add_profile_data(
        user_id=user_id,
        category="certification",
        data={
            "name": "AWS Certified Solutions Architect",
            "issuer": "Amazon Web Services",
            "issue_date": "2023-06-01",
            "expiry_date": "2026-06-01",
            "credential_id": "AWS-CSA-12345",
            "description": "Demonstrates expertise in designing distributed systems on AWS"
        },
        searchable=True,
        model_names=None
    )
    
    print(f"✅ Certification inserted:")
    print(f"   Profile ID: {result['profile_id']}")
    return result


async def insert_value_example(db: VectorDatabase, user_id: str):
    """Example: Insert personal value"""
    print("\n💎 Inserting Personal Value...")
    
    result = await db.add_personal_attribute(
        user_id=user_id,
        attribute_type="value",
        title="Continuous Learning",
        description="I believe in constantly improving my skills and knowledge. Learning new technologies and methodologies keeps me engaged and helps me solve problems more effectively.",
        examples=[
            "Completed 3 online courses this year",
            "Attend tech meetups regularly",
            "Read technical books weekly"
        ],
        importance_score=9,  # 1-10
        confidence_level=8,  # 1-10
        related_articles=[],
        related_experiences=[],
        searchable=True,
        model_names=None
    )
    
    print(f"✅ Value inserted:")
    print(f"   Attribute ID: {result['attribute_id']}")
    print(f"   Document ID: {result['document_id']}")
    return result


async def insert_principle_example(db: VectorDatabase, user_id: str):
    """Example: Insert principle"""
    print("\n⚖️ Inserting Principle...")
    
    result = await db.add_personal_attribute(
        user_id=user_id,
        attribute_type="principle",
        title="Code Quality Over Speed",
        description="I prioritize writing clean, maintainable code over quick fixes. Technical debt should be addressed proactively.",
        examples=[
            "Always write unit tests",
            "Refactor before adding new features",
            "Code reviews focus on maintainability"
        ],
        importance_score=8,
        confidence_level=9,
        searchable=True,
        model_names=None
    )
    
    print(f"✅ Principle inserted:")
    print(f"   Attribute ID: {result['attribute_id']}")
    return result


async def insert_aspiration_example(db: VectorDatabase, user_id: str):
    """Example: Insert aspiration"""
    print("\n🎯 Inserting Aspiration...")
    
    result = await db.add_personal_attribute(
        user_id=user_id,
        attribute_type="aspiration",
        title="Become a Tech Lead",
        description="I aspire to lead a team of engineers and contribute to architectural decisions. I want to mentor junior developers and drive technical excellence.",
        examples=[
            "Taking on more leadership responsibilities",
            "Mentoring junior developers",
            "Contributing to technical strategy"
        ],
        importance_score=10,
        confidence_level=7,
        searchable=True,
        model_names=None
    )
    
    print(f"✅ Aspiration inserted:")
    print(f"   Attribute ID: {result['attribute_id']}")
    return result


# ============================================================================
# BATCH INSERT FUNCTIONS
# ============================================================================

async def insert_multiple_work_experiences(db: VectorDatabase, user_id: str):
    """Example: Insert multiple work experiences at once"""
    print("\n💼 Inserting Multiple Work Experiences...")
    
    experiences = [
        {
            "title": "Software Engineer",
            "company": "Startup Inc",
            "location": "Remote",
            "description": "Developed web applications using React and Node.js",
            "start_date": "2020-01-01",
            "end_date": "2022-01-01",
            "current": False,
            "technologies": ["React", "Node.js", "TypeScript"]
        },
        {
            "title": "Junior Developer",
            "company": "Web Agency",
            "location": "New York, NY",
            "description": "Built responsive websites and maintained legacy code",
            "start_date": "2018-06-01",
            "end_date": "2019-12-31",
            "current": False,
            "technologies": ["HTML", "CSS", "JavaScript", "PHP"]
        }
    ]
    
    results = []
    for exp in experiences:
        result = await db.add_profile_data(
            user_id=user_id,
            category="work_experience",
            data=exp,
            searchable=True,
            model_names=None
        )
        results.append(result)
        print(f"   ✅ Inserted: {exp['title']} at {exp['company']}")
    
    return results


async def insert_multiple_skills(db: VectorDatabase, user_id: str):
    """Example: Insert multiple skills at once"""
    print("\n🛠️ Inserting Multiple Skills...")
    
    skills = [
        {
            "name": "JavaScript",
            "level": "advanced",
            "years_of_experience": 4,
            "description": "Proficient in modern JavaScript (ES6+), React, and Node.js"
        },
        {
            "name": "Python",
            "level": "expert",
            "years_of_experience": 5,
            "description": "Expert in Python for web development, data science, and automation"
        },
        {
            "name": "PostgreSQL",
            "level": "intermediate",
            "years_of_experience": 3,
            "description": "Experience with database design, optimization, and complex queries"
        }
    ]
    
    results = []
    for skill in skills:
        result = await db.add_profile_data(
            user_id=user_id,
            category="skill",
            data=skill,
            searchable=True,
            model_names=None
        )
        results.append(result)
        print(f"   ✅ Inserted: {skill['name']} ({skill['level']})")
    
    return results


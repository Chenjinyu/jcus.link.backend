"""
Quick Insert Helper - Simplest way to insert records

This is a simplified wrapper for the most common use cases.
"""

import asyncio
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.vector_database import VectorDatabase


class QuickInsert:
    """Simplified interface for quick record insertion"""

    def __init__(
        self, supabase_url: str, supabase_key: str, openai_key: Optional[str] = None
    ):
        self.db = VectorDatabase(
            supabase_url=supabase_url, supabase_key=supabase_key, openai_key=openai_key
        )

    async def document(
        self,
        user_id: str,
        title: str,
        content: str,
        content_type: str = "article",
        tags: List[str] = None,
    ) -> str:
        """Quick insert a document"""
        return await self.db.add_document(
            user_id=user_id,
            content_type=content_type,
            title=title,
            content=content,
            tags=tags or [],
        )

    async def article(
        self, user_id: str, title: str, content: str, status: str = "published"
    ) -> Dict:
        """Quick insert an article"""
        return await self.db.add_article(
            user_id=user_id, title=title, content=content, status=status
        )

    async def job(
        self,
        user_id: str,
        title: str,
        company: str,
        description: str,
        start_date: str,
        end_date: Optional[str] = None,
        technologies: List[str] = None,
    ) -> Dict:
        """Quick insert work experience"""
        return await self.db.add_profile_data(
            user_id=user_id,
            category="work_experience",
            data={
                "title": title,
                "company": company,
                "description": description,
                "start_date": start_date,
                "end_date": end_date,
                "current": end_date is None,
                "technologies": technologies or [],
            },
            searchable=True,
        )

    async def education(
        self,
        user_id: str,
        institution: str,
        degree: str,
        start_date: str,
        end_date: str,
    ) -> Dict:
        """Quick insert education"""
        return await self.db.add_profile_data(
            user_id=user_id,
            category="education",
            data={
                "institution": institution,
                "degree": degree,
                "start_date": start_date,
                "end_date": end_date,
            },
            searchable=True,
        )

    async def skill(self, user_id: str, name: str, level: str, years: int) -> Dict:
        """Quick insert skill"""
        return await self.db.add_profile_data(
            user_id=user_id,
            category="skill",
            data={"name": name, "level": level, "years_of_experience": years},
            searchable=True,
        )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


async def example():
    """Simple usage example"""

    # Initialize
    quick = QuickInsert(
        supabase_url=os.environ.get("SUPABASE_URL", "your-url"),
        supabase_key=os.environ.get("SUPABASE_SERVICE_KEY", "your-key"),
        openai_key=os.environ.get("OPENAI_API_KEY"),
    )

    user_id = "user-123"

    # Insert a job
    result = await quick.job(
        user_id=user_id,
        title="Senior Developer",
        company="Tech Corp",
        description="Built amazing products",
        start_date="2022-01-01",
        technologies=["Python", "FastAPI"],
    )
    print(f"✅ Job inserted: {result['profile_id']}")

    # Insert a skill
    result = await quick.skill(user_id=user_id, name="Python", level="expert", years=5)
    print(f"✅ Skill inserted: {result['profile_id']}")

    # Insert an article
    result = await quick.article(
        user_id=user_id,
        title="My Blog Post",
        content="This is the content of my blog post...",
    )
    print(f"✅ Article inserted: {result['article_id']}")


if __name__ == "__main__":
    asyncio.run(example())

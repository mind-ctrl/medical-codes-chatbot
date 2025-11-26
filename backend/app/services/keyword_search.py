"""
Keyword Search Service
Uses PostgreSQL full-text search (tsvector + GIN index)
"""

from typing import List, Dict, Optional
import logging
from ..database import db

logger = logging.getLogger(__name__)


async def search_cpt_codes_keyword(
    query: str,
    limit: int = 20,
    category: Optional[str] = None
) -> List[Dict]:
    """
    Search CPT codes using full-text search

    PostgreSQL full-text search:
    - Faster than vector search (~10ms vs ~100ms)
    - Catches exact medical terms that embeddings might miss
    - Uses GIN index on tsvector column

    Query processing:
    - "type 2 diabetes" → "type & 2 & diabetes"
    - Matches words with stemming (diabetes matches diabetic)

    Args:
        query: Search query string
        limit: Number of results to return
        category: Optional category filter

    Returns:
        List of CPT code results with ranking scores
    """
    # Create tsquery from search text
    # Split on spaces and join with & for AND logic
    tsquery = ' & '.join(query.split())

    if category:
        sql_query = """
            SELECT
                cpt_code,
                description,
                category,
                ts_rank(description_tsv, to_tsquery('english', $1)) as rank_score
            FROM cpt_codes
            WHERE description_tsv @@ to_tsquery('english', $1)
              AND category = $2
            ORDER BY rank_score DESC
            LIMIT $3
        """
        results = await db.fetch(sql_query, tsquery, category, limit)
    else:
        sql_query = """
            SELECT
                cpt_code,
                description,
                category,
                ts_rank(description_tsv, to_tsquery('english', $1)) as rank_score
            FROM cpt_codes
            WHERE description_tsv @@ to_tsquery('english', $1)
            ORDER BY rank_score DESC
            LIMIT $2
        """
        results = await db.fetch(sql_query, tsquery, limit)

    return [dict(row) for row in results]


async def search_icd10_codes_keyword(
    query: str,
    limit: int = 20,
    chapter: Optional[str] = None
) -> List[Dict]:
    """
    Search ICD-10 codes using full-text search

    Example queries:
    - "diabetes" → finds all diabetes-related codes
    - "type 2 diabetes" → finds type 2 diabetes specifically
    - "diabetes heart" → finds diabetes with heart complications

    Args:
        query: Search query string
        limit: Number of results to return
        chapter: Optional chapter filter (e.g., "E00-E89")

    Returns:
        List of ICD-10 code results with ranking scores
    """
    tsquery = ' & '.join(query.split())

    if chapter:
        sql_query = """
            SELECT
                icd10_code,
                description,
                chapter,
                block,
                ts_rank(description_tsv, to_tsquery('english', $1)) as rank_score
            FROM icd10_codes
            WHERE description_tsv @@ to_tsquery('english', $1)
              AND chapter = $2
            ORDER BY rank_score DESC
            LIMIT $3
        """
        results = await db.fetch(sql_query, tsquery, chapter, limit)
    else:
        sql_query = """
            SELECT
                icd10_code,
                description,
                chapter,
                block,
                ts_rank(description_tsv, to_tsquery('english', $1)) as rank_score
            FROM icd10_codes
            WHERE description_tsv @@ to_tsquery('english', $1)
            ORDER BY rank_score DESC
            LIMIT $2
        """
        results = await db.fetch(sql_query, tsquery, limit)

    return [dict(row) for row in results]

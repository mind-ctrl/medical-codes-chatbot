"""
Vector Similarity Search Service
Uses pgvector for semantic search with cosine similarity
"""

import numpy as np
from typing import List, Dict, Optional
import logging
from ..database import db

logger = logging.getLogger(__name__)


async def search_cpt_codes_vector(
    query_embedding: np.ndarray,
    limit: int = 20,
    category: Optional[str] = None
) -> List[Dict]:
    """
    Search CPT codes using vector similarity

    Uses pgvector's <=> operator for cosine distance:
    - Distance 0 = identical vectors
    - Distance 2 = opposite vectors
    - Converted to similarity (1 - distance) for intuitive scores

    Args:
        query_embedding: Query embedding vector (384-dim)
        limit: Number of results to return
        category: Optional category filter (e.g., "AAA", "AMP")

    Returns:
        List of CPT code results with similarity scores
    """
    # Convert embedding to pgvector string format: '[0.1, 0.2, 0.3]'
    embedding_list = query_embedding.tolist()
    embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'

    if category:
        query = """
            SELECT
                cpt_code,
                description,
                category,
                1 - (embedding <=> $1::vector) as similarity_score
            FROM cpt_codes
            WHERE category = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """
        results = await db.fetch(query, embedding_str, category, limit)
    else:
        query = """
            SELECT
                cpt_code,
                description,
                category,
                1 - (embedding <=> $1::vector) as similarity_score
            FROM cpt_codes
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        results = await db.fetch(query, embedding_str, limit)

    return [dict(row) for row in results]


async def search_icd10_codes_vector(
    query_embedding: np.ndarray,
    limit: int = 20,
    chapter: Optional[str] = None
) -> List[Dict]:
    """
    Search ICD-10 codes using vector similarity

    ICD-10 codes are hierarchical:
    - Chapter: E00-E89 (Endocrine diseases)
    - Block: E08-E13 (Diabetes)
    - Code: E11.9 (Type 2 diabetes without complications)

    Args:
        query_embedding: Query embedding vector (384-dim)
        limit: Number of results to return
        chapter: Optional chapter filter (e.g., "E00-E89")

    Returns:
        List of ICD-10 code results with similarity scores
    """
    # Convert embedding to pgvector string format: '[0.1, 0.2, 0.3]'
    embedding_list = query_embedding.tolist()
    embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'

    if chapter:
        query = """
            SELECT
                icd10_code,
                description,
                chapter,
                block,
                1 - (embedding <=> $1::vector) as similarity_score
            FROM icd10_codes
            WHERE chapter = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """
        results = await db.fetch(query, embedding_str, chapter, limit)
    else:
        query = """
            SELECT
                icd10_code,
                description,
                chapter,
                block,
                1 - (embedding <=> $1::vector) as similarity_score
            FROM icd10_codes
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        results = await db.fetch(query, embedding_str, limit)

    return [dict(row) for row in results]

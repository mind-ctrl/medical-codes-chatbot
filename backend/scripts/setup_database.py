"""
Database Setup Script
Creates tables and indices for medical coding system
"""

import asyncio
import asyncpg
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

CREATE_EXTENSIONS = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

CREATE_CPT_TABLE = """
CREATE TABLE IF NOT EXISTS cpt_codes (
    id SERIAL PRIMARY KEY,
    cpt_code VARCHAR(5) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50),
    code_status VARCHAR(20),

    -- Vector embedding
    embedding vector(384),

    -- Full-text search (auto-generated)
    description_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', description || ' ' || COALESCE(category, ''))
    ) STORED,

    -- Metadata
    usage_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

CREATE_ICD10_TABLE = """
CREATE TABLE IF NOT EXISTS icd10_codes (
    id SERIAL PRIMARY KEY,
    icd10_code VARCHAR(10) UNIQUE NOT NULL,
    description TEXT NOT NULL,

    -- Hierarchy (extracted from code pattern)
    chapter VARCHAR(10),
    block VARCHAR(20),

    -- Vector embedding
    embedding vector(384),

    -- Full-text search
    description_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', description)
    ) STORED,

    -- Metadata
    usage_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

CREATE_INDICES = """
-- CPT Indices
CREATE INDEX IF NOT EXISTS idx_cpt_vector
    ON cpt_codes USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_cpt_fts
    ON cpt_codes USING GIN (description_tsv);

CREATE INDEX IF NOT EXISTS idx_cpt_category
    ON cpt_codes (category);

-- ICD-10 Indices
CREATE INDEX IF NOT EXISTS idx_icd10_vector
    ON icd10_codes USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_icd10_fts
    ON icd10_codes USING GIN (description_tsv);

CREATE INDEX IF NOT EXISTS idx_icd10_chapter
    ON icd10_codes (chapter);
"""


async def setup_database():
    """Setup database schema"""
    db_url = os.getenv('NEON_DATABASE_URL')
    if not db_url:
        raise ValueError("NEON_DATABASE_URL environment variable not set")

    print("Connecting to database...")
    conn = await asyncpg.connect(db_url)

    try:
        print("Creating pgvector extension...")
        await conn.execute(CREATE_EXTENSIONS)

        print("Creating cpt_codes table...")
        await conn.execute(CREATE_CPT_TABLE)

        print("Creating icd10_codes table...")
        await conn.execute(CREATE_ICD10_TABLE)

        print("Creating indices (this may take a moment)...")
        await conn.execute(CREATE_INDICES)

        print("Database setup complete!")

        # Verify
        cpt_count = await conn.fetchval("SELECT COUNT(*) FROM cpt_codes")
        icd10_count = await conn.fetchval("SELECT COUNT(*) FROM icd10_codes")

        print(f"\nCurrent data:")
        print(f"   CPT codes: {cpt_count}")
        print(f"   ICD-10 codes: {icd10_count}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(setup_database())

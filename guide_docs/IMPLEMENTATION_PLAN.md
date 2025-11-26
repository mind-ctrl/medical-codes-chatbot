# Implementation Plan - Step-by-Step Guide

## 📅 Timeline: 8-10 Days to Portfolio-Ready

This guide breaks down the implementation into manageable phases with clear deliverables.

---

## Phase 1: Database Setup (Days 1-2)

### Day 1: Environment & Database Setup

#### 1.1 Create Neon Database Account
```bash
# Go to https://neon.tech
# Sign up with GitHub (free tier)
# Create new project: "medical-coding-rag"
# Copy connection string
```

**Deliverable**: Neon database created, connection string saved

#### 1.2 Project Structure
```bash
medical-coding-rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── request_models.py
│   │   │   └── response_models.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py
│   │   │   ├── vector_search.py
│   │   │   ├── keyword_search.py
│   │   │   ├── hybrid_search.py
│   │   │   ├── llm_service.py
│   │   │   └── ranking.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logger.py
│   │       └── exceptions.py
│   ├── scripts/
│   │   ├── setup_database.py
│   │   ├── load_cpt_codes.py
│   │   └── load_icd10_codes.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   └── (Next.js structure - Phase 4)
├── data/
│   ├── all-2025-cpt-codes.csv
│   └── icd10cm-codes-2025.txt
├── guide_docs/
├── .gitignore
└── README.md
```

**Commands**:
```bash
# Create backend structure
mkdir -p backend/app/{models,services,utils}
mkdir -p backend/scripts
touch backend/app/__init__.py
touch backend/app/main.py
touch backend/app/config.py
touch backend/app/database.py
```

**Deliverable**: Project structure created

#### 1.3 Environment Setup
Create `backend/.env.example`:
```bash
# Database
NEON_DATABASE_URL=postgresql://user:password@host/db

# LLM API
PERPLEXITY_API_KEY=your_perplexity_key_here

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# Application
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000

# Performance
DB_POOL_SIZE=20
ENABLE_CACHE=true
```

Create `backend/requirements.txt`:
```
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
asyncpg==0.29.0
pgvector==0.2.3

# Embeddings
sentence-transformers==2.2.2
torch==2.1.0

# LLM
openai==1.3.5  # Perplexity uses OpenAI-compatible API

# Utilities
python-dotenv==1.0.0
tqdm==4.66.1
numpy==1.24.3

# Logging
structlog==23.2.0
```

**Commands**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Deliverable**: Python environment set up, dependencies installed

#### 1.4 Database Schema Creation

Create `backend/scripts/setup_database.py`:
```python
"""
Database Setup Script
Creates tables and indices for medical coding system
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

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
        raise ValueError("NEON_DATABASE_URL not set")

    print("🔌 Connecting to database...")
    conn = await asyncpg.connect(db_url)

    try:
        print("🔧 Creating pgvector extension...")
        await conn.execute(CREATE_EXTENSIONS)

        print("📋 Creating cpt_codes table...")
        await conn.execute(CREATE_CPT_TABLE)

        print("📋 Creating icd10_codes table...")
        await conn.execute(CREATE_ICD10_TABLE)

        print("⚡ Creating indices (this may take a moment)...")
        await conn.execute(CREATE_INDICES)

        print("✅ Database setup complete!")

        # Verify
        cpt_count = await conn.fetchval("SELECT COUNT(*) FROM cpt_codes")
        icd10_count = await conn.fetchval("SELECT COUNT(*) FROM icd10_codes")

        print(f"\n📊 Current data:")
        print(f"   CPT codes: {cpt_count}")
        print(f"   ICD-10 codes: {icd10_count}")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(setup_database())
```

**Run**:
```bash
python scripts/setup_database.py
```

**Deliverable**: Database tables and indices created

---

### Day 2: Data Loading with Embeddings

#### 2.1 Create Embedding Service

Create `backend/app/services/embeddings.py`:
```python
"""
Embedding Generation Service
Uses sentence-transformers for local embedding generation
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Manages embedding generation using sentence-transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            Normalized embedding vector
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            Array of normalized embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings

# Global instance (singleton pattern)
_embedding_service = None

def get_embedding_service(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingService:
    """Get or create embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name)
    return _embedding_service
```

**Deliverable**: Embedding service created

#### 2.2 Load CPT Codes

Create `backend/scripts/load_cpt_codes.py`:
```python
"""
Load CPT codes from CSV and generate embeddings
"""

import asyncio
import asyncpg
import csv
import os
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embeddings import get_embedding_service

load_dotenv()

CPT_FILE = "data/all-2025-cpt-codes.csv"
BATCH_SIZE = 100

async def load_cpt_codes():
    """Load CPT codes with embeddings"""

    # Load CSV
    print(f"📖 Reading CPT codes from {CPT_FILE}...")
    cpt_data = []

    with open(CPT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpt_data.append({
                'category': row['Procedure Code Category'],
                'code': row['CPT Codes'].strip(),
                'description': row['Procedure Code Descriptions'].strip(),
                'status': row['Code Status']
            })

    print(f"✅ Loaded {len(cpt_data)} CPT codes")

    # Generate embeddings
    print("\n🧮 Generating embeddings...")
    embedding_service = get_embedding_service()

    descriptions = [item['description'] for item in cpt_data]
    embeddings = embedding_service.generate_embeddings_batch(descriptions, batch_size=BATCH_SIZE)

    # Add embeddings to data
    for i, item in enumerate(cpt_data):
        item['embedding'] = embeddings[i].tolist()

    # Insert into database
    print("\n💾 Inserting into database...")
    db_url = os.getenv('NEON_DATABASE_URL')
    conn = await asyncpg.connect(db_url)

    try:
        # Clear existing data
        await conn.execute("TRUNCATE TABLE cpt_codes CASCADE")

        # Batch insert
        insert_query = """
            INSERT INTO cpt_codes (cpt_code, description, category, code_status, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """

        for item in tqdm(cpt_data, desc="Inserting CPT codes"):
            await conn.execute(
                insert_query,
                item['code'],
                item['description'],
                item['category'],
                item['status'],
                item['embedding']
            )

        # Verify
        count = await conn.fetchval("SELECT COUNT(*) FROM cpt_codes")
        print(f"\n✅ Successfully loaded {count} CPT codes")

        # Sample query
        sample = await conn.fetchrow("""
            SELECT cpt_code, description, category
            FROM cpt_codes
            LIMIT 1
        """)
        print(f"\n📋 Sample record:")
        print(f"   Code: {sample['cpt_code']}")
        print(f"   Category: {sample['category']}")
        print(f"   Description: {sample['description'][:80]}...")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(load_cpt_codes())
```

**Run**:
```bash
cd backend
python scripts/load_cpt_codes.py
```

**Expected Output**:
```
📖 Reading CPT codes from data/all-2025-cpt-codes.csv...
✅ Loaded 1164 CPT codes

🧮 Generating embeddings...
100%|████████████| 1164/1164 [00:15<00:00, 74.21it/s]

💾 Inserting into database...
100%|████████████| 1164/1164 [00:08<00:00, 142.15it/s]

✅ Successfully loaded 1164 CPT codes
```

**Deliverable**: CPT codes loaded with embeddings (~1,164 codes)

#### 2.3 Load ICD-10 Codes

Create `backend/scripts/load_icd10_codes.py`:
```python
"""
Load ICD-10 codes from text file and generate embeddings
Extracts hierarchy information from code patterns
"""

import asyncio
import asyncpg
import os
import sys
import re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embeddings import get_embedding_service

load_dotenv()

ICD10_FILE = "data/icd10cm-codes-2025.txt"
BATCH_SIZE = 100

def parse_icd10_line(line: str) -> dict:
    """
    Parse ICD-10 line: "CODE    DESCRIPTION"

    Returns:
        dict with code, description, chapter, block
    """
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None

    code = parts[0].strip()
    description = parts[1].strip()

    # Extract hierarchy from code pattern
    # ICD-10 structure: A00.123
    #   Letter = Chapter
    #   First 3 chars = Category
    #   Full code = Specific diagnosis

    chapter = None
    block = None

    if len(code) >= 1:
        # Chapter: A00-B99, E00-E89, etc.
        letter = code[0]
        if letter.isalpha():
            # Simplified chapter mapping (first letter)
            chapter = f"{letter}00-{letter}99"

    if len(code) >= 3:
        # Block: A00-A09, E08-E13, etc.
        category = code[:3]
        # Round down to nearest 10 for block
        if code[1:3].isdigit():
            start = int(code[1:3]) // 10 * 10
            end = start + 9
            block = f"{code[0]}{start:02d}-{code[0]}{end:02d}"

    return {
        'code': code,
        'description': description,
        'chapter': chapter,
        'block': block
    }

async def load_icd10_codes():
    """Load ICD-10 codes with embeddings"""

    # Parse file
    print(f"📖 Reading ICD-10 codes from {ICD10_FILE}...")
    icd10_data = []

    with open(ICD10_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parsed = parse_icd10_line(line)
                if parsed:
                    icd10_data.append(parsed)

    print(f"✅ Loaded {len(icd10_data)} ICD-10 codes")

    # Generate embeddings
    print("\n🧮 Generating embeddings (this will take a few minutes)...")
    embedding_service = get_embedding_service()

    descriptions = [item['description'] for item in icd10_data]

    # Process in batches to avoid memory issues
    all_embeddings = []
    for i in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="Embedding batches"):
        batch = descriptions[i:i + BATCH_SIZE]
        batch_embeddings = embedding_service.generate_embeddings_batch(batch, batch_size=32)
        all_embeddings.extend(batch_embeddings.tolist())

    # Add embeddings to data
    for i, item in enumerate(icd10_data):
        item['embedding'] = all_embeddings[i]

    # Insert into database
    print("\n💾 Inserting into database...")
    db_url = os.getenv('NEON_DATABASE_URL')
    conn = await asyncpg.connect(db_url)

    try:
        # Clear existing data
        await conn.execute("TRUNCATE TABLE icd10_codes CASCADE")

        # Batch insert using executemany for better performance
        insert_query = """
            INSERT INTO icd10_codes (icd10_code, description, chapter, block, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """

        # Prepare data for executemany
        records = [
            (
                item['code'],
                item['description'],
                item['chapter'],
                item['block'],
                item['embedding']
            )
            for item in icd10_data
        ]

        # Insert in batches
        for i in tqdm(range(0, len(records), 1000), desc="Inserting ICD-10 codes"):
            batch = records[i:i + 1000]
            await conn.executemany(insert_query, batch)

        # Verify
        count = await conn.fetchval("SELECT COUNT(*) FROM icd10_codes")
        print(f"\n✅ Successfully loaded {count} ICD-10 codes")

        # Sample query
        sample = await conn.fetchrow("""
            SELECT icd10_code, description, chapter, block
            FROM icd10_codes
            WHERE chapter IS NOT NULL
            LIMIT 1
        """)
        print(f"\n📋 Sample record:")
        print(f"   Code: {sample['icd10_code']}")
        print(f"   Chapter: {sample['chapter']}")
        print(f"   Block: {sample['block']}")
        print(f"   Description: {sample['description'][:80]}...")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(load_icd10_codes())
```

**Run**:
```bash
python scripts/load_icd10_codes.py
```

**Expected Output**:
```
📖 Reading ICD-10 codes from data/icd10cm-codes-2025.txt...
✅ Loaded 74260 ICD-10 codes

🧮 Generating embeddings (this will take a few minutes)...
100%|████████████| 743/743 [06:22<00:00, 1.94it/s]

💾 Inserting into database...
100%|████████████| 75/75 [00:12<00:00, 5.98it/s]

✅ Successfully loaded 74260 ICD-10 codes
```

**Deliverable**: ICD-10 codes loaded with embeddings (~74,260 codes)

**🎉 Phase 1 Complete!** You now have a fully populated database with vector embeddings.

---

## Phase 2: Backend Core Services (Days 3-4)

### Day 3: Search Services

#### 3.1 Configuration Management

Create `backend/app/config.py`:
```python
"""
Application Configuration
Uses Pydantic Settings for type-safe environment variables
"""

from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings loaded from environment"""

    # Database
    NEON_DATABASE_URL: str
    DB_POOL_SIZE: int = 20

    # LLM API
    PERPLEXITY_API_KEY: str
    PERPLEXITY_MODEL: str = "llama-3.1-sonar-small-128k-online"

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Application
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Performance
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
```

**Deliverable**: Configuration management

#### 3.2 Database Connection Pool

Create `backend/app/database.py`:
```python
"""
Database Connection Management
Async connection pool for PostgreSQL
"""

import asyncpg
from typing import Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)

class Database:
    """Database connection pool manager"""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create connection pool"""
        logger.info("Creating database connection pool...")
        self.pool = await asyncpg.create_pool(
            settings.NEON_DATABASE_URL,
            min_size=5,
            max_size=settings.DB_POOL_SIZE
        )
        logger.info(f"Connection pool created (max size: {settings.DB_POOL_SIZE})")

    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def fetch(self, query: str, *args):
        """Fetch multiple rows"""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        """Fetch single row"""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Fetch single value"""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args):
        """Execute query without return"""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

# Global database instance
db = Database()
```

**Deliverable**: Database connection pool

#### 3.3 Vector Search Service

Create `backend/app/services/vector_search.py`:
```python
"""
Vector Similarity Search Service
Uses pgvector for semantic search
"""

import numpy as np
from typing import List, Dict
import logging
from ..database import db

logger = logging.getLogger(__name__)

async def search_cpt_codes_vector(
    query_embedding: np.ndarray,
    limit: int = 20,
    category: str = None
) -> List[Dict]:
    """
    Search CPT codes using vector similarity

    Args:
        query_embedding: Query embedding vector
        limit: Number of results to return
        category: Optional category filter

    Returns:
        List of CPT code results with similarity scores
    """
    embedding_list = query_embedding.tolist()

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
        results = await db.fetch(query, embedding_list, category, limit)
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
        results = await db.fetch(query, embedding_list, limit)

    return [dict(row) for row in results]


async def search_icd10_codes_vector(
    query_embedding: np.ndarray,
    limit: int = 20,
    chapter: str = None
) -> List[Dict]:
    """
    Search ICD-10 codes using vector similarity

    Args:
        query_embedding: Query embedding vector
        limit: Number of results to return
        chapter: Optional chapter filter (e.g., "E00-E89")

    Returns:
        List of ICD-10 code results with similarity scores
    """
    embedding_list = query_embedding.tolist()

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
        results = await db.fetch(query, embedding_list, chapter, limit)
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
        results = await db.fetch(query, embedding_list, limit)

    return [dict(row) for row in results]
```

**Deliverable**: Vector search service

#### 3.4 Keyword Search Service

Create `backend/app/services/keyword_search.py`:
```python
"""
Keyword Search Service
Uses PostgreSQL full-text search (tsvector)
"""

from typing import List, Dict
import logging
from ..database import db

logger = logging.getLogger(__name__)

async def search_cpt_codes_keyword(
    query: str,
    limit: int = 20,
    category: str = None
) -> List[Dict]:
    """
    Search CPT codes using full-text search

    Args:
        query: Search query string
        limit: Number of results to return
        category: Optional category filter

    Returns:
        List of CPT code results with ranking scores
    """
    # Create tsquery from search text
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
    chapter: str = None
) -> List[Dict]:
    """
    Search ICD-10 codes using full-text search

    Args:
        query: Search query string
        limit: Number of results to return
        chapter: Optional chapter filter

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
```

**Deliverable**: Keyword search service

**Day 3 Complete!** You now have both vector and keyword search working.

---

### Day 4: Hybrid Search & Ranking

#### 4.1 Reciprocal Rank Fusion

Create `backend/app/services/ranking.py`:
```python
"""
Ranking and Fusion Services
Implements Reciprocal Rank Fusion for combining search results
"""

from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    results_lists: List[List[Dict]],
    k: int = 60,
    key_field: str = 'cpt_code'
) -> List[Dict]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion

    RRF formula: score(d) = Σ(1 / (k + rank(d)))
    where k is a constant (typically 60) and rank starts at 1

    Args:
        results_lists: List of result lists (each from different search method)
        k: RRF constant (default 60)
        key_field: Field to use as unique identifier

    Returns:
        Combined and re-ranked results
    """
    # Collect all unique items with their RRF scores
    rrf_scores: Dict[str, Dict] = {}

    for results in results_lists:
        for rank, item in enumerate(results, start=1):
            key = item[key_field]

            if key not in rrf_scores:
                rrf_scores[key] = {
                    **item,
                    'rrf_score': 0,
                    'component_scores': []
                }

            # Add RRF score
            score = 1.0 / (k + rank)
            rrf_scores[key]['rrf_score'] += score
            rrf_scores[key]['component_scores'].append(score)

    # Sort by RRF score
    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )

    return ranked_results


def normalize_scores(results: List[Dict], score_field: str = 'rrf_score') -> List[Dict]:
    """
    Normalize scores to 0-1 range

    Args:
        results: List of results with scores
        score_field: Field containing the score

    Returns:
        Results with normalized confidence_score field
    """
    if not results:
        return results

    max_score = max(item[score_field] for item in results)
    min_score = min(item[score_field] for item in results)

    range_score = max_score - min_score if max_score != min_score else 1.0

    for item in results:
        item['confidence_score'] = (item[score_field] - min_score) / range_score

    return results
```

**Deliverable**: Ranking utilities

#### 4.2 Hybrid Search Service

Create `backend/app/services/hybrid_search.py`:
```python
"""
Hybrid Search Service
Combines vector and keyword search using Reciprocal Rank Fusion
"""

import asyncio
import numpy as np
from typing import List, Dict, Tuple
import logging

from .vector_search import search_cpt_codes_vector, search_icd10_codes_vector
from .keyword_search import search_cpt_codes_keyword, search_icd10_codes_keyword
from .ranking import reciprocal_rank_fusion, normalize_scores
from .embeddings import get_embedding_service

logger = logging.getLogger(__name__)

async def hybrid_search_cpt(
    query: str,
    query_embedding: np.ndarray,
    limit: int = 10,
    category: str = None
) -> List[Dict]:
    """
    Hybrid search for CPT codes
    Combines vector and keyword search

    Args:
        query: Text query
        query_embedding: Embedding vector
        limit: Number of final results
        category: Optional category filter

    Returns:
        Combined ranked results
    """
    # Run both searches in parallel
    vector_results, keyword_results = await asyncio.gather(
        search_cpt_codes_vector(query_embedding, limit=20, category=category),
        search_cpt_codes_keyword(query, limit=20, category=category)
    )

    # Combine using RRF
    combined = reciprocal_rank_fusion(
        [vector_results, keyword_results],
        key_field='cpt_code'
    )

    # Normalize scores
    combined = normalize_scores(combined, score_field='rrf_score')

    # Return top results
    return combined[:limit]


async def hybrid_search_icd10(
    query: str,
    query_embedding: np.ndarray,
    limit: int = 10,
    chapter: str = None
) -> List[Dict]:
    """
    Hybrid search for ICD-10 codes
    Combines vector and keyword search

    Args:
        query: Text query
        query_embedding: Embedding vector
        limit: Number of final results
        chapter: Optional chapter filter

    Returns:
        Combined ranked results
    """
    # Run both searches in parallel
    vector_results, keyword_results = await asyncio.gather(
        search_icd10_codes_vector(query_embedding, limit=20, chapter=chapter),
        search_icd10_codes_keyword(query, limit=20, chapter=chapter)
    )

    # Combine using RRF
    combined = reciprocal_rank_fusion(
        [vector_results, keyword_results],
        key_field='icd10_code'
    )

    # Normalize scores
    combined = normalize_scores(combined, score_field='rrf_score')

    # Return top results
    return combined[:limit]


async def search_all(
    query: str,
    limit_per_type: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Search both CPT and ICD-10 codes

    Args:
        query: Clinical description
        limit_per_type: Results per code type

    Returns:
        Tuple of (cpt_results, icd10_results)
    """
    # Generate embedding once
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.generate_embedding(query)

    # Search both in parallel
    cpt_results, icd10_results = await asyncio.gather(
        hybrid_search_cpt(query, query_embedding, limit=limit_per_type),
        hybrid_search_icd10(query, query_embedding, limit=limit_per_type)
    )

    return cpt_results, icd10_results
```

**Deliverable**: Hybrid search combining vector + keyword

**Day 4 Complete!** Your core search engine is now operational.

---

## Phase 3: API & LLM Integration (Day 5)

### Day 5: FastAPI Endpoints & LLM Service

#### 5.1 Pydantic Models

Create `backend/app/models/request_models.py`:
```python
"""
Request Models
Pydantic schemas for API requests
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class CodingQuery(BaseModel):
    """Request model for code search"""

    clinical_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Clinical description or symptoms"
    )

    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum results per code type"
    )

    search_mode: Literal["quick", "standard", "expert"] = Field(
        default="standard",
        description="Search mode: quick (no LLM), standard (cached), expert (full LLM)"
    )

    filter_category: Optional[str] = Field(
        default=None,
        description="Filter CPT codes by category"
    )

    filter_chapter: Optional[str] = Field(
        default=None,
        description="Filter ICD-10 codes by chapter"
    )

    @validator('clinical_description')
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError("Clinical description cannot be empty")
        return v.strip()
```

Create `backend/app/models/response_models.py`:
```python
"""
Response Models
Pydantic schemas for API responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class CodeSuggestion(BaseModel):
    """Individual code suggestion"""

    code: str = Field(..., description="Medical code (CPT or ICD-10)")
    description: str = Field(..., description="Code description")
    code_type: Literal["CPT", "ICD-10"] = Field(..., description="Type of code")
    category: Optional[str] = Field(None, description="Category or chapter")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    reasoning: Optional[str] = Field(None, description="Explanation (expert mode only)")

class CodingResponse(BaseModel):
    """Response model for code suggestions"""

    query: str = Field(..., description="Original query")
    cpt_codes: List[CodeSuggestion] = Field(..., description="CPT procedure codes")
    icd10_codes: List[CodeSuggestion] = Field(..., description="ICD-10 diagnosis codes")
    search_mode: str = Field(..., description="Mode used for search")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    explanation: Optional[str] = Field(None, description="Overall explanation (expert mode)")

class StatsResponse(BaseModel):
    """Database statistics"""

    total_cpt_codes: int
    total_icd10_codes: int
    categories: List[str]
    chapters: List[str]

class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    database: str
    embedding_model: str
```

**Deliverable**: API models defined

#### 5.2 LLM Service

Create `backend/app/services/llm_service.py`:
```python
"""
LLM Service for Expert Mode
Uses Perplexity API for code reranking and explanations
"""

from openai import AsyncOpenAI
from typing import List, Dict, Optional
import json
import logging
from ..config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Perplexity LLM client for code reranking"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai"
        )
        self.model = settings.PERPLEXITY_MODEL

    async def rerank_codes(
        self,
        query: str,
        cpt_codes: List[Dict],
        icd10_codes: List[Dict]
    ) -> Dict:
        """
        Rerank codes using LLM

        Args:
            query: Clinical description
            cpt_codes: Retrieved CPT codes
            icd10_codes: Retrieved ICD-10 codes

        Returns:
            Dict with reranked codes and explanation
        """
        # Format codes for prompt
        cpt_text = "\n".join([
            f"{i+1}. {code['cpt_code']} (score: {code['confidence_score']:.2f}) - {code['description'][:100]}"
            for i, code in enumerate(cpt_codes)
        ])

        icd10_text = "\n".join([
            f"{i+1}. {code['icd10_code']} (score: {code['confidence_score']:.2f}) - {code['description'][:100]}"
            for i, code in enumerate(icd10_codes)
        ])

        prompt = f"""You are a medical coding expert. Given a clinical description and candidate codes, select the most relevant codes and explain why.

Clinical Description: "{query}"

Candidate CPT Codes (Procedures):
{cpt_text}

Candidate ICD-10 Codes (Diagnoses):
{icd10_text}

Task: Select the top 5 most relevant codes from each category and provide:
1. Confidence score (0-1) for each
2. Brief reasoning for top 3 codes
3. Overall explanation

Return as JSON:
{{
  "cpt_codes": [
    {{"code": "99213", "confidence": 0.95, "reasoning": "Primary evaluation visit"}},
    ...
  ],
  "icd10_codes": [
    {{"code": "E11.9", "confidence": 0.98, "reasoning": "Type 2 diabetes diagnosis"}},
    ...
  ],
  "explanation": "Overall explanation of code selections"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical coding expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            return result

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            # Fallback: return original results
            return {
                "cpt_codes": [
                    {
                        "code": code['cpt_code'],
                        "confidence": code['confidence_score'],
                        "reasoning": None
                    }
                    for code in cpt_codes[:5]
                ],
                "icd10_codes": [
                    {
                        "code": code['icd10_code'],
                        "confidence": code['confidence_score'],
                        "reasoning": None
                    }
                    for code in icd10_codes[:5]
                ],
                "explanation": "LLM unavailable - results from hybrid search only"
            }

# Global instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get or create LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
```

**Deliverable**: LLM service for expert mode

#### 5.3 Main FastAPI Application

Create `backend/app/main.py`:
```python
"""
FastAPI Application
Main entry point for Medical Coding RAG API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from contextlib import asynccontextmanager

from .config import settings
from .database import db
from .models.request_models import CodingQuery
from .models.response_models import (
    CodingResponse, CodeSuggestion, StatsResponse, HealthResponse
)
from .services.hybrid_search import search_all
from .services.llm_service import get_llm_service
from .services.embeddings import get_embedding_service

# Setup logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Medical Coding RAG API...")
    await db.connect()

    # Load embedding model
    get_embedding_service(settings.EMBEDDING_MODEL_NAME)

    logger.info("API ready!")
    yield

    # Shutdown
    await db.disconnect()
    logger.info("API shutdown complete")

# Create app
app = FastAPI(
    title="Medical Coding RAG API",
    description="Retrieval Augmented Generation system for medical code suggestions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test database
        count = await db.fetchval("SELECT COUNT(*) FROM cpt_codes")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        database=db_status,
        embedding_model=settings.EMBEDDING_MODEL_NAME
    )

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    cpt_count = await db.fetchval("SELECT COUNT(*) FROM cpt_codes")
    icd10_count = await db.fetchval("SELECT COUNT(*) FROM icd10_codes")

    categories = await db.fetch("SELECT DISTINCT category FROM cpt_codes WHERE category IS NOT NULL ORDER BY category")
    chapters = await db.fetch("SELECT DISTINCT chapter FROM icd10_codes WHERE chapter IS NOT NULL ORDER BY chapter")

    return StatsResponse(
        total_cpt_codes=cpt_count,
        total_icd10_codes=icd10_count,
        categories=[row['category'] for row in categories],
        chapters=[row['chapter'] for row in chapters]
    )

@app.post("/api/code-suggestions", response_model=CodingResponse)
async def get_code_suggestions(query: CodingQuery):
    """Main endpoint for medical code suggestions"""
    start_time = time.time()

    try:
        # Hybrid search
        cpt_results, icd10_results = await search_all(
            query.clinical_description,
            limit_per_type=query.max_results
        )

        # Mode-based processing
        if query.search_mode == "expert":
            # LLM reranking
            llm_service = get_llm_service()
            llm_results = await llm_service.rerank_codes(
                query.clinical_description,
                cpt_results,
                icd10_results
            )

            # Map back to full code details
            cpt_map = {c['cpt_code']: c for c in cpt_results}
            icd10_map = {c['icd10_code']: c for c in icd10_results}

            cpt_codes = [
                CodeSuggestion(
                    code=item['code'],
                    description=cpt_map[item['code']]['description'],
                    code_type="CPT",
                    category=cpt_map[item['code']].get('category'),
                    confidence_score=item['confidence'],
                    reasoning=item.get('reasoning')
                )
                for item in llm_results['cpt_codes']
                if item['code'] in cpt_map
            ]

            icd10_codes = [
                CodeSuggestion(
                    code=item['code'],
                    description=icd10_map[item['code']]['description'],
                    code_type="ICD-10",
                    category=icd10_map[item['code']].get('chapter'),
                    confidence_score=item['confidence'],
                    reasoning=item.get('reasoning')
                )
                for item in llm_results['icd10_codes']
                if item['code'] in icd10_map
            ]

            explanation = llm_results.get('explanation')

        else:
            # Quick/Standard mode (no LLM)
            cpt_codes = [
                CodeSuggestion(
                    code=item['cpt_code'],
                    description=item['description'],
                    code_type="CPT",
                    category=item.get('category'),
                    confidence_score=item['confidence_score'],
                    reasoning=None
                )
                for item in cpt_results
            ]

            icd10_codes = [
                CodeSuggestion(
                    code=item['icd10_code'],
                    description=item['description'],
                    code_type="ICD-10",
                    category=item.get('chapter'),
                    confidence_score=item['confidence_score'],
                    reasoning=None
                )
                for item in icd10_results
            ]

            explanation = None

        processing_time = (time.time() - start_time) * 1000

        return CodingResponse(
            query=query.clinical_description,
            cpt_codes=cpt_codes,
            icd10_codes=icd10_codes,
            search_mode=query.search_mode,
            processing_time_ms=processing_time,
            explanation=explanation
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Deliverable**: Complete FastAPI application

#### 5.4 Test the API

Create `backend/.env`:
```bash
NEON_DATABASE_URL=postgresql://your_connection_string
PERPLEXITY_API_KEY=your_key_here
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000"]
```

**Run**:
```bash
cd backend
uvicorn app.main:app --reload
```

**Test**:
```bash
# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/api/stats

# Search (Quick mode)
curl -X POST http://localhost:8000/api/code-suggestions \
  -H "Content-Type: application/json" \
  -d '{"clinical_description": "patient with type 2 diabetes", "search_mode": "quick"}'
```

**Visit**: http://localhost:8000/docs (Swagger UI)

**Day 5 Complete!** Your backend API is fully operational.

---

## Phase 4: Frontend (Days 6-7)

### Day 6-7: Next.js Frontend

See [TECH_STACK.md](./TECH_STACK.md) for detailed frontend implementation.

**Quick Overview**:
1. Initialize Next.js project
2. Create API client with Axios
3. Build search interface components
4. Implement three-mode selector
5. Display results with confidence scores
6. Add error handling and loading states

**Deliverable**: Functional frontend connected to backend

---

## Phase 5: Documentation & Polish (Day 8)

### Day 8: Documentation

Create the following documentation files:

1. **README.md** - Project overview, setup instructions
2. **ARCHITECTURE.md** - Deep dive into system design
3. **API_DOCUMENTATION.md** - API reference
4. **.env.example** - Environment variable template
5. **docker-compose.yml** - Docker setup

Add comprehensive code comments throughout.

**Deliverable**: Professional documentation

---

## Phase 6: Testing & Refinement (Days 9-10)

### Day 9-10: Final Polish

1. **Manual Testing**
   - Test various medical queries
   - Verify all three modes work
   - Test error cases
   - Check performance

2. **Code Quality**
   - Add type hints everywhere
   - Write docstrings
   - Clean up unused code
   - Format with Black

3. **Docker Setup**
   - Create Dockerfile
   - Test Docker build
   - Write docker-compose.yml

4. **Git**
   - Clean commit history
   - Write good commit messages
   - Update .gitignore

**Deliverable**: Production-ready, portfolio-quality project

---

## ✅ Completion Checklist

- [ ] Database schema created
- [ ] CPT codes loaded with embeddings
- [ ] ICD-10 codes loaded with embeddings
- [ ] Vector search working
- [ ] Keyword search working
- [ ] Hybrid search implemented
- [ ] RRF ranking functional
- [ ] FastAPI endpoints working
- [ ] LLM service integrated
- [ ] Three modes operational
- [ ] Frontend deployed
- [ ] API connected to frontend
- [ ] Error handling comprehensive
- [ ] Documentation complete
- [ ] Code commented
- [ ] Docker working
- [ ] Git repository clean

---

## 🎯 Success Metrics

After completion, your project should achieve:

**Performance:**
- Quick mode: <500ms response time ✓
- Standard mode: <1s response time ✓
- Expert mode: <3s response time ✓

**Accuracy** (manual testing):
- Top 5 results contain relevant codes >80% of the time ✓
- No crashes or errors under normal use ✓

**Code Quality:**
- All functions documented ✓
- Type hints throughout ✓
- No hardcoded secrets ✓
- Clean git history ✓

**Professional Polish:**
- README is clear and comprehensive ✓
- Setup instructions work first time ✓
- Swagger docs are accurate ✓
- Docker builds successfully ✓

---

## 📞 Getting Help

If you encounter issues during implementation:

1. **Database Connection Issues**: Check Neon dashboard, verify connection string
2. **Embedding Errors**: Ensure sentence-transformers installed correctly
3. **pgvector Issues**: Verify extension enabled: `CREATE EXTENSION vector;`
4. **API Errors**: Check FastAPI logs, verify all environment variables set
5. **Performance Issues**: Check indices created, verify connection pooling

---

This implementation plan provides a clear, day-by-day roadmap to build your portfolio-ready Medical Coding RAG system. Follow each phase sequentially, testing as you go, and you'll have a professional project in 8-10 days.

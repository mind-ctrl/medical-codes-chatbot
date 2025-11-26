# Technology Stack - Detailed Reference

## Overview

This document provides comprehensive information about every technology used in the Medical Coding RAG system, including setup instructions, configuration details, and justifications.

---

## Backend Stack

### 1. Python 3.10+

**Why Python?**
- Best ecosystem for AI/ML development
- Excellent async support (critical for our use case)
- Strong typing with type hints
- Wide industry adoption

**Installation:**
```bash
# Check version
python --version  # Should be 3.10+

# Windows
# Download from python.org

# macOS
brew install python@3.10

# Linux
sudo apt install python3.10
```

**Version Requirements**: 3.10+ required for:
- `match` statements (nice to have)
- Better type hints
- Performance improvements
- Compatible with all our dependencies

---

### 2. FastAPI

**Version**: 0.104.1+

**Why FastAPI?**
- Automatic OpenAPI (Swagger) documentation
- Built-in Pydantic validation
- Excellent async performance
- Type-safe request/response models
- Easy to test

**Key Features We Use:**
- `@app.get()`, `@app.post()` decorators
- Automatic JSON serialization
- Request validation via Pydantic
- Dependency injection
- Lifespan events (startup/shutdown)
- CORS middleware

**Example:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Medical Coding API")

class Query(BaseModel):
    text: str

@app.post("/search")
async def search(query: Query):
    return {"results": []}
```

**Auto Documentation**: Visit `/docs` for Swagger UI

**Alternatives Considered:**
- Flask: Simpler but lacks async, no auto-docs
- Django: Too heavy for API-only service
- FastAPI: ✓ Best for async APIs with validation

---

### 3. Pydantic

**Version**: 2.5.0+

**Why Pydantic?**
- Data validation with Python type hints
- Automatic error messages
- JSON serialization/deserialization
- Environment variable management
- IDE autocomplete support

**Key Classes:**
```python
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# Request/Response models
class CodingQuery(BaseModel):
    clinical_description: str = Field(..., min_length=10)
    max_results: int = Field(default=5, ge=1, le=20)

    @validator('clinical_description')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Cannot be empty")
        return v.strip()

# Settings from environment
class Settings(BaseSettings):
    NEON_DATABASE_URL: str
    PERPLEXITY_API_KEY: str

    class Config:
        env_file = ".env"
```

**Benefits:**
- Runtime validation (catches bad inputs)
- Clear error messages for API users
- Type safety throughout application
- Settings management with .env files

---

### 4. PostgreSQL with pgvector

**Version**: PostgreSQL 15+, pgvector 0.5.0+

**Why PostgreSQL + pgvector?**
- Single database for structured + vector data
- Leverage SQL knowledge
- ACID transactions
- Proven reliability
- pgvector: Native vector similarity search

**pgvector Features:**
```sql
-- Create vector column
CREATE TABLE codes (
    id SERIAL PRIMARY KEY,
    embedding vector(384)  -- 384-dimensional vectors
);

-- Similarity search (cosine distance)
SELECT * FROM codes
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;

-- Index for fast search
CREATE INDEX ON codes
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Distance Operators:**
- `<->` : L2 distance (Euclidean)
- `<=>` : Cosine distance (we use this)
- `<#>` : Inner product

**Index Types:**
- **IVFFlat**: Good for 10K-1M vectors, faster build
- **HNSW**: Better for 1M+ vectors, slower build, faster search

**We use**: IVFFlat with 100 lists (good for our 74K vectors)

**Full-Text Search:**
```sql
-- Generated tsvector column
CREATE TABLE codes (
    description TEXT,
    description_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', description)
    ) STORED
);

-- GIN index for fast FTS
CREATE INDEX ON codes USING GIN (description_tsv);

-- Search
SELECT * FROM codes
WHERE description_tsv @@ to_tsquery('diabetes & type');
```

---

### 5. Neon.tech (Serverless PostgreSQL)

**Why Neon?**
- Serverless (auto-scaling)
- Built-in pgvector support
- Generous free tier
- Easy setup
- Branch-able databases (like Git)

**Free Tier:**
- 512 MB storage (enough for 100K+ vectors)
- 191 hours/month compute
- 1 project, unlimited databases
- 10 branches

**Connection:**
```bash
# Connection string format
postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require

# In Python (asyncpg)
import asyncpg
conn = await asyncpg.connect("postgresql://...")
```

**Setup:**
1. Sign up at neon.tech
2. Create project
3. Copy connection string
4. Enable pgvector: `CREATE EXTENSION vector;`

**Alternatives Considered:**
- Pinecone: Pure vector DB, can't do SQL joins
- Weaviate: Complex setup, overkill
- Supabase: Similar, but Neon has better pgvector support
- **Neon**: ✓ Best for our use case

---

### 6. asyncpg

**Version**: 0.29.0+

**Why asyncpg?**
- Fastest PostgreSQL driver for Python
- Native async/await support
- Connection pooling
- Prepared statements
- Type conversions

**vs psycopg2:**
- psycopg2: Synchronous, blocking
- asyncpg: Asynchronous, 3-5x faster

**Usage:**
```python
import asyncpg

# Create pool
pool = await asyncpg.create_pool(
    "postgresql://...",
    min_size=5,
    max_size=20
)

# Execute query
async with pool.acquire() as conn:
    rows = await conn.fetch("SELECT * FROM codes LIMIT 10")
    for row in rows:
        print(row['code'])

# With vector
embedding = [0.1, 0.2, ...]  # List of floats
rows = await conn.fetch(
    "SELECT * FROM codes ORDER BY embedding <=> $1 LIMIT 10",
    embedding  # asyncpg handles list → vector conversion
)
```

**Connection Pool**: Critical for performance
- Min size: 5 (always ready)
- Max size: 20 (prevents DB overload)
- Reuses connections (no overhead)

---

### 7. Sentence Transformers

**Version**: 2.2.2+

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

**Why Sentence Transformers?**
- Easy-to-use embeddings
- Runs locally (no API costs)
- Fast on CPU
- Pre-trained models
- Normalize by default

**Our Model Specs:**
- Dimensions: 384
- Speed: ~15ms per embedding (CPU)
- Size: ~80 MB
- Training: General domain (web text)
- Performance: 90%+ on semantic similarity tasks

**Usage:**
```python
from sentence_transformers import SentenceTransformer

# Load once (singleton)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Single embedding
embedding = model.encode("type 2 diabetes", normalize_embeddings=True)
# Returns: numpy array, shape (384,)

# Batch embeddings (much faster)
texts = ["text 1", "text 2", "text 3"]
embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
)
# Returns: numpy array, shape (3, 384)
```

**Why Normalize?**
- Cosine similarity works on normalized vectors
- Prevents magnitude bias
- All vectors have length 1.0

**Memory Usage:**
- Model: ~200 MB RAM
- Batch of 100: ~15 MB
- Total: <500 MB for our use case

**Alternatives:**
- OpenAI Embeddings: Paid, API call per query
- BioBERT: 3x slower, 5% better (not worth it)
- all-MiniLM-L6-v2: ✓ Best speed/accuracy trade-off

---

### 8. Perplexity API (LLM)

**Model**: `llama-3.1-sonar-small-128k-online`

**Why Perplexity?**
- OpenAI-compatible API
- Online mode (can access current info)
- Cost-effective
- Fast response times
- Good for reasoning tasks

**Pricing** (as of 2024):
- Sonar Small: ~$0.20 per 1M tokens
- Sonar Large: ~$1.00 per 1M tokens
- GPT-4: ~$10-30 per 1M tokens

**Usage with OpenAI SDK:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="pplx-xxx",
    base_url="https://api.perplexity.ai"
)

response = await client.chat.completions.create(
    model="llama-3.1-sonar-small-128k-online",
    messages=[
        {"role": "system", "content": "You are a medical coding expert."},
        {"role": "user", "content": "Rank these codes..."}
    ],
    temperature=0.2,  # Low for consistency
    max_tokens=2000
)

content = response.choices[0].message.content
```

**Online Mode**:
- Can search web for current medical guidelines
- Better for complex medical questions
- Slightly slower but more accurate

**Our Use Case**:
- Only in Expert mode
- Rerank top codes
- Provide explanations
- ~500 tokens per request
- Cost: ~$0.0001 per query

**Fallback Strategy**:
```python
try:
    llm_result = await perplexity_api_call()
except Exception:
    # Use similarity scores as confidence
    fallback_result = sort_by_similarity(codes)
```

---

### 9. Python-dotenv

**Why dotenv?**
- Manage environment variables
- Keep secrets out of code
- Different configs for dev/prod
- Standard practice

**Usage:**
```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file

db_url = os.getenv('NEON_DATABASE_URL')
api_key = os.getenv('PERPLEXITY_API_KEY')
```

**.env file:**
```bash
NEON_DATABASE_URL=postgresql://...
PERPLEXITY_API_KEY=pplx-xxx
LOG_LEVEL=INFO
```

**With Pydantic:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    NEON_DATABASE_URL: str
    PERPLEXITY_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()  # Auto-loads .env
```

---

## Frontend Stack

### 1. Next.js 14+ (App Router)

**Why Next.js?**
- React framework with built-in routing
- Server-side rendering (SSR)
- API routes (can add backend later)
- Great developer experience
- Production-ready out of box

**App Router (New):**
```
app/
├── page.tsx          # / route
├── layout.tsx        # Shared layout
└── api/
    └── route.ts      # API endpoints (optional)
```

**Installation:**
```bash
npx create-next-app@latest medical-coding-frontend
# Choose: TypeScript, Tailwind, App Router
```

---

### 2. TypeScript

**Why TypeScript?**
- Type safety (catch errors early)
- Better IDE support
- Self-documenting code
- Refactoring confidence

**Strict Mode:**
```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true
  }
}
```

**Example:**
```typescript
interface CodeSuggestion {
  code: string;
  description: string;
  confidence_score: number;
}

async function search(query: string): Promise<CodeSuggestion[]> {
  // TypeScript ensures we return correct type
  return [];
}
```

---

### 3. Tailwind CSS

**Why Tailwind?**
- Utility-first CSS
- Fast development
- Consistent design
- Responsive by default
- Small production bundle

**Example:**
```tsx
<div className="flex flex-col gap-4 p-6 bg-white rounded-lg shadow-md">
  <h2 className="text-2xl font-bold text-gray-800">Results</h2>
  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
    {/* Cards */}
  </div>
</div>
```

**Responsive:**
- `md:`: Medium screens and up
- `lg:`: Large screens and up
- Default: Mobile-first

---

### 4. Axios

**Why Axios over Fetch?**
- Interceptors (add auth headers)
- Automatic JSON parsing
- Timeout support
- Better error handling
- Request/response transformation

**Setup:**
```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor
api.interceptors.request.use(config => {
  console.log('Request:', config.method, config.url);
  return config;
});

// Response interceptor
api.interceptors.response.use(
  response => response.data,
  error => {
    console.error('API Error:', error);
    throw error;
  }
);
```

**Usage:**
```typescript
const response = await api.post('/api/code-suggestions', {
  clinical_description: query,
  search_mode: 'quick'
});
// response is automatically parsed JSON
```

---

## Development Tools

### 1. Git

**Best Practices:**
```bash
# Good commit messages
git commit -m "feat: add hybrid search combining vector and keyword"
git commit -m "fix: handle empty query edge case"
git commit -m "docs: update README with setup instructions"

# Feature branches
git checkout -b feature/llm-integration
git checkout -b fix/embedding-normalization

# .gitignore
.env
__pycache__/
node_modules/
.next/
venv/
```

---

### 2. Docker

**Why Docker?**
- Consistent environment
- Easy deployment
- Reproducible builds
- Portable

**Dockerfile (Backend):**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - NEON_DATABASE_URL=${NEON_DATABASE_URL}
      - PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

---

## Performance Optimization Tools

### 1. Connection Pooling

**asyncpg pool:**
```python
pool = await asyncpg.create_pool(
    dsn,
    min_size=5,    # Always maintain 5 connections
    max_size=20,   # Never exceed 20
    max_inactive_connection_lifetime=300  # Recycle after 5min
)
```

**Why?**
- Reuse connections (no overhead)
- Prevents connection exhaustion
- Faster queries (no connect time)

---

### 2. Batch Processing

**Embeddings:**
```python
# Bad: One at a time
for text in texts:
    embedding = model.encode(text)  # Slow!

# Good: Batch
embeddings = model.encode(texts, batch_size=32)  # 10x faster
```

**Database:**
```python
# Bad: Loop
for record in records:
    await conn.execute("INSERT ...", record)  # Slow!

# Good: executemany
await conn.executemany("INSERT ...", records)  # 50x faster
```

---

### 3. Async Operations

**Parallel Queries:**
```python
import asyncio

# Bad: Sequential
cpt = await search_cpt(query)
icd10 = await search_icd10(query)

# Good: Parallel
cpt, icd10 = await asyncio.gather(
    search_cpt(query),
    search_icd10(query)
)
# 2x faster
```

---

## Monitoring & Logging

### structlog

**Why structlog?**
- Structured logging (JSON)
- Easy to parse
- Context management
- Better than print()

**Setup:**
```python
import structlog

logger = structlog.get_logger()

logger.info("search_completed",
    query=query,
    mode=mode,
    results=len(results),
    duration_ms=duration
)

# Output:
# {"event": "search_completed", "query": "...", "mode": "quick", ...}
```

---

## Summary: Complete Stack

### Backend
- **Language**: Python 3.10+
- **Framework**: FastAPI 0.104+
- **Validation**: Pydantic 2.5+
- **Database**: PostgreSQL 15+ (Neon.tech)
- **Vector Search**: pgvector 0.5+
- **DB Driver**: asyncpg 0.29+
- **Embeddings**: sentence-transformers 2.2+
- **Model**: all-MiniLM-L6-v2 (384-dim)
- **LLM**: Perplexity API (llama-3.1-sonar)
- **Environment**: python-dotenv

### Frontend
- **Framework**: Next.js 14+
- **Language**: TypeScript (strict)
- **Styling**: Tailwind CSS
- **HTTP**: Axios
- **State**: React hooks

### DevOps
- **Version Control**: Git
- **Containerization**: Docker
- **Orchestration**: docker-compose

### Performance
- **Connection Pooling**: asyncpg pool
- **Async**: asyncio
- **Batch Processing**: Sentence Transformers batching
- **Caching**: In-memory (future: Redis)

---

## Environment Setup Checklist

- [ ] Python 3.10+ installed
- [ ] Node.js 18+ installed
- [ ] PostgreSQL client (psql) installed
- [ ] Git installed
- [ ] Docker installed (optional)
- [ ] Neon account created
- [ ] Perplexity API key obtained
- [ ] .env file created (backend)
- [ ] .env.local file created (frontend)
- [ ] Dependencies installed (pip, npm)
- [ ] Database schema created
- [ ] Data loaded with embeddings

---

## Cost Analysis

### Free Tier Usage

**Neon (Database):**
- Storage: ~50 MB (well under 512 MB limit)
- Compute: ~20 hours/month (well under 191 hours)
- Cost: **$0/month**

**Perplexity (LLM):**
- Assuming 1000 queries/month in Expert mode
- ~500 tokens per query
- 0.5M tokens total
- Cost: ~**$0.10/month**

**Sentence Transformers:**
- Runs locally
- Cost: **$0** (just compute)

**Hosting (Future):**
- Backend: Railway/Render free tier
- Frontend: Vercel free tier
- Cost: **$0/month**

**Total Monthly Cost**: <$1

---

## Scaling Considerations

**Current Capacity:**
- 74K vectors, <100ms search
- 100 requests/minute
- Single server

**To 1M vectors:**
- Switch to HNSW index
- Add read replicas
- Cost: ~$20-50/month

**To 10M vectors:**
- Dedicated vector DB (Pinecone/Qdrant)
- Multiple app servers
- Redis caching
- Cost: ~$100-200/month

---

This technology stack provides a solid foundation for a production-ready RAG system while keeping costs near zero and maintaining professional quality.

# Quick Start Guide

Get your Medical Coding RAG system up and running in under 30 minutes.

---

## Prerequisites

Before starting, ensure you have:

- [ ] **Python 3.10+** installed (`python --version`)
- [ ] **Git** installed
- [ ] **Code editor** (VS Code recommended)
- [ ] **Neon account** (sign up at https://neon.tech - free)
- [ ] **Perplexity API key** (sign up at https://www.perplexity.ai - free tier available)

---

## Step 1: Environment Setup (5 minutes)

### 1.1 Create Project Structure

```bash
# You're already in the project directory
cd "d:\Power BI Project\imbalanced_ BI\my_projects\GenAI\Ccursor-ICD-10"

# Create backend structure
mkdir -p backend/app/{models,services,utils}
mkdir -p backend/scripts
```

### 1.2 Create Virtual Environment

```bash
cd backend
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 1.3 Create requirements.txt

Create `backend/requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
asyncpg==0.29.0
pgvector==0.2.3
sentence-transformers==2.2.2
torch==2.1.0
openai==1.3.5
python-dotenv==1.0.0
tqdm==4.66.1
numpy==1.24.3
structlog==23.2.0
```

### 1.4 Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected time**: 3-5 minutes (downloading models)

---

## Step 2: Database Setup (10 minutes)

### 2.1 Create Neon Database

1. Go to https://neon.tech
2. Sign up with GitHub (free)
3. Create new project: "medical-coding-rag"
4. Copy your connection string (looks like `postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname`)

### 2.2 Create .env File

Create `backend/.env`:

```bash
# Database
NEON_DATABASE_URL=postgresql://your_connection_string_here

# LLM API (get free key from perplexity.ai)
PERPLEXITY_API_KEY=pplx-your_key_here

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# Application
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000"]

# Performance
DB_POOL_SIZE=20
ENABLE_CACHE=true
```

**⚠️ Important**: Replace the database URL and API key with your actual values!

### 2.3 Create Database Schema

Copy the `setup_database.py` script from [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) Day 1, Section 1.4.

Then run:

```bash
cd backend
python scripts/setup_database.py
```

**Expected output**:
```
🔌 Connecting to database...
🔧 Creating pgvector extension...
📋 Creating cpt_codes table...
📋 Creating icd10_codes table...
⚡ Creating indices...
✅ Database setup complete!
```

---

## Step 3: Load Data (15 minutes)

### 3.1 Load CPT Codes

Copy the `load_cpt_codes.py` script from [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) Day 2, Section 2.2.

Run:

```bash
python scripts/load_cpt_codes.py
```

**Expected time**: ~30 seconds

**Expected output**:
```
📖 Reading CPT codes from data/all-2025-cpt-codes.csv...
✅ Loaded 1164 CPT codes
🧮 Generating embeddings...
💾 Inserting into database...
✅ Successfully loaded 1164 CPT codes
```

### 3.2 Load ICD-10 Codes

Copy the `load_icd10_codes.py` script from [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) Day 2, Section 2.3.

Run:

```bash
python scripts/load_icd10_codes.py
```

**Expected time**: 6-8 minutes (generating 74K embeddings)

**Expected output**:
```
📖 Reading ICD-10 codes from data/icd10cm-codes-2025.txt...
✅ Loaded 74260 ICD-10 codes
🧮 Generating embeddings (this will take a few minutes)...
💾 Inserting into database...
✅ Successfully loaded 74260 ICD-10 codes
```

**☕ Grab a coffee while this runs!**

---

## Step 4: Verify Database (1 minute)

Check your data is loaded:

```bash
# Connect to your Neon database using psql or Neon dashboard

# Or use Python:
python -c "
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def check():
    conn = await asyncpg.connect(os.getenv('NEON_DATABASE_URL'))
    cpt = await conn.fetchval('SELECT COUNT(*) FROM cpt_codes')
    icd10 = await conn.fetchval('SELECT COUNT(*) FROM icd10_codes')
    print(f'✅ CPT codes: {cpt}')
    print(f'✅ ICD-10 codes: {icd10}')
    await conn.close()

asyncio.run(check())
"
```

**Expected output**:
```
✅ CPT codes: 1164
✅ ICD-10 codes: 74260
```

---

## Step 5: Test the API (2 minutes)

### 5.1 Create Minimal FastAPI App

Create a minimal version to test. Create `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Medical Coding RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    return {"message": "Medical Coding RAG API - See /docs for documentation"}
```

### 5.2 Run the Server

```bash
cd backend
uvicorn app.main:app --reload
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.3 Test It

Open your browser:
- http://localhost:8000 - Root endpoint
- http://localhost:8000/health - Health check
- http://localhost:8000/docs - Swagger UI (auto-generated docs)

**You should see the FastAPI Swagger interface!** 🎉

---

## Step 6: Implement Full Backend (Next Steps)

Now that your environment is working, follow [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) to:

1. **Day 3**: Implement search services (vector, keyword, hybrid)
2. **Day 4**: Add ranking and LLM integration
3. **Day 5**: Complete API endpoints
4. **Day 6-7**: Build frontend
5. **Day 8**: Documentation
6. **Day 9-10**: Polish and testing

---

## Troubleshooting

### Issue: "asyncpg.exceptions.UndefinedTableError"
**Solution**: Run `setup_database.py` first to create tables.

### Issue: "ModuleNotFoundError: No module named 'sentence_transformers'"
**Solution**: Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

### Issue: "Could not connect to database"
**Solution**:
- Check your `NEON_DATABASE_URL` in `.env`
- Verify your Neon project is active
- Check your internet connection

### Issue: "pgvector extension not found"
**Solution**: Run this in your Neon SQL editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Issue: Embedding generation is slow
**Solution**:
- Normal for CPU (6-8 minutes for 74K codes)
- If you have GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Issue: Out of memory during data loading
**Solution**: Reduce batch size in `load_icd10_codes.py`:
```python
BATCH_SIZE = 50  # Instead of 100
```

---

## Quick Commands Reference

```bash
# Activate virtual environment
cd backend
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Setup database
python scripts/setup_database.py

# Load data
python scripts/load_cpt_codes.py
python scripts/load_icd10_codes.py

# Run API server
uvicorn app.main:app --reload

# Run tests (future)
pytest

# Format code (future)
black app/
```

---

## Project Structure Overview

```
medical-coding-rag/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Settings
│   │   ├── database.py          # DB connection
│   │   ├── models/              # Pydantic schemas
│   │   ├── services/            # Business logic
│   │   └── utils/               # Helpers
│   ├── scripts/                 # Setup scripts
│   ├── requirements.txt
│   └── .env
├── data/                        # Your CSV/TXT files
├── guide_docs/                  # Documentation
│   ├── PROJECT_APPROACH.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── TECH_STACK.md
│   └── QUICK_START.md (this file)
└── medical-coding-rag-spec.md   # Original spec
```

---

## Next Steps

✅ **You've completed the quick start!**

Now proceed to [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) to build out the full system:

1. **Phase 2**: Core search services (Days 3-4)
2. **Phase 3**: API endpoints & LLM (Day 5)
3. **Phase 4**: Frontend (Days 6-7)
4. **Phase 5**: Documentation (Day 8)
5. **Phase 6**: Polish (Days 9-10)

**Estimated total time**: 8-10 days to portfolio-ready project

---

## Getting Help

If you get stuck:

1. **Check the guides**:
   - [PROJECT_APPROACH.md](./PROJECT_APPROACH.md) - Architecture & decisions
   - [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) - Step-by-step code
   - [TECH_STACK.md](./TECH_STACK.md) - Technology details

2. **Common issues**: See Troubleshooting section above

3. **Test incrementally**: After each step, verify it works before moving on

4. **Use the docs**:
   - FastAPI: https://fastapi.tiangolo.com
   - Neon: https://neon.tech/docs
   - pgvector: https://github.com/pgvector/pgvector

---

Happy coding! 🚀

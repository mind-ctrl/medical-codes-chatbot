import asyncio
from backend.app.database import db

async def check():
    await db.connect()
    cpt_count = await db.fetchval('SELECT COUNT(*) FROM cpt_codes')
    icd10_count = await db.fetchval('SELECT COUNT(*) FROM icd10_codes')
    print(f'CPT codes in database: {cpt_count}')
    print(f'ICD-10 codes in database: {icd10_count}')
    await db.disconnect()

asyncio.run(check())

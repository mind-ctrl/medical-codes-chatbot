"""
LLM Service for Expert Mode
Uses Perplexity API for code reranking and explanations

Only used in Expert mode to:
- Rerank top codes using medical knowledge
- Provide detailed reasoning
- Generate overall explanation
"""

from openai import AsyncOpenAI
from typing import List, Dict, Optional
import json
import logging
from ..config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Perplexity LLM client for medical code reranking

    Uses Llama 3.1 Sonar model with online mode:
    - Can access current medical guidelines
    - Better reasoning for complex cases
    - OpenAI-compatible API
    """

    def __init__(self):
        """Initialize Perplexity client"""
        self.client = AsyncOpenAI(
            api_key=settings.PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai"
        )
        self.model = settings.PERPLEXITY_MODEL
        logger.info(f"LLM service initialized with model: {self.model}")

    async def rerank_codes(
        self,
        query: str,
        cpt_codes: List[Dict],
        icd10_codes: List[Dict]
    ) -> Dict:
        """
        Rerank codes using LLM medical knowledge

        Process:
        1. Format codes with hybrid search scores
        2. Send to LLM with medical coding task
        3. Parse JSON response
        4. Fallback to similarity scores if LLM fails

        Args:
            query: Clinical description
            cpt_codes: Retrieved CPT codes (from hybrid search)
            icd10_codes: Retrieved ICD-10 codes (from hybrid search)

        Returns:
            Dict with reranked codes and explanation
        """
        # Format codes for LLM prompt
        cpt_text = "\n".join([
            f"{i+1}. {code['cpt_code']} (hybrid score: {code['confidence_score']:.2f}) - {code['description'][:100]}"
            for i, code in enumerate(cpt_codes[:10])  # Top 10 only
        ])

        icd10_text = "\n".join([
            f"{i+1}. {code['icd10_code']} (hybrid score: {code['confidence_score']:.2f}) - {code['description'][:100]}"
            for i, code in enumerate(icd10_codes[:10])
        ])

        # Construct prompt
        prompt = f"""You are a medical coding expert. Given a clinical description and candidate codes, select the most relevant codes and explain why.

Clinical Description: "{query}"

Candidate CPT Codes (Procedures):
{cpt_text}

Candidate ICD-10 Codes (Diagnoses):
{icd10_text}

Task: Select the top 5 most relevant codes from each category and provide:
1. Confidence score (0-1) for each based on relevance
2. Brief reasoning for top 3 codes in each category
3. Overall explanation of the clinical scenario

Return ONLY valid JSON in this exact format:
{{
  "cpt_codes": [
    {{"code": "99213", "confidence": 0.95, "reasoning": "Primary evaluation visit"}},
    {{"code": "82947", "confidence": 0.85, "reasoning": null}}
  ],
  "icd10_codes": [
    {{"code": "E11.9", "confidence": 0.98, "reasoning": "Type 2 diabetes diagnosis"}},
    {{"code": "E11.65", "confidence": 0.88, "reasoning": null}}
  ],
  "explanation": "This clinical presentation suggests..."
}}"""

        try:
            # Call Perplexity API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical coding expert. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Low for consistency
                max_tokens=2000,
                timeout=30.0
            )

            # Extract and parse response
            content = response.choices[0].message.content

            # Clean potential markdown code blocks
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            logger.info(f"LLM reranking successful for query: {query[:50]}...")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            return self._fallback_ranking(cpt_codes, icd10_codes)

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return self._fallback_ranking(cpt_codes, icd10_codes)

    def _fallback_ranking(
        self,
        cpt_codes: List[Dict],
        icd10_codes: List[Dict]
    ) -> Dict:
        """
        Fallback ranking when LLM is unavailable

        Uses hybrid search confidence scores directly.
        This ensures the system always works even if LLM fails.

        Args:
            cpt_codes: CPT codes with confidence scores
            icd10_codes: ICD-10 codes with confidence scores

        Returns:
            Dict in same format as LLM response
        """
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
            "explanation": "LLM unavailable - results ranked by hybrid search (vector + keyword)"
        }


# Global instance
_llm_service = None


def get_llm_service() -> LLMService:
    """
    Get or create LLM service singleton

    Returns:
        LLMService instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

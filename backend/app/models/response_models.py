"""
Response Models
Pydantic schemas for API responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class CodeSuggestion(BaseModel):
    """Individual code suggestion with metadata"""

    code: str = Field(..., description="Medical code (CPT or ICD-10)")
    description: str = Field(..., description="Code description")
    code_type: Literal["CPT", "ICD-10"] = Field(..., description="Type of code")
    category: Optional[str] = Field(None, description="Category or chapter")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    reasoning: Optional[str] = Field(None, description="Explanation (expert mode only)")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "E11.9",
                "description": "Type 2 diabetes mellitus without complications",
                "code_type": "ICD-10",
                "category": "E00-E89",
                "confidence_score": 0.95,
                "reasoning": "Primary diagnosis for type 2 diabetes"
            }
        }


class CodingResponse(BaseModel):
    """Response model for code suggestions"""

    query: str = Field(..., description="Original query")
    cpt_codes: List[CodeSuggestion] = Field(..., description="CPT procedure codes")
    icd10_codes: List[CodeSuggestion] = Field(..., description="ICD-10 diagnosis codes")
    search_mode: str = Field(..., description="Mode used for search")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    explanation: Optional[str] = Field(None, description="Overall explanation (expert mode)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Patient with type 2 diabetes",
                "cpt_codes": [
                    {
                        "code": "99213",
                        "description": "Office visit, established patient",
                        "code_type": "CPT",
                        "category": "Evaluation & Management",
                        "confidence_score": 0.89,
                        "reasoning": None
                    }
                ],
                "icd10_codes": [
                    {
                        "code": "E11.9",
                        "description": "Type 2 diabetes mellitus without complications",
                        "code_type": "ICD-10",
                        "category": "E00-E89",
                        "confidence_score": 0.98,
                        "reasoning": None
                    }
                ],
                "search_mode": "quick",
                "processing_time_ms": 234.5,
                "explanation": None
            }
        }


class StatsResponse(BaseModel):
    """Database statistics"""

    total_cpt_codes: int
    total_icd10_codes: int
    categories: List[str]
    chapters: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "total_cpt_codes": 1164,
                "total_icd10_codes": 74260,
                "categories": ["AAA", "AMP", "ANE", "CAR"],
                "chapters": ["A00-B99", "E00-E89", "I00-I99"]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    database: str
    embedding_model: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "database": "connected",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Clinical description must be at least 10 characters",
                "status_code": 422
            }
        }

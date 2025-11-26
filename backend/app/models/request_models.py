"""
Request Models
Pydantic schemas for API requests with validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


class CodingQuery(BaseModel):
    """Request model for code search"""

    clinical_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Clinical description or symptoms",
        examples=["Patient with type 2 diabetes and chest pain"]
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
        description="Filter CPT codes by category (e.g., 'AAA', 'AMP')"
    )

    filter_chapter: Optional[str] = Field(
        default=None,
        description="Filter ICD-10 codes by chapter (e.g., 'E00-E89')"
    )

    @field_validator('clinical_description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate and clean clinical description"""
        if not v.strip():
            raise ValueError("Clinical description cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "clinical_description": "Patient with type 2 diabetes and hypertension",
                "max_results": 5,
                "search_mode": "standard",
                "filter_category": None,
                "filter_chapter": None
            }
        }

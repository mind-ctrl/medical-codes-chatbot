"""
Custom Exception Classes
"""

class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass


class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails"""
    pass


class SearchError(Exception):
    """Raised when search operation fails"""
    pass


class LLMServiceError(Exception):
    """Raised when LLM service fails"""
    pass


class ValidationError(Exception):
    """Raised when input validation fails"""
    pass

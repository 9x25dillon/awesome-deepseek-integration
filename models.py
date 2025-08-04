from pydantic import BaseModel
from typing import List, Optional

class RAGRequest(BaseModel):
    """Request model for RAG endpoint"""
    query: str
    top_k: Optional[int] = 3
    use_polynomial_analysis: Optional[bool] = True

class PolynomialDocument(BaseModel):
    """Model representing a polynomial document with metadata"""
    expression: str
    degree: Optional[int] = None
    term_count: Optional[int] = None
    complexity_score: Optional[float] = None
    variables: Optional[List[str]] = None
    coefficients: Optional[List[float]] = None
    metadata: Optional[dict] = None

class RAGResponse(BaseModel):
    """Response model for RAG endpoint"""
    answer: str
    context: List[str]
    query: str
    top_k: int
    polynomial_analysis_used: Optional[bool] = None

class PolynomialAnalysisRequest(BaseModel):
    """Request model for polynomial analysis"""
    polynomials: List[str]
    analysis_type: Optional[str] = "basic"  # basic, advanced, similarity

class PolynomialAnalysisResponse(BaseModel):
    """Response model for polynomial analysis"""
    results: List[PolynomialDocument]
    analysis_metadata: Optional[dict] = None

class EntropyProcessingRequest(BaseModel):
    """Request model for entropy engine processing"""
    input_value: str
    transformations: List[str]
    max_depth: Optional[int] = 5
    entropy_limits: Optional[List[float]] = None
    use_julia: Optional[bool] = True

class EntropyProcessingResponse(BaseModel):
    """Response model for entropy engine processing"""
    input_value: str
    final_value: str
    input_type: str
    entropy_change: float
    entropy_trend: str
    transformations_applied: int
    processing_graph: dict
    polynomial_features: Optional[dict] = None
    julia_analysis: Optional[dict] = None

class EntropyRAGRequest(BaseModel):
    """Request model for entropy-enhanced RAG"""
    query: str
    entropy_transformations: Optional[List[str]] = None
    top_k: Optional[int] = 3
    use_polynomial_analysis: Optional[bool] = True
    entropy_threshold: Optional[float] = None
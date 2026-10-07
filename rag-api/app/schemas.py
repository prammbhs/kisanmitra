from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import DEFAULT_COLLECTIONS, TOP_K


class Filters(BaseModel):
    """Optional exact-match metadata filters (KCC values are UPPERCASE for state/district)."""
    StateName: Optional[str] = None
    DistrictName: Optional[str] = None
    Crop: Optional[str] = None
    Season: Optional[str] = None
    Sector: Optional[str] = None
    QueryType: Optional[str] = None

    def as_dict(self) -> Dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    collections: List[str] = Field(default_factory=lambda: list(DEFAULT_COLLECTIONS))
    k: int = Field(TOP_K, ge=1, le=50)
    filters: Optional[Filters] = None


class Source(BaseModel):
    index: int
    collection: str
    score: float
    text: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[Source]
    latency_ms: float


class ChatRequest(SearchRequest):
    pass


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieval_ms: float
    generation_ms: float

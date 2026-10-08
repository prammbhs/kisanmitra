from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.config import DEFAULT_COLLECTIONS, TOP_K


# --- Auth & Profile Schemas ---

class UserRegister(BaseModel):
    phone_or_email: str = Field(..., min_length=3, description="Phone number or email")
    password: str = Field(..., min_length=6, description="Minimum 6 characters")
    full_name: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    primary_crops: Optional[List[str]] = Field(default_factory=list, description="Optional primary crops")
    land_acres: Optional[float] = Field(default=None, description="Optional farm size in acres")
    preferred_language: Optional[str] = Field(default="hi", description="Preferred response language")


class UserLogin(BaseModel):
    phone_or_email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class ProfileOut(BaseModel):
    user_id: int
    phone_or_email: str
    full_name: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    primary_crops: List[str] = []
    land_acres: Optional[float] = None
    preferred_language: str = "hi"
    created_at: datetime


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    primary_crops: Optional[List[str]] = None
    land_acres: Optional[float] = None
    preferred_language: Optional[str] = None


# --- Search & RAG Schemas ---

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


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Farmer message or question")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID for multi-turn memory")
    collections: List[str] = Field(default_factory=lambda: list(DEFAULT_COLLECTIONS))
    k: int = Field(TOP_K, ge=1, le=50)
    filters: Optional[Filters] = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: Optional[str] = None
    sources: List[Source]
    retrieval_ms: float
    generation_ms: float

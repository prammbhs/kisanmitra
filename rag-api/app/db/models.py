from datetime import datetime, timezone
from typing import List, Optional
from sqlmodel import Field, Relationship, SQLModel
import json


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    phone_or_email: str = Field(index=True, unique=True, nullable=False)
    hashed_password: str = Field(nullable=False)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=get_utc_now)

    # 1-to-1 relationship with profile
    profile: Optional["FarmerProfile"] = Relationship(
        back_populates="user", sa_relationship_kwargs={"cascade": "all, delete"}
    )


class FarmerProfile(SQLModel, table=True):
    __tablename__ = "farmer_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, unique=True, nullable=False)
    full_name: Optional[str] = Field(default=None)
    state: Optional[str] = Field(default=None, index=True)
    district: Optional[str] = Field(default=None, index=True)
    
    # Store crops as JSON string in SQLite (e.g. '["Cotton", "Wheat"]')
    crops_json: str = Field(default="[]")
    
    # Land size in acres (optional)
    land_acres: Optional[float] = Field(default=None)
    
    # Preferred language (default Hindi)
    preferred_language: str = Field(default="hi")
    
    updated_at: datetime = Field(default_factory=get_utc_now)

    user: Optional[User] = Relationship(back_populates="profile")

    @property
    def primary_crops(self) -> List[str]:
        try:
            return json.loads(self.crops_json) if self.crops_json else []
        except Exception:
            return []

    @primary_crops.setter
    def primary_crops(self, crops: List[str]) -> None:
        self.crops_json = json.dumps(crops or [], ensure_ascii=False)

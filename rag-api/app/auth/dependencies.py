"""FastAPI authentication dependencies."""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session, select

from app.auth.security import decode_access_token
from app.db.database import get_session
from app.db.models import FarmerProfile, User

security = HTTPBearer(auto_error=False)


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: Session = Depends(get_session),
) -> Optional[User]:
    """Returns the authenticated user if valid token provided, else None."""
    if not credentials:
        return None
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    user = session.get(User, int(user_id))
    return user if user and user.is_active else None


def get_current_user(
    user: Optional[User] = Depends(get_current_user_optional),
) -> User:
    """Enforces authentication; raises 401 if missing or invalid."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_current_profile(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> FarmerProfile:
    """Returns the profile of the authenticated user."""
    profile = session.exec(select(FarmerProfile).where(FarmerProfile.user_id == user.id)).first()
    if not profile:
        # Create empty profile if not exists
        profile = FarmerProfile(user_id=user.id)
        session.add(profile)
        session.commit()
        session.refresh(profile)
    return profile

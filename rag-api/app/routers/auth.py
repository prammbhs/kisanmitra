"""Authentication Router: Registration and Login."""
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.auth.security import create_access_token, get_password_hash, verify_password
from app.config import ACCESS_TOKEN_EXPIRE_MINUTES
from app.db.database import get_session
from app.db.models import FarmerProfile, User
from app.schemas import Token, UserLogin, UserRegister

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(req: UserRegister, session: Session = Depends(get_session)):
    # Check if user already exists
    existing = session.exec(select(User).where(User.phone_or_email == req.phone_or_email.strip())).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this phone or email already exists."
        )

    # Create new User
    user = User(
        phone_or_email=req.phone_or_email.strip(),
        hashed_password=get_password_hash(req.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    # Initialize Farmer Profile
    profile = FarmerProfile(
        user_id=user.id,
        full_name=req.full_name,
        state=req.state.upper() if req.state else None,
        district=req.district.upper() if req.district else None,
        land_acres=req.land_acres,
        preferred_language=req.preferred_language or "hi",
    )
    if req.primary_crops:
        profile.primary_crops = req.primary_crops

    session.add(profile)
    session.commit()

    # Generate JWT
    token_str = create_access_token(
        data={"sub": str(user.id), "phone": user.phone_or_email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return Token(
        access_token=token_str,
        token_type="bearer",
        expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )


@router.post("/login", response_model=Token)
def login(req: UserLogin, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.phone_or_email == req.phone_or_email.strip())).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect phone/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated."
        )

    token_str = create_access_token(
        data={"sub": str(user.id), "phone": user.phone_or_email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return Token(
        access_token=token_str,
        token_type="bearer",
        expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )


@router.post("/logout")
def logout():
    """Client-side token disposal confirmation."""
    return {"status": "ok", "message": "Successfully logged out. Please discard your access token."}

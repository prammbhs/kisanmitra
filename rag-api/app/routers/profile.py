from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.auth.dependencies import get_current_profile, get_current_user
from app.db.database import get_session
from app.db.models import FarmerProfile, User
from app.schemas import ProfileOut, ProfileUpdate

router = APIRouter(prefix="/user", tags=["Farmer Profile"])


@router.get("/profile", response_model=ProfileOut)
def get_profile(
    user: User = Depends(get_current_user),
    profile: FarmerProfile = Depends(get_current_profile),
):
    return ProfileOut(
        user_id=user.id,
        phone_or_email=user.phone_or_email,
        full_name=profile.full_name,
        state=profile.state,
        district=profile.district,
        primary_crops=profile.primary_crops,
        land_acres=profile.land_acres,
        preferred_language=profile.preferred_language,
        created_at=user.created_at,
    )


@router.put("/profile", response_model=ProfileOut)
def update_profile(
    update_data: ProfileUpdate,
    user: User = Depends(get_current_user),
    profile: FarmerProfile = Depends(get_current_profile),
    session: Session = Depends(get_session),
):
    if update_data.full_name is not None:
        profile.full_name = update_data.full_name
    if update_data.state is not None:
        profile.state = update_data.state.upper()
    if update_data.district is not None:
        profile.district = update_data.district.upper()
    if update_data.primary_crops is not None:
        profile.primary_crops = update_data.primary_crops
    if update_data.land_acres is not None:
        profile.land_acres = update_data.land_acres
    if update_data.preferred_language is not None:
        profile.preferred_language = update_data.preferred_language

    profile.updated_at = datetime.now(timezone.utc)
    session.add(profile)
    session.commit()
    session.refresh(profile)

    return ProfileOut(
        user_id=user.id,
        phone_or_email=user.phone_or_email,
        full_name=profile.full_name,
        state=profile.state,
        district=profile.district,
        primary_crops=profile.primary_crops,
        land_acres=profile.land_acres,
        preferred_language=profile.preferred_language,
        created_at=user.created_at,
    )

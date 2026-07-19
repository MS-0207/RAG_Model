from ingest.loader import load_all_documents
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from Database.db_dependecies import get_db
from Database.models import Feedback


router = APIRouter(prefix="/feedback", tags=["Feedback"])

# -------------------------
# Request Models
# -------------------------


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(..., max_length=1000)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


# -------------------------
# Response Models
# -------------------------


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: int


class LoginResponse(BaseModel):
    status: str
    access_token: str
    token_type: str


class UploadResponse(BaseModel):
    status: str
    documents_loaded: int


# -------------------------
# Endpoints
# -------------------------


@router.post(
    "",
    response_model=FeedbackResponse,
)
def save_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    feedback = Feedback(
        query=request.query,
        rating=request.rating,
        comment=request.comment,
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return FeedbackResponse(
        status="success",
        message="Feedback saved successfully",
        feedback_id=feedback.id,
    )


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    # Authentication logic goes here
    return LoginResponse(
        status="success",
        access_token="dummy_token",
        token_type="Bearer",
    )


@router.post("/upload", response_model=UploadResponse)
def upload():
    documents = load_all_documents()

    return UploadResponse(
        status="success",
        documents_loaded=len(documents),
    )

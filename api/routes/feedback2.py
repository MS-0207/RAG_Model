from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.dependencies import get_db
from Database.models import Feedback

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)

class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(default="", max_length=1000)


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: int


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
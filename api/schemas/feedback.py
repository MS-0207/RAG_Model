from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(default="", max_length=1000)


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: int

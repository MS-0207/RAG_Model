from sqlalchemy.orm import Session
from Database.models import Feedback


def create_feedback(
    db: Session,
    query: str,
    rating: int,
    comment: str,
) -> Feedback:

    feedback = Feedback(
        query=query,
        rating=rating,
        comment=comment,
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return feedback
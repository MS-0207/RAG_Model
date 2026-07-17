from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from Database.connection import Base


class Feedback(Base):
    __tablename__ = "feedback_1"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )


    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    comment: Mapped[str] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
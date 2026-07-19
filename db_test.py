from Database.connection import SessionLocal
from Database.models import Feedback

db = SessionLocal()

feedback = Feedback(query="Hello", rating=5, comment="Inserted from Python")

db.add(feedback)
db.commit()
db.refresh(feedback)

print(feedback.id)

db.close()

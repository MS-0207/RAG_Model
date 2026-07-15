from Database.connection import engine, Base

# Import all models so SQLAlchemy knows about them
from Database.models import Feedback

Base.metadata.create_all(bind=engine)

print("✅ Database connected successfully.")
print("✅ Tables created (if they didn't already exist).")
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# DATABASE_URL = "sqlite:///./rag.db"
DATABASE_URL = ("postgresql+psycopg://postgres:Artillary177#@localhost:5432/Rag_DB")


# Engine - The Engine is the component in SQLAlchemy that creates and manages the connection to the database.
# All database communication goes through the Engine.

engine = create_engine(DATABASE_URL)

#A Session is a workspace used to interact with the database.
# It allows you to query, insert, update, and delete data, and manages transactions.
# Session is the object you use to perform CRUD operations on the database.

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

# Base is the parent class from which all ORM models inherit.
# It tells SQLAlchemy that these Python classes represent database tables.

class Base(DeclarativeBase):
    pass

# SQLAlchemy → The library that connects Python to the database.
# Engine → Creates and manages the database connection.
# Session → Performs CRUD operations through the engine.
# Base → Parent class that maps Python classes to database tables.
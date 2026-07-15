from fastapi import Header, HTTPException
from utils.config import settings
from collections.abc import Generator
from sqlalchemy.orm import Session
from Database.connection import SessionLocal


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    return x_api_key

#------------------------------------------
# Creating Dependencies
#------------------------------------------

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from retrieval.BM25 import load_vector_store

#------------------------------------------
# Below function has one job
# get_vector_store()    ↓
# load and return the FAISS vector store
#-------------------------------------------

def get_vector_store() -> FAISS:
    return load_vector_store()



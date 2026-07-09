# Configuration settings
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAW_DIR = os.getenv("RAW_DIR")
PROCESSED_DIR = os.getenv("PROCESSED_DIR")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR")
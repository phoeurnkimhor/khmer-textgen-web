from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
load_dotenv()

def get_engine():
    """
    Create and return a SQLAlchemy engine
    """
    DATABASE_URL = os.getenv("DB_URL")  # Neon full connection URL
    if not DATABASE_URL:
        raise ValueError("DB_URL not found in environment variables")

    engine = create_engine(DATABASE_URL, future=True)
    return engine


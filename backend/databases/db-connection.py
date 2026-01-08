from sqlalchemy import create_engine
from configs.config import DB_URL

def get_engine():
    """
    Create and return a SQLAlchemy engine
    """
    DATABASE_URL = DB_URL  # Neon full connection URL
    if not DATABASE_URL:
        raise ValueError("DB_URL not found in environment variables")

    engine = create_engine(DATABASE_URL, future=True)
    return engine


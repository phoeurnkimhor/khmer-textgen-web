from dotenv import load_dotenv
import os

load_dotenv()

FRONTEND_URLS = os.getenv("FRONTEND_URLS")
DB_URL = os.getenv("DB_URL")

from dotenv import load_dotenv
import os

load_dotenv()

FRONTEND_URLS = os.getenv(
    "FRONTEND_URL",
    "http://localhost:3000"
).split(",")

DB_URL = os.getenv("DB_URL")

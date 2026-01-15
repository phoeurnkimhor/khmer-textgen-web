from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from configs.config import FRONTEND_URLS
from routers import text_generation

app = FastAPI(
    title="Khmer Text Generation",
    description="API for Khmer Text Generation",
    version="1.0.0"
)

app.include_router(text_generation.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "API for Khmer Text Generation",
        "endpoints": {
            "translate": "/generate",
            "docs": "/docs",
            "health": "/health",
        }
    }

@app.get("/")
async def health_check():
    return {"status": "ok"}

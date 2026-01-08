from fastapi import APIRouter
from schemas.text_generation import GenerateRequest, GenerateResponse
from services.text_generation import generate_text
from models.loader import load_model

router = APIRouter(prefix="/generate", tags=["Text Generation"])

model = load_model('./best_model.pt')

@router.post("/", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    output = generate_text(
        model=model,
        seed_text=req.text,
        max_length=req.length,
        seq_len=req.seq_len
    )
    response = GenerateResponse(generated_text=output)
    return response

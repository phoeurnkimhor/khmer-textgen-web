from pydantic import BaseModel

class GenerateRequest(BaseModel):
    text: str
    length: int = 100
    seq_len: int = 50
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "សួស្តី"
            }
        }

class GenerateResponse(BaseModel):
    generated_text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "generated_text": "សួស្តីអ្នកទាំងអស់គ្នា"
            }
        }
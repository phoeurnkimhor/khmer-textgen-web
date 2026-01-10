from pydantic import BaseModel

class TrainingRequest(BaseModel):
    dataset_path: str
    chunk_size: int = 120
    seq_len: int = 50
    batch_size: int = 32
    epochs: int = 30
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    patience: int = 3

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_path": "/path/to/dataset.txt",
                "chunk_size": 120,
                "seq_len": 50,
                "batch_size": 32,
                "epochs": 30,
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "patience": 3
            }
        }

class TrainingResponse(BaseModel):
    message: str
    test_perplexity: float
    test_accuracy: float
    model_path: str
    public_url: str  

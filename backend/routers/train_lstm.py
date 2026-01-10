from fastapi import APIRouter, HTTPException
from schemas.training import TrainingRequest, TrainingResponse  
from utils.clean import clean_text
from utils.build_vocab import build_vocab
from utils.data_loader import get_dataloader
from utils.preprocessing import split_sentences, chunk_text
from utils.evaluate import evaluate_model
from services.training_loop import training
from models.architecture import LSTMTST
from services.save_model import upload_file
from supabase import create_client
import pandas as pd
import torch
import os
from datetime import datetime


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME", "models")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter(prefix="/train", tags=["Model Training"])

@router.post("/", response_model=TrainingResponse)
def train(train: TrainingRequest):

    # CONSTANT
    data_path = train.dataset_path
    chunk_size = train.chunk_size
    seq_len = train.seq_len
    batch_size = train.batch_size
    epochs = train.epochs
    patience = train.patience
    embedding_dim = train.embedding_dim
    hidden_dim = train.hidden_dim
    num_layers = train.num_layers

    try:
        df = pd.read.csv(data_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read dataset. Ensure the file path is correct and the file is a valid CSV.")
    
    col_name = df.columns[0]
    df[col_name] = df[col_name].astype(str)

    # clean text
    df[col_name] = df[col_name].apply(clean_text)

    # split long sentence into fixed size
    df['sentences'] = df[col_name].apply(split_sentences)
    df['chunks'] = df['sentences'].apply(lambda sents: [chunk for sent in sents for chunk in chunk_text(sent, chunk_size=chunk_size)])

    # explode chunks into separate rows and remove empty chunks
    df_exploded = df.explode('chunks', ignore_index=True)
    df_exploded = df_exploded[df_exploded['chunks'].notna() & (df_exploded['chunks'] != '')]

    # create new dataframe for training
    df = pd.DataFrame({
        'sentence': df_exploded['chunks'],
        'target': df_exploded['chunks']
    })

    all_text, vocab, stoi, itos = build_vocab(df)

    train_loader, val_loader, test_loader = get_dataloader(
        texts=all_text,
        seq_len=seq_len, 
        batch_size=batch_size, 
        stoi=stoi
    )

    model = LSTMTST(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Training returns the best model and its validation loss
    trained_model, best_val_loss = training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=len(vocab),
        stoi=stoi,
        itos=itos,
        epochs=epochs,
        patience=patience
    )
    
    perplexity, accuracy = evaluate_model(trained_model, test_loader)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("saved_models", exist_ok=True)
    model_filename = f"lstm_model_{timestamp}.pt"
    model_path = os.path.join("saved_models", model_filename)
    
    torch.save({
        'model_state_dict' : trained_model.state_dict(),
        'vocab_size': len(stoi),
        'stoi' : stoi,
        'itos' : itos
        }, model_path)

    with open(model_path, "rb") as f:
        try:
            upload_file(
                bucket_name=BUCKET_NAME,
                file_name=model_filename,
                file=f,
                access_token=SUPABASE_KEY,
                supabase_url=SUPABASE_URL
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase upload failed: {e}")
            
    expires_in_seconds = 3600
    signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(
    file_name=model_filename,
    expires_in=expires_in_seconds
    ).signed_url
            

    return TrainingResponse(
        message="Training completed and model uploaded successfully.",
        test_perplexity=perplexity,
        test_accuracy=accuracy,
        model_path=model_path,
        signed_url=signed_url
    )




from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from schemas.training import TrainingRequest, TrainingResponse  # Added TrainingResponse
from utils.clean import clean_text
from utils.build_vocab import build_vocab
from utils.data_loader import get_dataloader
from utils.preprocessing import split_sentences, chunk_text
from utils.evaluate import evaluate_model
from services.training_loop import training
from models.architecture import LSTMTST
import pandas as pd
import torch
import pickle
import os
from datetime import datetime


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

    # load data
    df = pd.read.csv(data_path)
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
    
    # Save final model and vocabulary for user download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = f"{models_dir}/lstm_model_{timestamp}.pth"
    vocab_path = f"{models_dir}/vocab_{timestamp}.pkl"
    
    torch.save(trained_model.state_dict(), model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump({'vocab': vocab, 'stoi': stoi, 'itos': itos}, f)
    
    # Clean up temporary checkpoint
    temp_checkpoint = "temp_checkpoints/best_model_temp.pt"
    if os.path.exists(temp_checkpoint):
        os.remove(temp_checkpoint)
    
    return TrainingResponse(
        message="Training completed successfully.",
        test_perplexity=perplexity,
        test_accuracy=accuracy,
        model_path=model_path,
        vocab_path=vocab_path
    )

@router.get("/download/{file_path:path}")
def download_model(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )



# Khmer Text Generation API

A FastAPI-based backend service for generating Khmer text using an LSTM neural network model. This API provides endpoints for text generation and model training specifically designed for the Khmer language.

## 🌟 Features

- **Text Generation**: Generate Khmer text based on seed input using a pre-trained LSTM model
- **Model Training**: Train new LSTM models with custom datasets
- **Database Integration**: Store predictions and training data using PostgreSQL
- **RESTful API**: Well-documented API endpoints with Pydantic schemas
- **Docker Support**: Containerized deployment ready
- **CORS Enabled**: Cross-origin resource sharing for frontend integration

## 🏗️ Architecture

The backend follows a clean, modular architecture:

```
backend/
├── main.py                 # FastAPI application entry point
├── configs/                # Configuration management
├── routers/               # API route handlers
├── schemas/               # Pydantic models for request/response
├── services/              # Business logic layer
├── models/                # Neural network architecture and model loading
├── databases/             # Database connection and operations
├── utils/                 # Utility functions for data processing
└── best_model.pt          # Pre-trained LSTM model
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL database
- PyTorch

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/phoeurnkimhor/khmer-tst-web
   cd backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   
   Create a `.env` file in the backend directory:
   ```env
   DB_URL=postgresql://user:password@host:port/database
   FRONTEND_URL=http://localhost:3000
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```


## 📋 API Endpoints

### Text Generation

**POST** `/generate/`

Generate Khmer text based on seed input.

**Request Body:**
```json
{
  "text": "សួស្តី",
  "length": 100,
  "seq_len": 50
}
```

**Response:**
```json
{
  "generated_text": "សួស្តីអ្នកទាំងអស់គ្នា..."
}
```

### Model Training

**POST** `/train/`

Train a new LSTM model with custom dataset.

**Request Body:**
```json
{
  "dataset_path": "/path/to/dataset.csv",
  "chunk_size": 120,
  "seq_len": 50,
  "batch_size": 32,
  "epochs": 30,
  "embedding_dim": 128,
  "hidden_dim": 256,
  "num_layers": 2,
  "patience": 3
}
```

### Health Check

**GET** `/`

Returns API status and available endpoints.

## 🧠 Model Architecture

The system uses an LSTM (Long Short-Term Memory) neural network specifically designed for Khmer text generation:

- **Model**: LSTMTST (LSTM Text-to-Speech Transformer)
- **Vocabulary**: 85 Khmer characters including consonants, vowels, and diacritics
- **Architecture**: Embedding layer → LSTM layers → Linear output layer
- **Default Configuration**:
  - Embedding Dimension: 128
  - Hidden Dimension: 256
  - Number of Layers: 2
  - Sequence Length: 50

### Model Components

1. **Embedding Layer**: Converts character indices to dense vectors
2. **LSTM Layers**: Sequential processing for character-level language modeling
3. **Linear Layer**: Maps hidden states to vocabulary probabilities

## 📊 Data Processing Pipeline

1. **Text Cleaning**: Removes unwanted characters and normalizes text
2. **Sentence Splitting**: Breaks long texts into manageable sentences
3. **Chunking**: Creates fixed-size chunks for training
4. **Vocabulary Building**: Maps characters to indices
5. **Data Loading**: Batched data preparation for training

## 🗄️ Database Schema

The application uses PostgreSQL to store:
- Prediction history (input text and generated output)
- Training metadata and results
- Model performance metrics

## 📁 Key Files Overview

### Core Application
- **`main.py`**: FastAPI application setup, middleware, and routing
- **`configs/config.py`**: Environment configuration management

### API Layer
- **`routers/text_generation.py`**: Text generation endpoint
- **`routers/train_lstm.py`**: Model training endpoint
- **`schemas/`**: Pydantic models for request/response validation

### Business Logic
- **`services/text_generation.py`**: Text generation logic
- **`services/training_loop.py`**: Model training implementation

### Model & ML
- **`models/architecture.py`**: LSTM model definition
- **`models/loader.py`**: Model loading and vocabulary mapping
- **`best_model.pt`**: Pre-trained model weights

### Data Processing
- **`utils/`**: Data preprocessing, cleaning, and evaluation utilities
- **`databases/`**: Database connection and data persistence

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_URL` | PostgreSQL connection string | Required |
| `FRONTEND_URL` | Allowed CORS origins | `http://localhost:3000` |

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_dim` | Embedding vector size | 128 |
| `hidden_dim` | LSTM hidden size | 256 |
| `num_layers` | Number of LSTM layers | 2 |
| `seq_len` | Input sequence length | 50 |
| `batch_size` | Training batch size | 32 |

## 📖 API Documentation

Once the server is running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

---



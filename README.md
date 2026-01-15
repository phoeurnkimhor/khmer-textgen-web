# Khmer Text Generation Web

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

A modern web application for Khmer text generation using AI-powered LSTM models. 

## Features

- **AI-Powered Text Generation**: LSTM-based neural network trained specifically for Khmer language
- **Real-time Processing**: Fast text generation with responsive UI
- **Modern Interface**: Clean, intuitive design built with Next.js and TailwindCSS
- **Responsive Design**: Works seamlessly across desktop and mobile devices
- **Docker Ready**: Containerized deployment for easy setup
- **REST API**: Well-documented FastAPI backend with automatic documentation
- **Data Persistence**: PostgreSQL database integration via Neon

## Architecture

```
khmertst-web/
├── backend/              # FastAPI + PyTorch LSTM Model
│   ├── models/           # Neural network architecture
│   ├── routers/          # API endpoints
│   ├── services/         # Business logic
│   ├── utils/            # Data processing utilities
│   └── best_model.pt     # Pre-trained LSTM model
├── frontend/             # Next.js React Application
│   └── noro-khmer/       # Main app directory
│       ├── app/          # App router pages
│       ├── components/   # React components
│       └── lib/          # Utility functions
└── docker-compose.yml    # Container orchestration
```

## Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended)
- **Node.js 18+** (for local development)
- **Python 3.11+** (for local development)
- **PostgreSQL** (or Neon account)

## API Endpoints

### Text Generation
- **POST** `/generate/` - Generate Khmer text from seed input
- **POST** `/train/` - Train new LSTM model with custom dataset

### Monitoring
- **GET** `/` - API information and available endpoints
- **GET** `/health` - Health check endpoint
- **GET** `/docs` - Interactive API documentation

## AI Model Details

### LSTM Architecture
- **Model**: Custom LSTM Text Generation Network
- **Vocabulary**: 85 Khmer characters (consonants, vowels, diacritics)
- **Parameters**: 
  - Embedding Dimension: 128
  - Hidden Dimension: 256
  - Number of Layers: 2
  - Sequence Length: 50

### Text Processing Pipeline
1. **Text Cleaning**: Normalization and character filtering
2. **Tokenization**: Character-level tokenization for Khmer script
3. **Sequence Generation**: LSTM-based next-character prediction
4. **Post-processing**: Text formatting and style application

## Frontend Features

### Components
- **Sidebar**: Model selection with visual indicators
- **Translator Area**: Input/output interface with real-time feedback
- **UI Components**: Modern design system with Radix UI primitives

### Styling Options
- **Royal Translator**: Formal royalty addressing tone
- **Formal Translator**: Professional business communication
- **Casual Translator**: Friendly conversational style
- **Simplifier**: Easy-to-understand simplified text

## Tech Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **PyTorch**: Deep learning framework for LSTM model
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: Database ORM
- **PostgreSQL**: Primary database
- **Uvicorn**: ASGI server

### Frontend
- **Next.js 16**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **TailwindCSS**: Utility-first CSS framework
- **Radix UI**: Headless UI components
- **Lucide React**: Modern icon library
- **React Hook Form**: Form management
- **Sonner**: Toast notifications

### Infrastructure
- **Docker**: Containerization
- **Neon**: PostgreSQL hosting

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEON_DB_URL` | PostgreSQL connection string | Required |
| `API_URL` | Backend API URL for frontend | `http://localhost:8000` |
| `FRONTEND_URL` | Allowed CORS origins | `http://localhost:3000` |

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_dim` | Embedding vector size | 128 |
| `hidden_dim` | LSTM hidden size | 256 |
| `num_layers` | Number of LSTM layers | 2 |
| `seq_len` | Input sequence length | 50 |
| `max_length` | Generated text length | 100 |

## Development

### Project Structure

#### Backend (`/backend`)
```
backend/
├── main.py              # FastAPI application
├── configs/             # Configuration management
├── routers/             # API route definitions
├── schemas/             # Pydantic models
├── services/            # Business logic
├── models/              # ML model architecture
├── databases/           # Database operations
├── utils/               # Helper functions
└── best_model.pt        # Pre-trained model
```

#### Frontend (`/frontend/noro-khmer`)
```
noro-khmer/
├── app/                 # Next.js app router
├── components/          # React components
├── lib/                 # Utilities
├── hooks/              # Custom React hooks
├── public/             # Static assets
└── styles/             # Global styles
```

## Contributors

| Avatar | Name | GitHub |
|--------|------|--------|
| ![BunkhloemPEL](https://github.com/BunkhloemPEL.png?size=40) | PEL Bunkhloem | [BunkhloemPEL](https://github.com/BunkhloemPEL) |
| ![PHETH-SORIYUON](https://github.com/PHETH-SORIYUON.png?size=40) | PHETH Soriyuon | [PHETH-SORIYUON](https://github.com/PHETH-SORIYUON) |
| ![DRDomm79](https://github.com/DRDomm79.png?size=40) | PHO Rotha | [DRDomm79](https://github.com/DRDomm79) |
| ![jame-xplore](https://github.com/jame-xplore.png?size=40) | PHOEUN Rajame | [jame-xplore](https://github.com/jame-xplore) |
| ![phoeurnkimhor](https://github.com/phoeurnkimhor.png?size=40) | PHOEURN Kimhor | [phoeurnkimhor](https://github.com/phoeurnkimhor) |
| ![Povsundra](https://github.com/Povsundra.png?size=40) | PHORN Sreypov | [Povsundra](https://github.com/Povsundra) |
| ![Petit-x-garcon](https://github.com/Petit-x-garcon.png?size=40) | YIN Sambat | [Petit-x-garcon](https://github.com/Petit-x-garcon) |

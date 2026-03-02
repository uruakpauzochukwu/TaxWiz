# TaxWiz 📚

TaxWiz is an AI-powered document retrieval and knowledge assistant for Nigerian tax laws. It leverages OpenAI embeddings and Pinecone vector database to enable semantic search and efficient retrieval of sections, definitions, and related information from tax documents.

> Built with **FastAPI**, **Python**, and designed for modular expansion across multiple users and documents.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Environment Variables](#environment-variables)
- [Running the API](#running-the-api)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

TaxWiz allows users to:

- Extract and chunk large legal documents (e.g., Nigerian Tax Acts)
- Store and index document embeddings in Pinecone
- Perform semantic search using OpenAI embeddings
- Retrieve relevant definitions, sections, and subsections from legal texts

It is designed to be team-friendly with relative file paths, modular Python utilities, and a FastAPI backend for easy integration with frontends.

---

## Features

- ✅ Automatic document chunking with overlap to preserve context
- ✅ OpenAI `text-embedding-3-small` integration for semantic understanding
- ✅ Pinecone vector database integration for fast retrieval
- ✅ Modular utils for embeddings, chunking, and upserting
- ✅ FastAPI backend with clean routes and schemas
- ✅ Notebook support for data preparation and Pinecone upload

---

## Project Structure

```
TaxWiz/
│
├── api/                   # FastAPI backend
│   ├── main.py            # Entry point of API
│   ├── routes.py          # API endpoints
│   ├── schema.py          # Request/response schemas
│   └── __init__.py
│
├── notebooks/             # Jupyter notebooks
│   ├── Chunking_notebook.ipynb
│   ├── Extraction_notebook.ipynb
│   ├── pinecone_data_upload.ipynb
│   └── retrieval.ipynb
│
├── utils/                 # Utility functions
│   ├── utils.py
│   └── __init__.py
│
├── data/                  # Raw and processed documents
│   ├── NIGERIA_TAX_ACT_2025.pdf
│   └── processed/
│       ├── NIGERIA_TAX_ACT_2025_processed.json
│       └── NIGERIA_TAX_ACT_2025_processed_chunks.json
│
├── outputs/               # Generated files
│   ├── NIGERIA_TAX_ACT_2025_processed.json
│   └── NIGERIA_TAX_ACT_2025_processed_chunks.json
│
├── prompts.py             # Prompt templates for OpenAI
├── requirements.txt       # Python dependencies
├── chunks_dict.pkl        # Pickled chunk data
└── .env                   # API keys and environment variables
```

---

## Setup and Installation

**1. Clone the repository:**

```bash
git clone <repository_url>
cd TaxWiz/TaxWiz
```

**2. Create a virtual environment:**

```bash
python -m venv venv
```

**3. Activate the environment:**

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

**4. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws  # replace with your Pinecone environment
```

> ⚠️ **Important:** Never commit your `.env` file to version control.

---

## Running the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

The API will be available at: **http://127.0.0.1:8000**

Test the endpoints via **Swagger UI** at `http://127.0.0.1:8000/docs`.

---

## Usage

### 1. Chunk Documents

```python
from utils.utils import chunk_text, load_chunks_pickle

# Load document chunks
chunks = load_chunks_pickle("chunks_dict.pkl")
all_chunks = []
for item in chunks:
    all_chunks += chunk_text(item['content'], chunk_size=1050, overlap=50)
```

### 2. Generate Embeddings & Upsert to Pinecone

```python
from utils.utils import upsert_chunks_to_pinecone

index_name = "taxwiz"
upsert_chunks_to_pinecone(all_chunks, index_name)
```

### 3. Semantic Search via API

- `POST` your query to the `/search/` endpoint
- Retrieve ranked results based on semantic similarity

---

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/awesome-feature`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature/awesome-feature`
5. Open a pull request

---

## License

**Developers Foundry AI/ML Track** © 2026 TaxWiz Team

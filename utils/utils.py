import os
import re
import numpy as np
import logging
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from prompts import promptFactory

# -------------------------
# Environment Setup
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LIGHT_MODEL = os.getenv("light_model")
HEAVY_MODEL = os.getenv("heavy_model")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Lazy Client Initializers (Best Practice)
# -------------------------

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def get_pinecone_client():
    return Pinecone(api_key=PINECONE_API_KEY)

def get_pinecone_index(index_name: str):
    pc = get_pinecone_client()
    return pc.Index(index_name)

# -------------------------
# Index Creation
# -------------------------

def create_index(name: str, dimension: int = 1536):
    pc = get_pinecone_client()

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if name not in existing_indexes:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info(f"Index '{name}' created.")
    else:
        logger.info(f"Index '{name}' already exists.")

# -------------------------
# Text Chunking
# -------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

# -------------------------
# OpenAI Embeddings
# -------------------------

def generate_openai_embeddings(text_chunks: List[str]) -> List[List[float]]:
    client = get_openai_client()
    embeddings = []

    for chunk in text_chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)

    return embeddings

# -------------------------
# Cosine Similarity
# -------------------------

def cosine_similarity_matrix(A, B, eps=1e-12):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)

    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Embedding dims must match: {A.shape[1]} vs {B.shape[1]}")

    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)

    return A_norm @ B_norm.T

# -------------------------
# Pinecone Upsert
# -------------------------

def upsert_chunks_to_pinecone(
    text_chunks: List[str],
    index_name: str,
    namespace: str = "__default__",
    batch_size: int = 100,
):
    index = get_pinecone_index(index_name)

    vectors = generate_openai_embeddings(text_chunks)
    total = len(text_chunks)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_records = []

        for i in range(batch_start, batch_end):
            batch_records.append(
                (
                    f"chunk-{i}",
                    vectors[i],
                    {"text": text_chunks[i]}
                )
            )

        index.upsert(
            vectors=batch_records,
            namespace=namespace
        )

        logger.info(
            f"Upserted chunks {batch_start} to {batch_end - 1} "
            f"into namespace '{namespace}'."
        )

# -------------------------
# HTML Extraction
# -------------------------

def extract_plain_text(text: str):
    match = re.search(r"```html\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text
import os
import re
import numpy as np
import logging
from prompts import promptFactory
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

OPENAI_API_KEY = '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'
pinecone_api_key="pcsk_5SdECw_9FYdBrMDUB934JZaYAvtkF5Ukcbrm2LW7hrQJgN8mTeR8rmY4TvEezqDWgViA65"

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

# Initialize Pinecone
# pinecone_api_key = os.environ.get('pinecone_api_key')
client = OpenAI(api_key = OPENAI_API_KEY)
sys_prompt = promptFactory.sys_prompt()
light_model = os.environ.get('light_model')
heavy_model = os.environ.get('heavy_model')
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(host='https://taxwiz2-368c1ea.svc.aped-4627-b74a.pinecone.io', name='taxwiz')
def create_index(name):
    # Check if the index already exists
    if name not in [k.get('name', None) for k in pc.list_indexes()]:
        # Create the index
        pc.create_index(
            name=name,
            dimension=1536,  # Set the correct dimension for your embeddings
            metric="cosine",  # Options: 'cosine', 'euclidean', 'dotproduct'
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{name}' created.")
    else:
        print(f"Index '{name}' already exists.")

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split `text` into chunks of `chunk_size` words each, with `overlap` words
    repeated from the end of one chunk to the beginning of the next.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words overlapping between chunks (0 <= overlap < chunk_size).

    Returns:
        list[str]: A list of text chunks (each as a single string).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be >= 0 and less than chunk_size.")

    words = text.split()
    total_words = len(words)
    chunks = []

    # We advance by chunk_size - overlap words each time
    step = chunk_size - overlap
    start = 0

    while start < total_words:
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def generate_openai_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for each text chunk using OpenAI's API.

    Args:
        text_chunks (List[str]): List of text chunks to embed.

    Returns:
        List[List[float]]: List of embedding vectors (each a list of floats).
    """
    embeddings = []
    for chunk in text_chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        # Extract the vector from the API response
        vector = response.data[0].embedding
        embeddings.append(vector)
    return embeddings

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors using np.dot
    
    Parameters:
    a, b : numpy arrays of same shape (n,)
    
    Returns:
    cosine similarity value between a and b
    """
    # Compute dot product
    dot_product = np.dot(a[0], b[0])
    
    # Compute norms (L2 norms) of each vector
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Compute cosine similarity
    cosine_sim = dot_product / (norm_a * norm_b)
    
    return cosine_sim

def cosine_similarity_matrix(A, B, eps=1e-12):
    """
    Pairwise cosine similarity between rows of A and rows of B.

    A: (m, d)
    B: (n, d)
    returns: (m, n)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)

    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Embedding dims must match. Got {A.shape[1]} and {B.shape[1]}")

    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)

    return A_norm @ B_norm.T  # (m,d) @ (d,n) -> (m,n)

def upsert_chunks_to_pinecone(
    text_chunks: List[str],
    index=index,
    namespace: str = "__default__",
    use_integrated_embedding: bool = False,
    batch_size: int = 100
) -> None:
    """
    Upsert a list of text chunks into a Pinecone index with optional integrated embedding.

    Args:
        text_chunks (List[str]): List of document chunks (strings).
        index (pinecone.Index): Initialized Pinecone index object.
        namespace (str): Pinecone namespace to upsert into. Defaults to "__default__".
        use_integrated_embedding (bool): If True, let Pinecone embed text; otherwise generate embeddings client-side.
        batch_size (int): Number of vectors to upsert per request (for batching).
    """

    total_chunks = len(text_chunks)
    # 1. If using client-side embeddings, pre-generate all vectors
    if not use_integrated_embedding:
        # Generate embeddings via OpenAI for each chunk
        vectors = generate_openai_embeddings(text_chunks)
    else:
        vectors = None  # Pinecone will embed server-side

    # 2. Prepare and upsert in batches
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunks = text_chunks[batch_start:batch_end]
        # 3. Build records for this batch
        records = []
        for i, chunk in enumerate(batch_chunks, start=batch_start):
            chunk_id = f"chunk-{i}"  # Unique ID for each chunk
            if use_integrated_embedding:
                # Let Pinecone generate the embedding from 'text'
                record = {
                    "id": chunk_id,
                    "values": None,          # No client-side vector
                    "metadata": {"text": chunk}
                }
            else:
                # Use precomputed embedding
                record = {
                    "id": chunk_id,
                    "values": vectors[i],    # The embedding vector from OpenAI
                    "metadata": {"text": chunk}
                }
            records.append(record)

        # 4. Upsert this batch into Pinecone
        if use_integrated_embedding:
            # For integrated embedding, call upsert_records
            index.upsert_records(records, namespace=namespace)
        else:
            # For client-side embeddings, call upsert with id and values
            index.upsert(
                vectors=[(r["id"], r["values"], r["metadata"]) for r in records],
                namespace=namespace
            )

        print(f"Upserted chunks {batch_start} to {batch_end - 1} into Pinecone (namespace: '{namespace}').")

def extract_plain_text(text):
    # Use regex to extract content between ```html and ```
    match = re.search(r"```html\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        # Strip surrounding whitespace and return plain text
        return match.group(1).strip()
    return text
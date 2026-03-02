from fastapi import APIRouter
from .schema import QueryRequest, QueryResponse

from utils.utils import (
    generate_openai_embeddings,
    get_pinecone_index
)

router = APIRouter()


@router.get("/")
def health_check():
    return {"status": "TaxWiz API running"}


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    # 1️⃣ Generate embedding
    embedding = generate_openai_embeddings([request.query])[0]

    # 2️⃣ Query Pinecone
    index = get_pinecone_index("taxwiz")

    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )

    matches = [
        match["metadata"]["text"]
        for match in results["matches"]
    ]

    return QueryResponse(
        query=request.query,
        retrieved_chunks=matches
    )
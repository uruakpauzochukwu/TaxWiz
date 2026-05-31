from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    retrieved_chunks: List[str]
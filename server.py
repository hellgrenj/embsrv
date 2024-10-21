import math
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

model_name = 'intfloat/e5-large-v2'
print(f"loading {model_name}")
model = SentenceTransformer(model_name)


class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1,
                      description="text must be at least 1 character.")


class NumberOfTokensRequest(BaseModel):
    text: str = Field(..., min_length=1,
                      description="text must be at least 1 character.")


class TruncateRequest(BaseModel):
    text: str = Field(..., min_length=1,
                      description="text must be at least 1 character.")
    max_tokens: int = Field(..., gt=0,
                            description="max_tokens must be greater than 0")


class TextInTokenSizedChunksRequest(BaseModel):
    text: str = Field(..., min_length=1,
                      description="text must be atleast 1 character.")
    max_tokens: int = Field(..., gt=0,
                            description="max_tokens must be greater than 0")


@app.get("/")
@app.get("/info")
def read_root():
    return {
        "model": model_name,
        "embedding size": 1024,
        "sequence limit": 512
    }


@app.post("/embeddings/encode/")
def get_emb_endpoint(request: EmbeddingRequest):
    emb = get_embedding(request.text)
    return {"embeddings": emb.tolist()}


@app.post("/text/truncate/")
def get_truncated_text_endpoint(request: TruncateRequest):
    truncated_text = truncate_text_to_max_tokens(request.text, request.max_tokens)
    return {"truncated_text": truncated_text}


@app.post("/text/number-of-tokens/")
def get_number_of_tokens_endpoint(request: NumberOfTokensRequest):
    number_of_tokens = num_tokens_from_string(request.text)
    return {"number_of_tokens": number_of_tokens}


@app.post("/text/chunks/")
def get_text_in_chunks_endpoint(request: TextInTokenSizedChunksRequest):
    text_in_chunks = text_chunks_by_token_size(request.text, request.max_tokens)
    return {"text_in_chunks": text_in_chunks}


def get_embedding(text: str):
    """Returns embedding for the provided text"""
    embeddings = model.encode(text)
    return embeddings


def truncate_text_to_max_tokens(text: str, max_tokens: int = 512) -> str:
    """Returns the provided text truncated to the provided max_tokens (default to 512)"""
    tokens = model.tokenizer.encode(text)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = model.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text


def text_chunks_by_token_size(text: str, token_size: int = 512) -> list:
    """Returns a list of chunks of the requested token_size (default to 512)"""
    tokens = model.tokenizer.encode(text)
    num_tokens = len(tokens)
    num_chunks = math.ceil(num_tokens / token_size)  # round up so we don't lose any tokens
    chunk_texts = []
    for i in range(num_chunks):
        chunk = tokens[i * token_size: (i + 1) * token_size]
        chunk_text = model.tokenizer.decode(chunk, skip_special_tokens=True)
        chunk_texts.append(chunk_text)
    return chunk_texts


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    tokens = model.tokenizer.tokenize(string)
    num_tokens = len(tokens)
    return num_tokens

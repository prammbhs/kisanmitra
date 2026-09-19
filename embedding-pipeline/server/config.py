import os
from dotenv import load_dotenv

load_dotenv()

# Server setup
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# AWS Bedrock & Model settings
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "True").lower() == "true"
BEDROCK_MAX_WORKERS = int(os.getenv("BEDROCK_MAX_WORKERS", "10"))

# Storage Settings (EBS ChromaDB)
CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/chroma")
ALLOWED_COLLECTIONS = ["kcc_docs", "other_docs"]
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "kcc_docs")

# Optional Auth
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Server setup
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Embedding Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
GPU_BATCH_SIZE = int(os.getenv("GPU_BATCH_SIZE", "32"))

# Device auto-detection
PREFERRED_DEVICE = os.getenv("DEVICE", "").lower()
if PREFERRED_DEVICE in ["cuda", "cpu"]:
    DEVICE = PREFERRED_DEVICE
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Storage Settings (EBS ChromaDB)
CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/chroma")
ALLOWED_COLLECTIONS = ["kcc_docs", "other_docs"]
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "kcc_docs")

# Optional Auth
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

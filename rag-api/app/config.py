import os
from dotenv import load_dotenv

load_dotenv()

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
RAG_API_KEY = os.getenv("RAG_API_KEY", "")  # empty = auth disabled

# --- Chroma (EBS mount) ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/chroma")
COLLECTIONS = ["kcc_docs"]
DEFAULT_COLLECTIONS = os.getenv("DEFAULT_COLLECTIONS", "kcc_docs").split(",")

# --- App Database (SQLite on EBS for user profiles & LangGraph threads) ---
APP_DB_PATH = os.getenv("APP_DB_PATH", os.path.join(CHROMA_PATH, "kisanmitra_app.db"))
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", os.path.join(CHROMA_PATH, "chat_checkpoints.sqlite3"))

# --- JWT Authentication ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "kisanmitra-super-secret-jwt-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "43200"))  # 30 days

# --- Embeddings: MUST match ingestion (voyage-4-lite, 1024d) ---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL_ID = os.getenv("VOYAGE_MODEL_ID", "voyage-4-lite")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# --- LLM: Fireworks ---
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/glm-5p3-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.0"))  # cosine similarity floor (0..1)

import os
from dotenv import load_dotenv

load_dotenv()

# Embedding API configuration
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8000")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

# Batch & Retry settings
HTTP_BATCH_SIZE = int(os.getenv("HTTP_BATCH_SIZE", "128"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
INITIAL_BACKOFF = float(os.getenv("INITIAL_BACKOFF", "2.0"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300.0"))

# Checkpoint settings
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "../checkpoints/progress.json")

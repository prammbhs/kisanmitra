import time
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from server.config import MODEL_NAME, DEVICE, GPU_BATCH_SIZE, USE_FP16, ATTN_IMPLEMENTATION

class EmbeddingModelManager:
    _instance = None

    def __init__(self):
        print(f"[INIT] Loading embedding model '{MODEL_NAME}' on device '{DEVICE}'...")
        start_t = time.time()
        self.device = DEVICE
        self.gpu_batch_size = GPU_BATCH_SIZE
        
        model_kwargs = {}
        if self.device == "cuda":
            # Enable TF32 for NVIDIA Tensor Core GPU acceleration
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable PyTorch 2.x Scaled Dot Product Attention (SDPA) or FlashAttention-2
            if ATTN_IMPLEMENTATION:
                model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
                print(f"[INIT] Enabling attention implementation: '{ATTN_IMPLEMENTATION}'")

        # SentenceTransformer loads BAAI/bge-m3 with optimized model_kwargs
        try:
            self.model = SentenceTransformer(
                MODEL_NAME,
                device=self.device,
                model_kwargs=model_kwargs if model_kwargs else None
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize with model_kwargs={model_kwargs}: {e}. Falling back to default loader.")
            self.model = SentenceTransformer(MODEL_NAME, device=self.device)

        elapsed = time.time() - start_t
        print(f"[INIT] Model successfully loaded in {elapsed:.2f}s on {self.device}.")

    @classmethod
    def get_instance(cls) -> "EmbeddingModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_texts(self, texts: List[str], device: Optional[str] = None) -> List[List[float]]:
        """
        Encode a list of text strings into 1024-dim embeddings using BGE-M3.
        """
        if not texts:
            return []

        target_device = device if device else self.device

        # Sanitize text input to valid utf-8 strings
        clean_texts = [
            t if isinstance(t, str) else str(t)
            for t in texts
        ]

        use_fp16 = USE_FP16 if target_device == "cuda" else False

        with torch.inference_mode():
            embeddings = self.model.encode(
                clean_texts,
                device=target_device,
                batch_size=self.gpu_batch_size if target_device != "cpu" else 16,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                precision="fp16" if use_fp16 else "float32"
            )

        dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        if dim != 1024:
            raise ValueError(f"Expected embedding dimension 1024 from BGE-M3, but got {dim}")

        return embeddings.tolist()

    def encode_query(self, query: str, device: str = "cpu") -> List[float]:
        """
        Encode a single query string into a 1024-dim embedding vector in CPU mode.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        res = self.encode_texts([query], device=device)
        return res[0]



import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env at import time
load_dotenv()


def get_env(name: str, default: str | None = None) -> str:
	value = os.getenv(name, default)
	if value is None or value == "":
		raise RuntimeError(f"Missing required environment variable: {name}")
	return value


@dataclass(frozen=True)
class Settings:
	openrouter_api_key: str = get_env("OPENROUTER_API_KEY")
	hf_token: str = get_env("HF_TOKEN")
	lancedb_path: str = os.getenv("LANCEDB_PATH", "./lancedb_data")
	lancedb_table: str = os.getenv("LANCEDB_TABLE", "user_guide")

	embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "google/embeddinggemma-300m")
	reranker_model_id: str = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
	llm_model_id: str = os.getenv("LLM_MODEL_ID", "qwen/qwen3-30b-a3b-instruct-2507")
	openrouter_endpoint: str = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")

	device: str = os.getenv("DEVICE", "cuda")
	batch_size_embed: int = int(os.getenv("BATCH_SIZE_EMBED", "64"))
	batch_size_rerank: int = int(os.getenv("BATCH_SIZE_RERANK", "16"))

	chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "1000"))
	chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "200"))


settings = Settings()

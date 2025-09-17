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
	# Tavily
	tavily_api_key: str = get_env("TAVILY_API_KEY")
	lancedb_path: str = os.getenv("LANCEDB_PATH", "./lancedb_data")
	lancedb_table: str = os.getenv("LANCEDB_TABLE", "user_guide")

	embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "google/embeddinggemma-300m")
	reranker_model_id: str = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
	llm_model_id: str = os.getenv("LLM_MODEL_ID", "qwen/qwen3-30b-a3b-instruct-2507")
	openrouter_endpoint: str = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")

	memory_enabled: bool = os.getenv("MEMORY_ENABLED", "true").lower() in ("1", "true", "yes")
	memory_model_id: str = os.getenv("MEMORY_MODEL_ID", "qwen/qwen3-30b-a3b-instruct-2507")
	memory_temperature: float = float(os.getenv("MEMORY_TEMPERATURE", "0.2"))
	memory_max_tokens: int = int(os.getenv("MEMORY_MAX_TOKENS", "2000"))
	memory_search_limit: int = int(os.getenv("MEMORY_SEARCH_LIMIT", "3"))
	memory_default_user_id: str = os.getenv("MEMORY_DEFAULT_USER_ID", "alma")
	memory_category: str = os.getenv("MEMORY_CATEGORY", "general")
	memory_site_url: str | None = os.getenv("MEMORY_SITE_URL")
	memory_app_name: str | None = os.getenv("MEMORY_APP_NAME")
	# Mem0 embedder & vector store настройки
	memory_embedder_provider: str = os.getenv("MEMORY_EMBEDDER_PROVIDER", "huggingface")
	memory_embedder_model_id: str = os.getenv("MEMORY_EMBEDDER_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
	memory_vector_store_provider: str = os.getenv("MEMORY_VECTOR_STORE_PROVIDER", "faiss")
	# Текстовое руководство для модели по использованию memories (на случай, если понадобится)
	memory_prompt_guidance: str | None = os.getenv("MEMORY_PROMPT_GUIDANCE")
	# Директория для локального индекса FAISS (используется Mem0 vector_store)
	memory_faiss_path: str = os.getenv("MEMORY_FAISS_PATH", "./mem0_data/faiss")
	# Размерность эмбеддингов для Mem0 vector_store (должна совпадать с эмбеддером)
	memory_embedding_dims: int = int(os.getenv("MEMORY_EMBEDDING_DIMS", "384"))
	# Параметры метрики для FAISS в Mem0: 'euclidean' | 'inner_product' | 'cosine'
	memory_vector_distance_strategy: str = os.getenv("MEMORY_VECTOR_DISTANCE", "cosine")
	# Нормализация L2 для косинусной близости
	memory_vector_normalize_l2: bool = os.getenv("MEMORY_VECTOR_NORMALIZE_L2", "true").lower() in ("1", "true", "yes")



	device: str = os.getenv("DEVICE", "cuda")
	batch_size_embed: int = int(os.getenv("BATCH_SIZE_EMBED", "64"))
	batch_size_rerank: int = int(os.getenv("BATCH_SIZE_RERANK", "16"))

	chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "1000"))
	chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "200"))
	# Ingest controls
	prefer_docling_api: bool = os.getenv("PREFER_DOCLING_API", "true").lower() in ("1", "true", "yes")
	enable_cli_fallback: bool = os.getenv("ENABLE_CLI_FALLBACK", "true").lower() in ("1", "true", "yes")
	enable_ocr: bool = os.getenv("ENABLE_OCR", "true").lower() in ("1", "true", "yes")
	preserve_case: bool = os.getenv("PRESERVE_CASE", "true").lower() in ("1", "true", "yes")
	strip_headers_footers: bool = os.getenv("STRIP_HEADERS_FOOTERS", "true").lower() in ("1", "true", "yes")
	merge_hyphenation: bool = os.getenv("MERGE_HYPHENATION", "true").lower() in ("1", "true", "yes")
	table_mode: str = os.getenv("TABLE_MODE", "md")  # md|html|csv (пока влияет только на метаданные)
	max_parallel_pages: int = int(os.getenv("MAX_PARALLEL_PAGES", "4"))
	allowed_exts: str = os.getenv("ALLOWED_EXTS", ".pdf,.docx,.pptx,.xlsx,.html,.htm,.txt")

	# Rerank meta-weights
	rerank_bonus_heading: float = float(os.getenv("RERANK_BONUS_HEADING", "0.08"))
	rerank_bonus_table: float = float(os.getenv("RERANK_BONUS_TABLE", "0.05"))
	rerank_bonus_code: float = float(os.getenv("RERANK_BONUS_CODE", "0.05"))
	rerank_bonus_list: float = float(os.getenv("RERANK_BONUS_LIST", "0.02"))
	rerank_bonus_math: float = float(os.getenv("RERANK_BONUS_MATH", "0.04"))
	rerank_bonus_paragraph: float = float(os.getenv("RERANK_BONUS_PARAGRAPH", "0.0"))
	rerank_section_depth_penalty: float = float(os.getenv("RERANK_SECTION_DEPTH_PENALTY", "0.01"))
	rerank_max_meta_bonus: float = float(os.getenv("RERANK_MAX_META_BONUS", "0.15"))

	# По умолчанию количество кандидатов для поиска в БД
	default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "200"))

	# Web fallback configuration
	enable_web_fallback: bool = os.getenv("ENABLE_WEB_FALLBACK", "true").lower() in ("1", "true", "yes")
	manual_min_rerank_score: float = float(os.getenv("MANUAL_MIN_RERANK_SCORE", "0.35"))
	manual_min_context_tokens: int = int(os.getenv("MANUAL_MIN_CONTEXT_TOKENS", "120"))
	web_search_max_results: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "10"))
	web_fetch_top_n: int = int(os.getenv("WEB_FETCH_TOP_N", "3"))
	web_context_max_tokens: int = int(os.getenv("WEB_CONTEXT_MAX_TOKENS", "3200"))
	# Параметры Tavily поиска
	web_search_query_prefix: str = os.getenv(
		"WEB_SEARCH_QUERY_PREFIX",
		"Ты — ведущий инженер-аналитик в нефтегазовой отрасли."
	)
	web_search_depth: str = os.getenv("WEB_SEARCH_DEPTH", "advanced")
	web_search_include_raw_content: str = os.getenv("WEB_SEARCH_INCLUDE_RAW_CONTENT", "markdown")
	web_search_chunks_per_source: int = int(os.getenv("WEB_SEARCH_CHUNKS_PER_SOURCE", "5"))

	# DuckDuckGo настройки удалены; используем Tavily API

	# HTTP-запросы к веб‑страницам
	web_http_timeout_seconds: float = float(os.getenv("WEB_HTTP_TIMEOUT", "10.0"))
	web_http_verify_tls: bool = os.getenv("WEB_HTTP_VERIFY_TLS", "true").lower() in ("1", "true", "yes")
	web_user_agent: str = os.getenv(
		"WEB_USER_AGENT",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
		"AppleWebKit/537.36 (KHTML, like Gecko) "
		"Chrome/124.0 Safari/537.36",
	)
	# Максимальная длина извлекаемого текста со страницы (символы)
	web_fetch_max_chars: int = int(os.getenv("WEB_FETCH_MAX_CHARS", "4000"))

	# Фильтрация результатов веб-поиска
	web_search_filter_enabled: bool = os.getenv("WEB_SEARCH_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")
	web_search_stopwords_ru: str = os.getenv(
		"WEB_SEARCH_STOPWORDS_RU",
		"что,такое,кто,где,зачем,как,почему,это,есть,и,или,в,на,по,за,из,для,про,от,до,над,под,с,со,об,обо,у,же,ли",
	)
	web_search_min_token_len: int = int(os.getenv("WEB_SEARCH_MIN_TOKEN_LEN", "3"))
	web_search_required_token_matches: int = int(os.getenv("WEB_SEARCH_REQUIRED_TOKEN_MATCHES", "1"))


settings = Settings()

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import RAGPipeline
from rag.config import settings


def iter_documents(root: Path):
	allowed = {ext.strip().lower() for ext in settings.allowed_exts.split(',') if ext.strip()}
	if root.is_file():
		if root.suffix.lower() in allowed:
			yield root
		return
	for p in root.rglob('*'):
		if p.is_file() and p.suffix.lower() in allowed:
			yield p


def main() -> int:
	if len(sys.argv) < 2:
		print("Usage: python scripts/ingest_path.py <file-or-dir>", file=sys.stderr)
		return 1
	path = Path(sys.argv[1])
	if not path.exists():
		print(f"Path not found: {path}", file=sys.stderr)
		return 1
	pipe = RAGPipeline()
	total = 0
	for doc in iter_documents(path):
		count = pipe.ingest_pdf(str(doc))
		print(f"Ingested {count} chunks from {doc}")
		total += count
	print(f"Total chunks ingested: {total}")
	return 0


if __name__ == "__main__":
	sys.exit(main())

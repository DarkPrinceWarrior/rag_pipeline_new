from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import RAGPipeline


def main() -> int:
	pdf_path = Path("docs") / "User_Guide.pdf"
	if not pdf_path.exists():
		print(f"PDF not found: {pdf_path}", file=sys.stderr)
		return 1
	pipe = RAGPipeline()
	count = pipe.ingest_pdf(str(pdf_path))
	print(f"Ingested {count} chunks from {pdf_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
